"""
Splits the HAM10000 pool into federated clients.

Two strategies are supported: IID (baseline) and non-IID Dirichlet
(main setup). Lesions are never split across clients, and each client
also gets its own local train/val split.

@author Amruta Kirtikumar Khot, ak3792@rit.edu
@author Cazal Tirkey, ct9857@rit.edu
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def group_images_by_lesion(lesion_ids, labels):
    """
    Group pool rows by lesion_id.
    Returns a list of (lesion_id, class_number, image_indices) tuples.
    Raises if the same lesion shows up with two different labels.

    :param lesion_ids: array of lesion_id strings from pool.npz
    :param labels: array of numeric labels from pool.npz
    :return: list of (lesion_id, class_number, image_indices) tuples
    """
    groups = {}

    for image_index in range(len(lesion_ids)):
        lesion_id = str(lesion_ids[image_index])

        if lesion_id not in groups:
            groups[lesion_id] = []

        groups[lesion_id].append(image_index)

    lesion_records = []

    for lesion_id in groups:
        image_indices = np.array(groups[lesion_id], dtype=int)
        lesion_labels = np.unique(labels[image_indices])

        if len(lesion_labels) != 1:
            raise ValueError(
                "The same lesion has more than one class label: " + lesion_id
            )

        class_number = int(lesion_labels[0])
        lesion_records.append((lesion_id, class_number, image_indices))

    return lesion_records


def create_iid_split(lesion_records, number_of_clients, random_generator):
    """
    Assign lesions to clients in a round-robin way.
    Lesion records are shuffled first, then dealt out one at a time, so
    each client ends up with roughly the same mix of classes.

    :param lesion_records: output of group_images_by_lesion
    :param number_of_clients: how many clients to create
    :param random_generator: numpy Generator for reproducibility
    :return: list of lists, one list of record indices per client
    """
    shuffled_record_numbers = random_generator.permutation(len(lesion_records))
    client_records = []

    for _ in range(number_of_clients):
        client_records.append([])

    for position in range(len(shuffled_record_numbers)):
        record_number = int(shuffled_record_numbers[position])
        client_number = position % number_of_clients
        client_records[client_number].append(record_number)

    return client_records


def create_non_iid_split(
    lesion_records,
    number_of_clients,
    alpha,
    minimum_images,
    random_generator,
):
    """
    Build a label-skewed non-IID split with a Dirichlet distribution.

    For each class we draw Dirichlet probabilities over the clients and
    use them to decide who gets each lesion. Smaller clients are slightly
    favoured so no single client ends up huge. Whole lesions are kept
    together. The draw is retried up to 500 times if any client ends up
    below minimum_images.

    :param lesion_records: output of group_images_by_lesion
    :param number_of_clients: how many clients to create
    :param alpha: Dirichlet concentration, lower = more skew
    :param minimum_images: smallest client size we accept
    :param random_generator: numpy Generator for reproducibility
    :return: list of lists, one list of record indices per client
    """
    lesion_classes = np.array(
        [record[1] for record in lesion_records],
        dtype=int,
    )

    lesion_image_counts = np.array(
        [len(record[2]) for record in lesion_records],
        dtype=int,
    )

    last_client_sizes = None

    # A random Dirichlet split can occasionally create a tiny client.
    # Retry until every client has at least minimum images.
    for _ in range(500):
        client_records = []
        for _client in range(number_of_clients):
            client_records.append([])

        client_image_counts = np.zeros(number_of_clients, dtype=float)

        for class_number in range(len(CLASSES)):
            record_numbers = np.where(lesion_classes == class_number)[0]

            if len(record_numbers) == 0:
                continue

            record_numbers = record_numbers.copy()
            random_generator.shuffle(record_numbers)

            # Example: the 20 probabilities tell us how strongly this class
            # should be represented on each of the 20 clients.
            probabilities = random_generator.dirichlet(
                np.full(number_of_clients, alpha, dtype=float)
            )

            for record_number in record_numbers:
                # Keep the desired non-IID probabilities, but slightly prefer
                # smaller clients so one client does not become extremely large.
                adjusted_probabilities = probabilities / (1.0 + client_image_counts)
                adjusted_probabilities = adjusted_probabilities / adjusted_probabilities.sum()

                client_number = int(
                    random_generator.choice(
                        number_of_clients,
                        p=adjusted_probabilities,
                    )
                )

                client_records[client_number].append(int(record_number))
                client_image_counts[client_number] += lesion_image_counts[record_number]

        last_client_sizes = client_image_counts.astype(int)

        if last_client_sizes.min() >= minimum_images:
            return client_records

    raise RuntimeError(
        "Could not create a valid non-IID split. "
        "Try increasing --alpha or lowering --min-size. "
        "Smallest client had " + str(int(last_client_sizes.min())) + " images."
    )


def record_numbers_to_image_indices(lesion_records, selected_record_numbers):
    """
    Turn selected lesion records into the pool row indices they cover.

    :param lesion_records: output of group_images_by_lesion
    :param selected_record_numbers: record indices to expand
    :return: sorted int array of pool row indices
    """
    image_indices = []

    for record_number in selected_record_numbers:
        record_indices = lesion_records[record_number][2]

        for image_index in record_indices:
            image_indices.append(int(image_index))

    image_indices.sort()
    return np.array(image_indices, dtype=int)


def split_client_train_validation(
    client_lesion_ids,
    client_labels,
    validation_fraction,
    random_generator,
):
    """
    Per-client train/val split that keeps each lesion whole.

    We hold out a fraction of the lesions per class. If the client is too
    small to produce any validation data, we move one whole lesion over,
    and we always leave at least one lesion for training.

    :param client_lesion_ids: lesion_id strings for this client
    :param client_labels: numeric labels for this client
    :param validation_fraction: fraction of lesions per class to hold out
    :param random_generator: numpy Generator for reproducibility
    :return: (train_mask, validation_mask) boolean arrays
    """
    unique_lesion_ids = np.unique(client_lesion_ids)
    validation_lesions = set()

    for class_number in range(len(CLASSES)):
        lesions_in_class = []

        for lesion_id in unique_lesion_ids:
            positions = np.where(client_lesion_ids == lesion_id)[0]
            first_position = positions[0]

            if int(client_labels[first_position]) == class_number:
                lesions_in_class.append(lesion_id)

        if len(lesions_in_class) <= 1:
            continue

        lesions_in_class = np.array(lesions_in_class)
        random_generator.shuffle(lesions_in_class)

        number_for_validation = int(
            round(validation_fraction * len(lesions_in_class))
        )
        number_for_validation = max(1, number_for_validation)
        number_for_validation = min(
            number_for_validation,
            len(lesions_in_class) - 1,
        )

        for lesion_id in lesions_in_class[:number_for_validation]:
            validation_lesions.add(str(lesion_id))

    validation_mask = []

    for lesion_id in client_lesion_ids:
        validation_mask.append(str(lesion_id) in validation_lesions)

    validation_mask = np.array(validation_mask, dtype=bool)

    # Very small clients/classes can sometimes produce no validation data.
    # In that case, move one complete lesion to validation.
    if not validation_mask.any() and len(unique_lesion_ids) > 1:
        selected_lesion = random_generator.choice(unique_lesion_ids)
        validation_mask = client_lesion_ids == selected_lesion

    # Also ensure at least one lesion remains for training.
    if validation_mask.all() and len(unique_lesion_ids) > 1:
        selected_lesion = unique_lesion_ids[0]
        validation_mask[client_lesion_ids == selected_lesion] = False

    training_mask = ~validation_mask
    return training_mask, validation_mask


def save_partition_heatmap(summary, output_folder, number_of_clients):
    """
    Save the partition_heatmap.png used in the report.
    Rows are clients, columns are the 7 classes, cell values are the
    number of training images. The heatmap is what we use to show the
    non-IID skew.

    :param summary: per-client DataFrame built in main()
    :param output_folder: folder to write the PNG into
    :param number_of_clients: used in the plot title
    :return: None
    """
    class_columns = []
    for class_name in CLASSES:
        class_columns.append("train_" + class_name)

    matrix = summary[class_columns].to_numpy()

    figure_height = 0.32 * len(matrix) + 1.8
    figure, axis = plt.subplots(figsize=(7, figure_height))

    image = axis.imshow(matrix, aspect="auto")
    axis.set_xticks(range(len(CLASSES)))
    axis.set_xticklabels(CLASSES)
    axis.set_yticks(range(len(matrix)))

    y_labels = []
    for client_number in summary["client"]:
        y_labels.append("c" + str(client_number))
    axis.set_yticklabels(y_labels)

    axis.set_xlabel("Class")
    axis.set_ylabel("Client")
    axis.set_title(
        "HAM10000 training distribution across "
        + str(number_of_clients)
        + " clients"
    )

    figure.colorbar(image, ax=axis, label="Number of training images")
    figure.tight_layout()
    figure.savefig(
        os.path.join(output_folder, "partition_heatmap.png"),
        dpi=180,
    )
    plt.close(figure)


def main():
    script_folder = os.path.dirname(os.path.abspath(__file__))
    project_folder = os.path.dirname(os.path.dirname(script_folder))

    default_pool = os.path.join(
        project_folder, "data", "processed", "pool.npz"
    )
    default_output = os.path.join(project_folder, "data", "clients")

    parser = argparse.ArgumentParser(description="Create lesion-safe HAM10000 FL clients.")
    parser.add_argument("--pool", default=default_pool)
    parser.add_argument("--out", default=default_output)
    parser.add_argument("--clients", type=int, default=20)
    parser.add_argument("--strategy", choices=["iid", "dirichlet"], default="dirichlet")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--val-frac", type=float, default=0.20)
    parser.add_argument("--min-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.isfile(args.pool):
        raise SystemExit(
            "pool.npz was not found. Run prepare_ham10000.py first.\n"
            "Expected file:\n" + args.pool
        )

    random_generator = np.random.default_rng(args.seed)

    # 1. Load the prepared training pool
    data = np.load(args.pool, allow_pickle=False)
    images = data["x"]
    labels = data["y"]
    lesion_ids = data["lesion_id"]

    lesion_records = group_images_by_lesion(lesion_ids, labels)

    if args.clients < 2:
        raise SystemExit("At least 2 clients are required")

    if args.clients > len(lesion_records):
        raise SystemExit("More clients were requested than there are unique lesions")

    # 2. Assign complete lesions to clients
    if args.strategy == "iid":
        assignments = create_iid_split(
            lesion_records,
            args.clients,
            random_generator,
        )
    else:
        assignments = create_non_iid_split(
            lesion_records,
            args.clients,
            args.alpha,
            args.min_size,
            random_generator,
        )

    os.makedirs(args.out, exist_ok=True)

    summary_rows = []
    lesion_to_client = {}

    # 3. Build and save each client's local dataset
    for client_number in range(args.clients):
        selected_records = assignments[client_number]
        image_indices = record_numbers_to_image_indices(
            lesion_records,
            selected_records,
        )

        client_images = images[image_indices]
        client_labels = labels[image_indices]
        client_lesions = lesion_ids[image_indices]

        # Check that the same lesion has not already been used by another client.
        unique_client_lesions = np.unique(client_lesions)
        for lesion_id in unique_client_lesions:
            lesion_key = str(lesion_id)

            if lesion_key in lesion_to_client:
                raise AssertionError(
                    "Lesion " + lesion_key + " was assigned to multiple clients"
                )

            lesion_to_client[lesion_key] = client_number

        # Create this client's local training and validation subsets.
        train_mask, validation_mask = split_client_train_validation(
            client_lesions,
            client_labels,
            args.val_frac,
            random_generator,
        )

        x_train = client_images[train_mask]
        y_train = client_labels[train_mask]
        lesion_train = client_lesions[train_mask]

        x_validation = client_images[validation_mask]
        y_validation = client_labels[validation_mask]
        lesion_validation = client_lesions[validation_mask]

        # Final leakage check for the client's local train/validation split.
        train_lesion_set = set(str(x) for x in np.unique(lesion_train))
        validation_lesion_set = set(str(x) for x in np.unique(lesion_validation))

        if train_lesion_set.intersection(validation_lesion_set):
            raise AssertionError(
                "Client "
                + str(client_number)
                + " has lesion leakage between train and validation"
            )

        np.savez_compressed(
            os.path.join(
                args.out,
                "client_" + format(client_number, "02d") + ".npz",
            ),
            x_train=x_train,
            y_train=y_train,
            x_val=x_validation,
            y_val=y_validation,
            lesion_train=lesion_train,
            lesion_val=lesion_validation,
            classes=np.array(CLASSES),
        )

        # Save one summary row for this client.
        row = {
            "client": client_number,
            "n_train": len(y_train),
            "n_val": len(y_validation),
            "n_unique_lesions": len(unique_client_lesions),
        }

        for class_number in range(len(CLASSES)):
            class_name = CLASSES[class_number]
            row["train_" + class_name] = int(
                (y_train == class_number).sum()
            )

        summary_rows.append(row)

    # Every lesion should have been assigned exactly once.
    if len(lesion_to_client) != len(lesion_records):
        raise AssertionError("Not every lesion was assigned exactly once")

    # 4. Save audit files and the client-distribution heatmap
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        os.path.join(args.out, "partition_summary.csv"),
        index=False,
    )

    config = {
        "pool": args.pool,
        "out": args.out,
        "clients": args.clients,
        "strategy": args.strategy,
        "alpha": args.alpha,
        "validation_fraction": args.val_frac,
        "minimum_client_images": args.min_size,
        "seed": args.seed,
    }

    with open(
        os.path.join(args.out, "partition_config.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config, file, indent=2)

    mapping_rows = []
    for lesion_id in sorted(lesion_to_client):
        mapping_rows.append({
            "lesion_id": lesion_id,
            "client": lesion_to_client[lesion_id],
        })

    pd.DataFrame(mapping_rows).to_csv(
        os.path.join(args.out, "lesion_client_map.csv"),
        index=False,
    )

    save_partition_heatmap(summary, args.out, args.clients)

    print(summary.to_string(index=False))
    print("\nUnique lesions assigned:", len(lesion_to_client))
    print("Cross-client lesion leakage: 0")
    print("Wrote partition_heatmap.png")


if __name__ == "__main__":
    main()
