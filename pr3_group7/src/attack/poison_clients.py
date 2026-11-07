"""
Creates label-flipped (poisoned) copies of selected client datasets.

The poisoned datasets are written to the file system and loaded directly by the FL simulation.

Attack model (Group 7, Table 1 of the spec):
  * poisoned clients : 70% of the 20 clients (14 clients) by default
  * flipping rates   : 0%, 25%, 50%, 100% of each malicious client's labels

Label flipping rule: a flipped label y becomes (y + 1) mod 7, so every
flipped label is guaranteed to be wrong (akiec->bcc, ..., nv->vasc,
vasc->akiec). Both the local TRAIN and the local VALIDATION labels of a
malicious client are flipped, because the spec requires each client to be
evaluated on its own (poisoned) local subset. The global held-out test set
is never touched.

Two choices make the different attack strengths directly comparable:
  * Malicious clients are taken from one fixed random ordering of the
    clients, so the 10% set is inside the 30% set, which is inside the
    50% set, and so on.
  * Within a client, the flipped samples are taken from one fixed random
    ordering, so the 25% flipped samples are inside the 50% set, which is
    inside the 100% set.

Output, e.g. data/poisoned/p70_flip50/:
  client_XX.npz        poisoned copies of the malicious clients only
                       (benign clients are read from data/clients/)
  scenario.json        attack settings and the malicious client ids
  poison_summary.csv   per malicious client: how many labels were flipped

@author Amruta Kirtikumar Khot, ak3792@rit.edu
@author Cazal Tirkey, ct9857@rit.edu
"""

import argparse
import json
import os

import numpy as np
import pandas as pd


NUMBER_OF_CLASSES = 7


def scenario_name(poison_fraction, flip_rate):
    """
    Build the folder/tag name for one attack scenario.
    :param poison_fraction: fraction of clients that are malicious (0..1)
    :param flip_rate: fraction of labels flipped on each malicious client (0..1)
    :return: string such as "p70_flip50"
    """
    return "p" + str(int(round(poison_fraction * 100))) + "_flip" + str(int(round(flip_rate * 100)))


def choose_malicious_clients(number_of_clients, poison_fraction, seed):
    """
    Pick the malicious client ids.
    The same seed always gives the same client ordering, so smaller attack
    sets are subsets of larger ones.
    :param number_of_clients: total number of clients
    :param poison_fraction: fraction of clients to poison (0..1)
    :param seed: random seed
    :return: sorted list of malicious client ids
    """
    random_generator = np.random.default_rng(seed)
    order = random_generator.permutation(number_of_clients)
    number_malicious = int(round(poison_fraction * number_of_clients))
    return sorted(int(c) for c in order[:number_malicious])


def flip_labels(labels, flip_rate, random_generator):
    """
    Flip a fraction of the labels to (y + 1) mod 7.
    :param labels: int array of clean labels
    :param flip_rate: fraction of labels to flip (0..1)
    :param random_generator: numpy Generator (fixed per client)
    :return: (flipped label array, number of labels flipped)
    """
    flipped = labels.copy()
    order = random_generator.permutation(len(labels))
    number_to_flip = int(round(flip_rate * len(labels)))
    chosen = order[:number_to_flip]
    flipped[chosen] = (flipped[chosen] + 1) % NUMBER_OF_CLASSES
    return flipped, number_to_flip


def create_scenario(clients_dir, out_root, number_of_clients, poison_fraction, flip_rate, seed):
    """
    Write the poisoned client files for one attack scenario.
    :param clients_dir: folder with the clean client_XX.npz files
    :param out_root: parent folder for poisoned scenarios
    :param number_of_clients: total number of clients
    :param poison_fraction: fraction of clients that are malicious
    :param flip_rate: fraction of labels flipped per malicious client
    :param seed: random seed
    :return: path of the scenario folder
    """
    name = scenario_name(poison_fraction, flip_rate)
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)

   
    for old_file in os.listdir(out_dir):
        if old_file.startswith("client_") and old_file.endswith(".npz"):
            os.remove(os.path.join(out_dir, old_file))

    malicious = choose_malicious_clients(number_of_clients, poison_fraction, seed)
    summary_rows = []

    for client_id in malicious:
        filename = "client_" + format(client_id, "02d") + ".npz"
        clean = np.load(os.path.join(clients_dir, filename), allow_pickle=False)

        train_rng = np.random.default_rng(seed * 1000 + client_id * 2)
        val_rng = np.random.default_rng(seed * 1000 + client_id * 2 + 1)

        y_train, n_train_flipped = flip_labels(clean["y_train"], flip_rate, train_rng)
        y_val, n_val_flipped = flip_labels(clean["y_val"], flip_rate, val_rng)

        assert int((y_train != clean["y_train"]).sum()) == n_train_flipped
        assert int((y_val != clean["y_val"]).sum()) == n_val_flipped

        np.savez_compressed(
            os.path.join(out_dir, filename),
            x_train=clean["x_train"],
            y_train=y_train,
            x_val=clean["x_val"],
            y_val=y_val,
            y_train_clean=clean["y_train"],
            y_val_clean=clean["y_val"],
            lesion_train=clean["lesion_train"],
            lesion_val=clean["lesion_val"],
            classes=clean["classes"],
            is_malicious=np.array(True),
        )

        summary_rows.append({
            "client": client_id,
            "n_train": len(y_train),
            "n_train_flipped": n_train_flipped,
            "n_val": len(y_val),
            "n_val_flipped": n_val_flipped,
            "train_flip_fraction": round(n_train_flipped / max(1, len(y_train)), 4),
            "val_flip_fraction": round(n_val_flipped / max(1, len(y_val)), 4),
        })

    pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, "poison_summary.csv"), index=False)

    scenario = {
        "name": name,
        "number_of_clients": number_of_clients,
        "poison_fraction": poison_fraction,
        "flip_rate": flip_rate,
        "malicious_clients": malicious,
        "number_malicious": len(malicious),
        "flip_rule": "y -> (y + 1) mod 7, applied to train AND local validation labels",
        "seed": seed,
    }
    with open(os.path.join(out_dir, "scenario.json"), "w", encoding="utf-8") as file:
        json.dump(scenario, file, indent=2)

    print(name + ": malicious clients " + str(malicious))
    return out_dir


def main():
    
    project_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    parser = argparse.ArgumentParser(description="Create label-flipped client datasets.")
    parser.add_argument("--clients-dir", default=os.path.join(project_folder, "data", "clients"))
    parser.add_argument("--out-root", default=os.path.join(project_folder, "data", "poisoned"))
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--poison-frac", type=float, default=0.70)
    parser.add_argument("--flip-rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    create_scenario(
        args.clients_dir,
        args.out_root,
        args.num_clients,
        args.poison_frac,
        args.flip_rate,
        args.seed,
    )


if __name__ == "__main__":
    main()
