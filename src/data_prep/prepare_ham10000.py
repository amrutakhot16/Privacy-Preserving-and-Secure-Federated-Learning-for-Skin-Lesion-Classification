"""
Prepares HAM10000 for the federated learning project: matches metadata to
image files, removes duplicate images, splits by lesion into pool and test
sets, resizes to 64x64, and saves everything as npz + a few research paper
figures.

@author Amruta Kirtikumar Khot, ak3792@rit.edu
@author Cazal Tirkey, ct9857@rit.edu
"""

import argparse
import hashlib
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_NUMBER = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6,
}


def find_image_files(image_folders):
    """
    Return {image_id: full_path} for every image under the given folders.
    :param image_folders: list of folder paths to search
    :return: dict mapping image_id (str) to absolute file path (str)
    """
    image_paths = {}

    for folder in image_folders:
        if not os.path.isdir(folder):
            print("Warning: folder not found:", folder)
            continue

        for root, _, files in os.walk(folder):
            for filename in files:
                lower_name = filename.lower()
                if lower_name.endswith(".jpg") or lower_name.endswith(".jpeg") or lower_name.endswith(".png"):
                    image_id = os.path.splitext(filename)[0]
                    image_paths[image_id] = os.path.join(root, filename)

    return image_paths


def file_md5(path):
    """
    MD5 fingerprint, used only to catch byte identical duplicate images.
    :param path: path to the file to hash
    :return: hex digest string of the MD5 hash
    """
    md5_hash = hashlib.md5()

    with open(path, "rb") as file:
        while True:
            block = file.read(1024 * 1024)
            if not block:
                break
            md5_hash.update(block)

    return md5_hash.hexdigest()


def remove_exact_duplicates(metadata):
    """
     Drop every copy after the first of each identical image file.
    :param metadata: pandas DataFrame containing a "path" column
    :return: tuple (cleaned DataFrame, number of rows removed)
    """
    seen_hashes = set()
    rows_to_keep = []
    removed_count = 0

    for path in metadata["path"]:
        image_hash = file_md5(path)

        if image_hash in seen_hashes:
            rows_to_keep.append(False)
            removed_count += 1
        else:
            seen_hashes.add(image_hash)
            rows_to_keep.append(True)

    cleaned = metadata[np.array(rows_to_keep, dtype=bool)].copy()
    cleaned = cleaned.reset_index(drop=True)
    return cleaned, removed_count


def load_and_resize_image(path, image_size):
    """
    Center-crop to a square, then resize.
    :param path: path to the source image
    :param image_size: output size in pixels (both width and height)
    :return: uint8 numpy array of shape (image_size, image_size, 3)
    """
    image = Image.open(path).convert("RGB")
    width, height = image.size

    square_size = min(width, height)
    left = (width - square_size) // 2
    top = (height - square_size) // 2

    image = image.crop((left, top, left + square_size, top + square_size))
    image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)

    return np.array(image, dtype=np.uint8)


def split_by_lesion(metadata, test_fraction, seed):
    """
    Split pool/test per class, keeping each lesion on one side only.
    :param metadata: pandas DataFrame with "lesion_id" and "dx" columns
    :param test_fraction: fraction of lesions per class to place in test
    :param seed: random seed for reproducible splitting
    :return: tuple (pool DataFrame, test DataFrame)
    """
    random_generator = np.random.default_rng(seed)

    # One row per physical lesion is enough for choosing the split.
    lesion_table = metadata.groupby("lesion_id", as_index=False)["dx"].first()
    test_lesion_ids = set()

    for class_name in CLASSES:
        class_lesions = lesion_table[lesion_table["dx"] == class_name]
        lesion_ids = class_lesions["lesion_id"].to_numpy().copy()

        if len(lesion_ids) <= 1:
            continue

        random_generator.shuffle(lesion_ids)

        number_for_test = int(round(test_fraction * len(lesion_ids)))
        number_for_test = max(1, number_for_test)
        number_for_test = min(number_for_test, len(lesion_ids) - 1)

        for lesion_id in lesion_ids[:number_for_test]:
            test_lesion_ids.add(lesion_id)

    is_test = metadata["lesion_id"].isin(test_lesion_ids)

    pool_metadata = metadata[~is_test].copy().reset_index(drop=True)
    test_metadata = metadata[is_test].copy().reset_index(drop=True)

    return pool_metadata, test_metadata


def save_npz_dataset(metadata, output_path, image_size, name):
    """
    Load images and save them (plus labels and ids) into one npz file.
    Returns the in-memory image array and label array.
    :param metadata: pandas DataFrame with "path", "label", "image_id",
                     and "lesion_id" columns
    :param output_path: path where the npz file will be written
    :param image_size: output image size in pixels
    :param name: label used in progress messages (e.g. "pool", "test")
    :return: tuple (uint8 image array, int64 label array)
    """
    number_of_images = len(metadata)
    images = np.zeros((number_of_images, image_size, image_size, 3), dtype=np.uint8)

    for index in range(number_of_images):
        image_path = metadata.iloc[index]["path"]
        images[index] = load_and_resize_image(image_path, image_size)

        if (index + 1) % 1000 == 0:
            print(name + ":", index + 1, "/", number_of_images)

    labels = metadata["label"].to_numpy(dtype=np.int64)
    image_ids = np.array(metadata["image_id"].astype(str).tolist(), dtype="U64")
    lesion_ids = np.array(metadata["lesion_id"].astype(str).tolist(), dtype="U64")

    np.savez_compressed(
        output_path,
        x=images,
        y=labels,
        image_id=image_ids,
        lesion_id=lesion_ids,
        classes=np.array(CLASSES),
    )

    return images, labels


def save_class_distribution(metadata, output_folder):
    """
    Bar chart of class counts after cleaning.
    :param metadata: pandas DataFrame with a "dx" column
    :param output_folder: folder where the PNG will be written
    :return: None
    """
    counts = metadata["dx"].value_counts().reindex(CLASSES, fill_value=0)

    figure, axis = plt.subplots(figsize=(7, 4))
    bars = axis.bar(CLASSES, counts.values)
    axis.set_title("HAM10000 class distribution after cleaning")
    axis.set_xlabel("Diagnosis class")
    axis.set_ylabel("Number of images")
    axis.bar_label(bars, padding=2, fontsize=8)

    figure.tight_layout()
    figure.savefig(os.path.join(output_folder, "class_distribution.png"), dpi=180)
    plt.close(figure)


def save_sample_images(metadata, output_folder):
    """
    One example image per class.
    :param metadata: pandas DataFrame with "dx" and "path" columns
    :param output_folder: folder where the PNG will be written
    :return: None
    """
    figure, axes = plt.subplots(2, 4, figsize=(8, 4.5))
    axes = axes.ravel()

    for class_number in range(len(CLASSES)):
        class_name = CLASSES[class_number]
        class_rows = metadata[metadata["dx"] == class_name]

        if len(class_rows) == 0:
            axes[class_number].axis("off")
            continue

        sample_path = class_rows.iloc[0]["path"]
        sample_image = load_and_resize_image(sample_path, 128)

        axes[class_number].imshow(sample_image)
        axes[class_number].set_title(class_name.upper(), fontsize=9)
        axes[class_number].axis("off")

    axes[7].axis("off")
    figure.suptitle("Example HAM10000 dermoscopic images", fontsize=11)
    figure.tight_layout()
    figure.savefig(os.path.join(output_folder, "sample_lesions.png"), dpi=180)
    plt.close(figure)


def save_class_count_csv(metadata, pool_labels, test_labels, output_folder):
    """
    Per-class counts for the full data, the pool, and the test set.
    :param metadata: pandas DataFrame with a "label" column
    :param pool_labels: int64 array of pool labels
    :param test_labels: int64 array of test labels
    :param output_folder: folder where the CSV will be written
    :return: None
    """
    rows = []

    for class_number in range(len(CLASSES)):
        class_name = CLASSES[class_number]

        total_count = int((metadata["label"] == class_number).sum())
        pool_count = int((pool_labels == class_number).sum())
        test_count = int((test_labels == class_number).sum())

        rows.append({
            "class": class_name,
            "total_count": total_count,
            "pool_count": pool_count,
            "test_count": test_count,
        })

    pd.DataFrame(rows).to_csv(
        os.path.join(output_folder, "class_counts.csv"),
        index=False,
    )


def main():
    # Project root is the parent folder of src/.
    script_folder = os.path.dirname(os.path.abspath(__file__))
    project_folder = os.path.dirname(os.path.dirname(script_folder))

    default_metadata = os.path.join(
        project_folder, "data", "raw", "HAM10000_metadata.csv"
    )
    default_image_part_1 = os.path.join(
        project_folder, "data", "raw", "HAM10000_images_part_1"
    )
    default_image_part_2 = os.path.join(
        project_folder, "data", "raw", "HAM10000_images_part_2"
    )
    default_output = os.path.join(project_folder, "data", "processed")

    parser = argparse.ArgumentParser(description="Prepare HAM10000 for federated learning.")
    parser.add_argument("--metadata", default=default_metadata)
    parser.add_argument(
        "--images",
        nargs="+",
        default=[default_image_part_1, default_image_part_2],
    )
    parser.add_argument("--out", default=default_output)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-dedup", action="store_true")
    args = parser.parse_args()

    if not os.path.isfile(args.metadata):
        raise SystemExit(
            "HAM10000_metadata.csv was not found. Put it here:\n" + args.metadata
        )

    for image_folder in args.images:
        if not os.path.isdir(image_folder):
            raise SystemExit(
                "HAM10000 image folder was not found. Put it here:\n"
                + image_folder
            )

    os.makedirs(args.out, exist_ok=True)

    print("Using metadata:", args.metadata)
    print("Using image folders:")
    for image_folder in args.images:
        print(" -", image_folder)
    print("Output folder:", args.out)

    # 1. Read and check the metadata
    metadata = pd.read_csv(args.metadata)
    original_rows = len(metadata)

    required_columns = ["lesion_id", "image_id", "dx"]
    for column in required_columns:
        if column not in metadata.columns:
            raise SystemExit("Missing required metadata column: " + column)

    # 2. Match metadata rows with image files
    image_paths = find_image_files(args.images)
    metadata["path"] = metadata["image_id"].map(image_paths)

    missing_files = int(metadata["path"].isna().sum())
    metadata = metadata.dropna(subset=["path"]).reset_index(drop=True)

    # 3. Remove exact duplicate image files
    duplicates_removed = 0
    if not args.no_dedup:
        metadata, duplicates_removed = remove_exact_duplicates(metadata)

    # Keep only the 7 classes used by HAM10000.
    metadata = metadata[metadata["dx"].isin(CLASSES)].copy().reset_index(drop=True)
    metadata["label"] = metadata["dx"].map(CLASS_TO_NUMBER).astype(np.int64)

    labels_per_lesion = metadata.groupby("lesion_id")["dx"].nunique()
    conflicting_lesions = labels_per_lesion[labels_per_lesion > 1]
    lesion_label_conflicts = len(conflicting_lesions)

    if lesion_label_conflicts > 0:
        raise ValueError(
            "Found "
            + str(lesion_label_conflicts)
            + " lesion(s) with more than one diagnosis label."
        )

    # 4. Split by lesion_id into training pool and held-out test set
    pool_metadata, test_metadata = split_by_lesion(
        metadata,
        args.test_frac,
        args.seed,
    )

    pool_lesions = set(pool_metadata["lesion_id"].astype(str))
    test_lesions = set(test_metadata["lesion_id"].astype(str))

    if pool_lesions.intersection(test_lesions):
        raise AssertionError("Lesion leakage detected between pool and test set")

    print("Raw metadata rows       :", original_rows)
    print("Missing image files     :", missing_files)
    print("Exact duplicates removed:", duplicates_removed)
    print("Lesion-label conflicts  :", lesion_label_conflicts)
    print("Usable images           :", len(metadata))
    print("Unique lesions          :", metadata["lesion_id"].nunique())
    print("Pool images             :", len(pool_metadata))
    print("Test images             :", len(test_metadata))
    print("Pool-test lesion leakage: 0")

    # 5. Resize and save images
    pool_images, pool_labels = save_npz_dataset(
        pool_metadata,
        os.path.join(args.out, "pool.npz"),
        args.size,
        "pool",
    )

    _, test_labels = save_npz_dataset(
        test_metadata,
        os.path.join(args.out, "test.npz"),
        args.size,
        "test",
    )

    # 6. Calculate normalization values from the training pool only.
    mean = pool_images.mean(axis=(0, 1, 2), dtype=np.float64) / 255.0
    std = pool_images.std(axis=(0, 1, 2), dtype=np.float64) / 255.0

    normalization_info = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "image_size": args.size,
        "classes": CLASSES,
    }

    with open(os.path.join(args.out, "norm_stats.json"), "w", encoding="utf-8") as file:
        json.dump(normalization_info, file, indent=2)

    # 7. Save simple audit files and report figures
    cleaning_report = pd.DataFrame([
        {
            "raw_rows": original_rows,
            "missing_files": missing_files,
            "byte_duplicates_removed": duplicates_removed,
            "lesion_label_conflicts": lesion_label_conflicts,
            "usable_images": len(metadata),
            "unique_lesions": int(metadata["lesion_id"].nunique()),
            "pool_images": len(pool_metadata),
            "test_images": len(test_metadata),
            "image_size": args.size,
            "seed": args.seed,
        }
    ])
    cleaning_report.to_csv(os.path.join(args.out, "cleaning_report.csv"), index=False)

    save_class_count_csv(metadata, pool_labels, test_labels, args.out)
    save_class_distribution(metadata, args.out)
    save_sample_images(metadata, args.out)

    print("\nFinished. Files were written to:", args.out)

if __name__ == "__main__":
    main()
