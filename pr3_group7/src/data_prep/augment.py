"""
Simple image augmentation and normalization helpers.

@author Amruta Kirtikumar Khot, ak3792@rit.edu
@author Cazal Tirkey, ct9857@rit.edu
"""

import numpy as np
from PIL import Image, ImageEnhance


def random_number(random_generator, low, high):
    """
    Return one random floating-point number between low and high.
    :param random_generator: numpy Generator
    :param low: lower bound
    :param high: upper bound
    :return: float in [low, high]
    """
    return float(random_generator.uniform(low, high))


def augment_image(image_array, random_generator):
    """
    Create one slightly changed copy of a training image.
    :param image_array: uint8 NHWC image
    :param random_generator: numpy Generator
    :return: uint8 NHWC image of the same size
    """
    image = Image.fromarray(image_array)

    # Make sure the image has 3 RGB channels.
    if image.mode != "RGB":
        image = image.convert("RGB")

    width, height = image.size

    # Random horizontal flip.
    if random_generator.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    # Random vertical flip.
    if random_generator.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    # Small rotation between -20 and +20 degrees.
    angle = random_number(random_generator, -20.0, 20.0)
    image = image.rotate(
        angle,
        resample=Image.Resampling.BILINEAR,
        fillcolor=(0, 0, 0),
    )

    # Small random zoom.
    zoom = random_number(random_generator, 0.9, 1.1)
    crop_size = int(round(min(width, height) / zoom))
    crop_size = max(8, min(crop_size, min(width, height)))

    # Small random shift around the image center.
    max_shift = int(0.08 * width)
    center_x = width // 2 + int(
        random_generator.integers(-max_shift, max_shift + 1)
    )
    center_y = height // 2 + int(
        random_generator.integers(-max_shift, max_shift + 1)
    )

    left = center_x - crop_size // 2
    top = center_y - crop_size // 2

    left = max(0, min(left, width - crop_size))
    top = max(0, min(top, height - crop_size))

    image = image.crop(
        (left, top, left + crop_size, top + crop_size)
    )
    image = image.resize((width, height), Image.Resampling.BILINEAR)

    # Keep colour changes mild because colour is useful in lesion images.
    brightness = random_number(random_generator, 0.85, 1.15)
    colour = random_number(random_generator, 0.90, 1.10)
    contrast = random_number(random_generator, 0.85, 1.15)

    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Color(image).enhance(colour)
    image = ImageEnhance.Contrast(image).enhance(contrast)

    return np.array(image, dtype=np.uint8)


def augment_batch(images, random_generator=None):
    """
    Augment every image in a batch.
    :param images: uint8 array of shape (N, H, W, 3)
    :param random_generator: numpy Generator, optional
    :return: uint8 array of shape (N, H, W, 3)
    """
    if random_generator is None:
        random_generator = np.random.default_rng()

    augmented_images = []

    for image in images:
        augmented_images.append(
            augment_image(image, random_generator)
        )

    return np.stack(augmented_images)


def balance_by_augmentation(
    images,
    labels,
    random_generator,
    number_of_classes=7,
    target="median",
    cap=4.0,
):
    """
    Optionally add augmented copies of minority-class training images.
    :param images: uint8 array of shape (N, H, W, 3)
    :param labels: int array of shape (N,)
    :param random_generator: numpy Generator
    :param number_of_classes: number of classes
    :param target: "median", "max", or an int target count per class
    :param cap: never add more than cap * current_count per class
    :return: (balanced images, balanced labels, added per class array)
    """
    class_counts = []

    for class_number in range(number_of_classes):
        class_counts.append(int((labels == class_number).sum()))

    class_counts = np.array(class_counts)
    non_zero_counts = class_counts[class_counts > 0]

    if len(non_zero_counts) == 0:
        return images, labels, np.zeros(number_of_classes, dtype=int)

    if target == "median":
        desired_count = int(np.median(non_zero_counts))
    elif target == "max":
        desired_count = int(non_zero_counts.max())
    else:
        desired_count = int(target)

    all_images = [images]
    all_labels = [labels]
    added_per_class = np.zeros(number_of_classes, dtype=int)

    for class_number in range(number_of_classes):
        current_count = int(class_counts[class_number])

        if current_count == 0:
            continue

        maximum_allowed = int(cap * current_count)
        target_count = min(desired_count, maximum_allowed)
        number_to_add = target_count - current_count

        if number_to_add <= 0:
            continue

        class_indices = np.where(labels == class_number)[0]
        selected_indices = random_generator.choice(
            class_indices,
            size=number_to_add,
            replace=True,
        )

        new_images = augment_batch(
            images[selected_indices],
            random_generator,
        )
        new_labels = np.full(
            number_to_add,
            class_number,
            dtype=labels.dtype,
        )

        all_images.append(new_images)
        all_labels.append(new_labels)
        added_per_class[class_number] = number_to_add

    combined_images = np.concatenate(all_images, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)

    order = random_generator.permutation(len(combined_labels))
    combined_images = combined_images[order]
    combined_labels = combined_labels[order]

    return combined_images, combined_labels, added_per_class


def normalize(images, mean, std):
    """
    Convert NHWC uint8 images to normalized NCHW float32 arrays.
    :param images: uint8 NHWC array, shape (N, H, W, 3)
    :param mean: list or array of 3 channel means
    :param std: list or array of 3 channel stds
    :return: float32 NCHW array, shape (N, 3, H, W)
    """
    images = images.astype(np.float32) / 255.0

    # Shape mean/std explicitly as RGB channel values for NHWC images.
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 1, -1)
    std = np.array(std, dtype=np.float32).reshape(1, 1, 1, -1)

    images = (images - mean) / std
    images = np.transpose(images, (0, 3, 1, 2))

    return images.copy()
