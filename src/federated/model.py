"""
Initial CNN for the HAM10000 7-class skin-lesion task.

Kept small enough that 20 clients can run locally on one machine, but
still has three convolutional stages so it can learn real features.

@author Amruta Kirtikumar Khot, ak3792@rit.edu
@author Cazal Tirkey, ct9857@rit.edu
"""
import numpy as np
import torch
import torch.nn as nn


NUMBER_OF_CLASSES = 7
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def create_convolution_block(input_channels, output_channels):
    """
    Create one simple CNN block: Conv -> Norm -> ReLU -> Conv -> Norm -> ReLU -> Pool.
    :param input_channels: channels coming into the block
    :param output_channels: channels produced by the block
    :return: nn.Sequential block
    """
    block = nn.Sequential(
        nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        ),
        nn.GroupNorm(min(8, output_channels), output_channels),
        nn.ReLU(inplace=True),

        nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        ),
        nn.GroupNorm(min(8, output_channels), output_channels),
        nn.ReLU(inplace=True),

        nn.MaxPool2d(kernel_size=2),
    )

    return block


class DermCNN(nn.Module):
    """
    Small convolutional network with 7 output scores.

    Three conv blocks feed a global average pool and a small classifier
    head. Dropout is applied before each fully-connected layer.
    """

    def __init__(self, number_of_classes=NUMBER_OF_CLASSES, dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(
            create_convolution_block(3, 32),
            create_convolution_block(32, 64),
            create_convolution_block(64, 128),
        )

        self.average_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, number_of_classes),
        )

        self.initialize_weights()

    def initialize_weights(self, seed=42):
        """
        Initialize weights once so every client starts from the same model.

        :param seed: torch manual seed, fixed so runs are reproducible
        :return: None
        """
        torch.manual_seed(seed)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(
                    layer.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

            elif isinstance(layer, nn.GroupNorm):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, images):
        """
        Forward pass.

        :param images: float tensor of shape (N, 3, H, W)
        :return: float tensor of shape (N, number_of_classes)
        """
        features = self.features(images)
        pooled_features = self.average_pool(features)
        output_scores = self.classifier(pooled_features)
        return output_scores


def calculate_class_weights(labels, number_of_classes=NUMBER_OF_CLASSES):
    """
    Return optional inverse-frequency weights for cross-entropy loss.
    :param labels: array-like of numeric labels
    :param number_of_classes: total number of classes
    :return: float32 torch tensor of shape (number_of_classes,)
    """
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=number_of_classes).astype(np.float64)

    weights = np.zeros(number_of_classes, dtype=np.float64)
    total_samples = counts.sum()

    for class_number in range(number_of_classes):
        if counts[class_number] > 0:
            weights[class_number] = (
                total_samples
                / (number_of_classes * counts[class_number])
            )

    return torch.tensor(weights, dtype=torch.float32)


if __name__ == "__main__":
    model = DermCNN()

    trainable_parameters = 0
    for parameter in model.parameters():
        trainable_parameters += parameter.numel()

    sample_batch = torch.zeros(2, 3, 64, 64)
    sample_output = model(sample_batch)

    print(model)
    print("Trainable parameters:", format(trainable_parameters, ","))
    print(
        "FP32 parameter size :",
        format(trainable_parameters * 4 / 1_000_000, ".2f"),
        "MB",
    )
    print("Output shape        :", sample_output.shape)
