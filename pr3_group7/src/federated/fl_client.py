"""
Flower client for the HAM10000 federated learning simulation.

Each simulated client:
  1. loads ONLY its own client_XX.npz file (clean, or the poisoned copy if
    it is a malicious client in the current attack scenario)
  2. receives the global DermCNN weights from the server
  3. trains locally for a few epochs with data augmentation (fit)
  4. evaluates the global model on its own local validation subset and
    reports loss and accuracy (evaluate).

Malicious clients run exactly the same code; the only difference is that
their files on disk contain flipped labels. This is a pure data-poisoning
attack.

@author Amruta Kirtikumar Khot, ak3792@rit.edu
@author Cazal Tirkey, ct9857@rit.edu
"""

import json
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from flwr.client import NumPyClient
from flwr.common import Context

from augment import augment_batch, normalize
from model import DermCNN, calculate_class_weights


def get_weights(model):
    """
    Model parameters as a list of numpy arrays (the format Flower sends).
    :param model: torch model
    :return: list of numpy arrays in state_dict order
    """
    return [value.detach().cpu().numpy() for value in model.state_dict().values()]


def set_weights(model, weights):
    """
    Load a list of numpy arrays into the model, in state_dict order.
    :param model: torch model
    :param weights: list of numpy arrays
    :return: None
    """
    keys = list(model.state_dict().keys())
    state = OrderedDict((key, torch.tensor(np.asarray(w))) for key, w in zip(keys, weights))
    model.load_state_dict(state, strict=True)


def load_client_data(client_id, clean_dir, poisoned_dir=None):
    """
    Load one client's local dataset from disk.
    If the attack scenario folder has a file for this client, the client is
    malicious and its poisoned copy is used otherwise the clean file is used.
    :param client_id: int client id
    :param clean_dir: folder with clean client_XX.npz files
    :param poisoned_dir: scenario folder with poisoned files, or None
    :return: (dict of arrays, is_malicious)
    """
    filename = "client_" + format(client_id, "02d") + ".npz"
    path = os.path.join(clean_dir, filename)
    is_malicious = False

    if poisoned_dir:
        # The scenario file is the single source of truth for who is malicious.
        with open(os.path.join(poisoned_dir, "scenario.json"), encoding="utf-8") as file:
            malicious_ids = json.load(file)["malicious_clients"]
        if client_id in malicious_ids:
            path = os.path.join(poisoned_dir, filename)
            if not os.path.isfile(path):
                raise FileNotFoundError("Poisoned file missing for malicious client: " + path)
            is_malicious = True

    data = np.load(path, allow_pickle=False)
    arrays = {
        "x_train": data["x_train"],
        "y_train": data["y_train"].astype(np.int64),
        "x_val": data["x_val"],
        "y_val": data["y_val"].astype(np.int64),
    }
    return arrays, is_malicious


class HamClient(NumPyClient):
    """
    One federated client holding a local HAM10000 subset.
    """

    def __init__(self, client_id, data, is_malicious, mean, std, settings):
        self.client_id = client_id
        self.data = data
        self.is_malicious = is_malicious
        self.mean = mean
        self.std = std
        self.settings = settings
        # Use the GPU only if one was requested and is available.
        use_gpu = settings.get("use_gpu", False) and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_gpu else "cpu")
        # Keep each simulated client to one CPU thread so 20 clients
        # running in parallel do not fight over cores.
        torch.set_num_threads(1)

    def get_parameters(self, config):
        return get_weights(DermCNN())

    def fit(self, parameters, config):
        """
        Local training on this client's (possibly poisoned) training data.
        :param parameters: global model weights from the server
        :param config: dict from the server, contains "server_round"
        :return: (updated weights, number of training examples, metrics)
        """
        server_round = int(config.get("server_round", 0))

        model = DermCNN()
        set_weights(model, parameters)
        model.to(self.device)

        # Seed AFTER building the model
        # so every client/round has its own reproducible randomness.
        seed = self.settings["seed"] + 1000 * server_round + self.client_id
        torch.manual_seed(seed)
        random_generator = np.random.default_rng(seed)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.settings["lr"],
            weight_decay=self.settings["weight_decay"],
        )
        if self.settings.get("class_weights", False):
            weights = torch.clamp(calculate_class_weights(self.data["y_train"]), max=10.0)
            loss_function = nn.CrossEntropyLoss(weight=weights.to(self.device))
        else:
            loss_function = nn.CrossEntropyLoss()

        x_train = self.data["x_train"]
        y_train = self.data["y_train"]
        batch_size = self.settings["batch_size"]

        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for _epoch in range(self.settings["local_epochs"]):
            order = random_generator.permutation(len(y_train))

            for start in range(0, len(order), batch_size):
                batch_indices = order[start:start + batch_size]
                images = x_train[batch_indices]
                if self.settings["augment"]:
                    images = augment_batch(images, random_generator)

                inputs = torch.from_numpy(normalize(images, self.mean, self.std)).to(self.device)
                targets = torch.from_numpy(y_train[batch_indices]).to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item()) * len(targets)
                total_correct += int((outputs.argmax(dim=1) == targets).sum().item())
                total_seen += len(targets)

        metrics = {
            "client_id": self.client_id,
            "malicious": int(self.is_malicious),
            "train_loss": total_loss / max(1, total_seen),
            "train_accuracy": total_correct / max(1, total_seen),
        }
        return get_weights(model), len(y_train), metrics

    def evaluate(self, parameters, config):
        """
        Evaluate the GLOBAL model on this client's local validation subset.
        For a malicious client this subset carries the flipped labels.
        :return: (loss, number of validation examples, metrics)
        """
        model = DermCNN()
        set_weights(model, parameters)
        model.to(self.device)
        model.eval()

        x_val = self.data["x_val"]
        y_val = self.data["y_val"]

        if len(y_val) == 0:
            return 0.0, 0, {"client_id": self.client_id, "malicious": int(self.is_malicious), "accuracy": 0.0}

        loss_function = nn.CrossEntropyLoss(reduction="sum")
        total_loss = 0.0
        total_correct = 0

        with torch.no_grad():
            for start in range(0, len(y_val), 256):
                images = x_val[start:start + 256]
                inputs = torch.from_numpy(normalize(images, self.mean, self.std)).to(self.device)
                targets = torch.from_numpy(y_val[start:start + 256]).to(self.device)
                outputs = model(inputs)
                total_loss += float(loss_function(outputs, targets).item())
                total_correct += int((outputs.argmax(dim=1) == targets).sum().item())

        loss = total_loss / len(y_val)
        accuracy = total_correct / len(y_val)
        metrics = {
            "client_id": self.client_id,
            "malicious": int(self.is_malicious),
            "accuracy": accuracy,
        }
        return float(loss), len(y_val), metrics


def make_client_fn(clean_dir, poisoned_dir, mean, std, settings):
    """
    Build the client_fn that Flower calls to create a client.
    Flower's simulation engine gives every virtual client a
    "partition-id" from 0 to num_clients-1; we use it as the client id,
    so client 3 always loads client_03.npz.
    """

    def client_fn(context: Context):
        client_id = int(context.node_config["partition-id"])
        data, is_malicious = load_client_data(client_id, clean_dir, poisoned_dir)
        return HamClient(client_id, data, is_malicious, mean, std, settings).to_client()

    return client_fn
