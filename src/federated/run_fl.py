"""
Runs ONE conventional federated learning simulation with Flower.

  1. The server builds DermCNN once and broadcasts the same initial weights to all 20 clients.
  2. Every round, ALL clients train locally (fraction_fit = 1.0) and send 
     their weights back; the server aggregates them with FedAvg.
  3. After every aggregation, ALL clients evaluate the new global model
     on their local validation subset (fraction_evaluate = 1.0), the
     server also evaluates it on the clean held-out test set.
  4. All metrics are written to CSV and the four plots are made.

@author Amruta Kirtikumar Khot, ak3792@rit.edu
@author Cazal Tirkey, ct9857@rit.edu
"""

import argparse
import json
import os
import sys
import time

import numpy as np

SRC_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_FOLDERS = [os.path.join(SRC_FOLDER, name) for name in ("data_prep", "attack", "federated", "analysis")]
for folder in CODE_FOLDERS:
    if folder not in sys.path:
        sys.path.insert(0, folder)
os.environ["PYTHONPATH"] = os.pathsep.join(CODE_FOLDERS + [os.environ.get("PYTHONPATH", "")])

import torch
import torch.nn as nn
import flwr as fl
from flwr.common import ndarrays_to_parameters

from augment import normalize
from fl_client import get_weights, make_client_fn, set_weights
from fl_strategy import LoggingFedAvg
from metrics_logger import MetricsLogger, classification_metrics
from model import DermCNN
from plot_results import plot_run


def make_server_evaluate_fn(test_path, mean, std, logger):
    """
    Server-side evaluation of the global model on the clean test set.
    """
    test = np.load(test_path, allow_pickle=False)
    x_test = test["x"]
    y_test = test["y"].astype(np.int64)
    loss_function = nn.CrossEntropyLoss(reduction="sum")

    def evaluate(server_round, parameters, config):
        model = DermCNN()
        set_weights(model, parameters)
        model.eval()
        total_loss = 0.0
        predictions = []
        with torch.no_grad():
            for start in range(0, len(y_test), 256):
                inputs = torch.from_numpy(normalize(x_test[start:start + 256], mean, std))
                targets = torch.from_numpy(y_test[start:start + 256])
                outputs = model(inputs)
                total_loss += float(loss_function(outputs, targets).item())
                predictions.append(outputs.argmax(dim=1).numpy())
        metrics = classification_metrics(y_test, np.concatenate(predictions))
        metrics["loss"] = total_loss / len(y_test)
        logger.add_global(server_round, metrics)
        print(
            "[round " + str(server_round) + "] global test: loss "
            + format(metrics["loss"], ".4f") + ", acc " + format(metrics["accuracy"], ".4f")
            + ", balanced acc " + format(metrics["balanced_accuracy"], ".4f")
        )
        return metrics["loss"], metrics

    return evaluate


def main():
    project_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    parser = argparse.ArgumentParser(description="Run one conventional FedAvg simulation.")
    parser.add_argument("--tag", required=True, help="prefix for output files, e.g. clean or p70_flip50")
    parser.add_argument("--out", required=True, help="results/no_attack or results/attack (files go into plots/, csv/, json/)")
    parser.add_argument("--clients-dir", default=os.path.join(project_folder, "data", "clients"))
    parser.add_argument("--poisoned-dir", default=None, help="attack scenario folder; omit for the clean run")
    parser.add_argument("--test", default=os.path.join(project_folder, "data", "processed", "test.npz"))
    parser.add_argument("--norm", default=os.path.join(project_folder, "data", "processed", "norm_stats.json"))
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--no-augment", action="store_true")
    # Class-weighted loss is ON by default: without it the global model
    # collapsed to always predicting "nv"
    parser.add_argument("--class-weights", dest="class_weights", action="store_true", default=True,
                        help="inverse-frequency weighted loss on each client (default: on)")
    parser.add_argument("--no-class-weights", dest="class_weights", action="store_false",
                        help="plain cross-entropy (reproduces the majority-class collapse)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpus-per-client", type=float, default=1.0)
    parser.add_argument("--gpus-per-client", type=float, default=0.0,
                        help="e.g. 0.1 lets 10 clients share one GPU; 0 = CPU only")
    args = parser.parse_args()

    with open(args.norm, encoding="utf-8") as file:
        norm = json.load(file)
    mean, std = norm["mean"], norm["std"]

    # Malicious ids are read from the scenario file so plots/CSVs can mark them.
    malicious_ids = []
    if args.poisoned_dir:
        with open(os.path.join(args.poisoned_dir, "scenario.json"), encoding="utf-8") as file:
            scenario = json.load(file)
        malicious_ids = scenario["malicious_clients"]
    print("Run " + args.tag + ": malicious clients = " + str(malicious_ids))

    settings = {
        "seed": args.seed,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "local_epochs": args.local_epochs,
        "augment": not args.no_augment,
        "use_gpu": args.gpus_per_client > 0,
        "class_weights": args.class_weights,
    }

    logger = MetricsLogger(malicious_ids)

    # 1. Centralised model initialisation: one DermCNN, fixed seed.
    initial_model = DermCNN()
    initial_parameters = ndarrays_to_parameters(get_weights(initial_model))

    def round_config(server_round):
        return {"server_round": server_round}

    strategy = LoggingFedAvg(
        logger=logger,
        fraction_fit=1.0,             # all 20 clients train every round
        fraction_evaluate=1.0,        # all 20 clients evaluate every round
        min_fit_clients=args.num_clients,
        min_evaluate_clients=args.num_clients,
        min_available_clients=args.num_clients,
        initial_parameters=initial_parameters,
        on_fit_config_fn=round_config,
        on_evaluate_config_fn=round_config,
        evaluate_fn=make_server_evaluate_fn(args.test, mean, std, logger),
        accept_failures=False,
    )

    client_fn = make_client_fn(args.clients_dir, args.poisoned_dir, mean, std, settings)

    start = time.time()
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources={"num_cpus": args.cpus_per_client, "num_gpus": args.gpus_per_client},
    )
    elapsed = time.time() - start

    config = vars(args).copy()
    config.update({
        "framework": "flower " + fl.__version__,
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "strategy": "FedAvg (weighted by local training examples), no client exclusion",
        "total_seconds": round(elapsed, 1),
        "norm_mean": mean,
        "norm_std": std,
    })
    logger.save(args.out, args.tag, config)
    plot_run(args.out, args.tag)
    print("Finished " + args.tag + " in " + format(elapsed / 60, ".1f") + " min. Files in " + args.out)


if __name__ == "__main__":
    main()
