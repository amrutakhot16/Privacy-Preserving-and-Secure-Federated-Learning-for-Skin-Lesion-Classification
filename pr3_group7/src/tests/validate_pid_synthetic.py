"""
Validation of the PID implementation on synthetic client updates (Part 3).

Why this exists: on HAM10000 with non-IID clients, label-flipping clients
are not further from the centroid than honest clients, so the PID defense
removes almost nobody (see the report). To show that this is a property of
the data and NOT a bug in the code, this script runs the real PIDFedAvg
selection code on synthetic model updates where the correct answer is known:

  * every honest client sends   w + step * good_direction + small personal offset
  * every malicious client sends w + step * bad_direction  + small personal offset

so malicious updates are clearly different from honest ones. The script
checks, for 0%, 10%, 30%, 50% and 70% malicious clients, which clients the
PID strategy excludes. Expected behaviour of a distance-to-centroid rule:

  0%   nobody is excluded
  10%  all malicious clients excluded        (they are the outliers)
  30%  all malicious clients excluded
  50%  no clear outliers: the two groups are equally far from the centroid
  70%  the HONEST clients are excluded       (the attackers ARE the centroid)

Exactly the same PIDFedAvg.select_clients_for_aggregation(), PID formula,
threshold and permanent exclusion as in the real runs are used. Only the
Flower training/communication is replaced by a loop; nothing is trained.
Runs in a few seconds, needs no dataset and no Ray.

Usage (from the project root):
  python src/tests/validate_pid_synthetic.py
  python src/tests/validate_pid_synthetic.py --kp 1 --ki 0.05 --kd 0.5 --threshold-k 1.75

Output in results/validation/:
  csv/pid_synthetic_validation.csv            one row per scenario: TP, FN, FP, TN
  csv/synthetic_<scenario>_pid_scores.csv     PID scores per round and client
  plots/synthetic_<scenario>_6_pid_scores.png, _7_client_removal.png

@author Amruta Kirtikumar Khot, ak3792@rit.edu
@author Cazal Tirkey, ct9857@rit.edu
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

SRC_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _name in ("federated", "analysis"):
    sys.path.insert(0, os.path.join(SRC_FOLDER, _name))

from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays  # noqa: E402

from metrics_logger import MetricsLogger, run_subdir, CSV_DIR  # noqa: E402
from pid_strategy import PIDFedAvg  # noqa: E402
from plot_results import plot_client_removal, plot_pid_scores  # noqa: E402


class SyntheticProxy:
    """Stands in for Flower's ClientProxy; only .cid is used by the strategy."""

    def __init__(self, cid):
        self.cid = cid


class SyntheticFitRes:
    """Stands in for Flower's FitRes; only the fields the strategy reads."""

    def __init__(self, vector, num_examples, client_id):
        # Split into two "layers" like a real model's list of arrays.
        half = len(vector) // 2
        self.parameters = ndarrays_to_parameters([vector[:half].astype(np.float32),
                                                  vector[half:].astype(np.float32)])
        self.num_examples = num_examples
        self.metrics = {"client_id": client_id}


def run_scenario(malicious_ids, args, seed):
    """
    Simulate args.rounds rounds with the real PID selection code.
    :return: MetricsLogger with the PID rows
    """
    rng = np.random.default_rng(seed)
    dim = args.dim
    good = rng.normal(size=dim)
    good /= np.linalg.norm(good)
    bad = rng.normal(size=dim)
    bad /= np.linalg.norm(bad)
    # Each client has a fixed personal offset (like its own local data).
    offsets = {c: rng.normal(size=dim) * args.heterogeneity / np.sqrt(dim) for c in range(args.clients)}
    sizes = rng.integers(150, 500, args.clients)

    logger = MetricsLogger(malicious_ids)
    strategy = PIDFedAvg(logger=logger, kp=args.kp, ki=args.ki, kd=args.kd,
                         threshold_mode=args.threshold_mode, threshold=args.threshold,
                         threshold_k=args.threshold_k, distance_norm="mean", min_keep=2)
    proxies = [SyntheticProxy("node-" + str(1000 + c)) for c in range(args.clients)]

    weights = np.zeros(dim)
    for server_round in range(1, args.rounds + 1):
        results = []
        for client_id, proxy in enumerate(proxies):
            if proxy.cid in strategy.excluded_proxy_cids:
                continue  # excluded clients no longer take part (permanent exclusion)
            direction = bad if client_id in malicious_ids else good
            update = (weights + args.step * direction + offsets[client_id]
                      + rng.normal(size=dim) * args.noise / np.sqrt(dim))
            results.append((proxy, SyntheticFitRes(update, int(sizes[client_id]), client_id)))

        kept, _excluded = strategy.select_clients_for_aggregation(server_round, results)

        # FedAvg over the kept clients (weighted by number of examples).
        total = sum(res.num_examples for _p, res in kept)
        weights = sum(res.num_examples * np.concatenate(
            [np.asarray(a, dtype=np.float64) for a in parameters_to_ndarrays(res.parameters)])
            for _p, res in kept) / total
    return logger


def main():
    project_folder = os.path.dirname(SRC_FOLDER)
    parser = argparse.ArgumentParser(description="Validate PIDFedAvg on synthetic updates.")
    parser.add_argument("--kp", type=float, default=1.0)
    parser.add_argument("--ki", type=float, default=0.05)
    parser.add_argument("--kd", type=float, default=0.5)
    parser.add_argument("--threshold-mode", choices=["median", "fixed", "std"], default="median")
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--threshold-k", type=float, default=1.75)
    parser.add_argument("--clients", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--dim", type=int, default=5000, help="length of the synthetic weight vector")
    parser.add_argument("--step", type=float, default=0.5, help="size of each client's update")
    parser.add_argument("--heterogeneity", type=float, default=0.1, help="size of each client's personal offset")
    parser.add_argument("--noise", type=float, default=0.1, help="per-round random noise")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=os.path.join(project_folder, "results", "validation"))
    args = parser.parse_args()

    # Same fixed client ordering idea as poison_clients.py: smaller sets are subsets of larger ones.
    order = [int(c) for c in np.random.default_rng(args.seed).permutation(args.clients)]
    rows = []
    for percent in (0, 10, 30, 50, 70):
        malicious = sorted(order[: round(args.clients * percent / 100)])
        tag = "synthetic_p" + str(percent)
        logger = run_scenario(set(malicious), args, args.seed)
        # Only the PID scores exist here (nothing is trained or evaluated).
        pd.DataFrame(logger.pid_rows).to_csv(
            os.path.join(run_subdir(args.out, CSV_DIR, create=True), tag + "_pid_scores.csv"), index=False)
        plot_pid_scores(args.out, tag)
        plot_client_removal(args.out, tag)
        summary = logger.detection_summary(args.clients)
        counts = summary["outcome"].value_counts()
        rows.append({
            "malicious_percent": percent,
            "malicious_clients": len(malicious),
            "true_positive": int(counts.get("true_positive", 0)),
            "false_negative": int(counts.get("false_negative", 0)),
            "false_positive": int(counts.get("false_positive", 0)),
            "true_negative": int(counts.get("true_negative", 0)),
        })

    table = pd.DataFrame(rows)
    path = os.path.join(run_subdir(args.out, CSV_DIR, create=True), "pid_synthetic_validation.csv")
    table.to_csv(path, index=False)
    print("\nPID on synthetic updates (Kp=%g, Ki=%g, Kd=%g, %s threshold k=%g):"
          % (args.kp, args.ki, args.kd, args.threshold_mode, args.threshold_k))
    print(table.to_string(index=False))
    print("\nExpected: 0% -> nobody excluded; 10%/30% -> all malicious excluded; "
          "50% -> no clear outliers; 70% -> honest clients excluded.")
    print("Saved " + path + " and plots in " + os.path.join(args.out, "plots"))


if __name__ == "__main__":
    main()
