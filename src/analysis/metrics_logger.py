"""
Collects FL metrics during a simulation and saves them as CSV files.

This module has no Flower/PyTorch dependency so it can be unit-tested on
its own. The strategy (fl_strategy.py) calls the add_* methods after every
aggregation round; run_fl.py calls save() when the simulation finishes.

Files written for a run with tag T (e.g. "clean" or "p70_flip50") into
results/no_attack/ or results/attack/ (both use the same layout, one
subfolder per file type):
  csv/
    T_loss_per_client.csv        rounds x clients table of evaluation loss      (required)
    T_accuracy_per_client.csv    rounds x clients table of evaluation accuracy  (required)
    T_avg_loss.csv               average evaluation loss among clients          (required)
    T_avg_accuracy.csv           average evaluation accuracy among clients      (required)
    T_per_client_eval_long.csv   one row per (round, client): loss, accuracy
    T_train_per_client_long.csv  local training loss/accuracy reported in fit
    T_global_test.csv            global model on the clean held-out test set
    T_aggregation_log.csv        clients aggregated / excluded each round
  json/
    T_config.json                every setting needed to reproduce the run
  plots/                         written by plot_results.py

@author Amruta Kirtikumar Khot, ak3792@rit.edu
@author Cazal Tirkey, ct9857@rit.edu
"""

import json
import os

import numpy as np
import pandas as pd


CSV_DIR = "csv"
JSON_DIR = "json"
PLOTS_DIR = "plots"


def run_subdir(run_dir, kind, create=False):
    """
    Path of one of the results subfolders: "csv", "json" or "plots".
    :param run_dir: results/no_attack or results/attack
    :param kind: CSV_DIR, JSON_DIR or PLOTS_DIR
    :param create: make the folder if it does not exist
    :return: folder path
    """
    path = os.path.join(run_dir, kind)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def classification_metrics(y_true, y_pred, number_of_classes=7):
    """
    Accuracy, balanced accuracy and macro-F1 (numpy only).
    Balanced accuracy and macro-F1 matter for HAM10000 because "nv" is
    about two thirds of the data, so plain accuracy can look good for a
    model that mostly predicts "nv".
    :param y_true: int array of true labels
    :param y_pred: int array of predicted labels
    :return: dict
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    recalls = []
    f1_scores = []
    for c in range(number_of_classes):
        true_c = y_true == c
        pred_c = y_pred == c
        if true_c.sum() == 0:
            continue
        tp = float((true_c & pred_c).sum())
        recall = tp / true_c.sum()
        precision = tp / pred_c.sum() if pred_c.sum() > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        recalls.append(recall)
        f1_scores.append(f1)
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_scores)),
    }


def client_column_name(client_id, malicious_ids):
    """
    Column name used in the wide per-client tables.
    :param client_id: int client id
    :param malicious_ids: set of malicious client ids
    :return: e.g. "client_03_malicious" or "client_04_benign"
    """
    role = "malicious" if client_id in malicious_ids else "benign"
    return "client_" + format(client_id, "02d") + "_" + role


class MetricsLogger:
    """
    Stores per-round metrics in memory and writes them to CSV.
    """

    def __init__(self, malicious_ids=None):
        self.malicious_ids = set(int(c) for c in (malicious_ids or []))
        self.eval_rows = []
        self.train_rows = []
        self.global_rows = []
        self.aggregation_rows = []

    def add_eval(self, server_round, client_id, loss, accuracy, num_examples):
        """
        Record one client's evaluation of the global model on its local
        validation subset (poisoned subset for malicious clients).
        """
        self.eval_rows.append({
            "round": int(server_round),
            "client": int(client_id),
            "malicious": int(client_id in self.malicious_ids),
            "loss": float(loss),
            "accuracy": float(accuracy),
            "num_examples": int(num_examples),
        })

    def add_train(self, server_round, client_id, train_loss, train_accuracy, num_examples):
        """
        Record one client's local training metrics from the fit step.
        """
        self.train_rows.append({
            "round": int(server_round),
            "client": int(client_id),
            "malicious": int(client_id in self.malicious_ids),
            "train_loss": float(train_loss),
            "train_accuracy": float(train_accuracy),
            "num_examples": int(num_examples),
        })

    def add_global(self, server_round, metrics):
        """
        Record the server-side evaluation on the clean global test set.
        :param metrics: dict with loss, accuracy, balanced_accuracy, macro_f1
        """
        row = {"round": int(server_round)}
        row.update({k: float(v) for k, v in metrics.items()})
        self.global_rows.append(row)

    def add_aggregation(self, server_round, aggregated_ids, excluded_ids, failures, seconds=None):
        """
        Record which clients were aggregated and which were excluded.
        Conventional FedAvg never excludes anyone, so excluded_ids is empty
        """
        self.aggregation_rows.append({
            "round": int(server_round),
            "num_aggregated": len(aggregated_ids),
            "aggregated_clients": " ".join(str(c) for c in sorted(aggregated_ids)),
            "num_excluded": len(excluded_ids),
            "excluded_clients": " ".join(str(c) for c in sorted(excluded_ids)),
            "num_malicious_aggregated": sum(1 for c in aggregated_ids if c in self.malicious_ids),
            "num_failures": int(failures),
            "round_seconds": None if seconds is None else round(float(seconds), 2),
        })

    def _wide_table(self, eval_df, value):
        """
        Pivot the long eval table to rounds x clients.
        """
        wide = eval_df.pivot(index="round", columns="client", values=value).sort_index(axis=1)
        wide.columns = [client_column_name(int(c), self.malicious_ids) for c in wide.columns]
        return wide.reset_index()

    def _average_table(self, eval_df, value):
        """
        Per-round averages among clients.
          mean_all_clients      simple mean of the per-client values
          weighted_mean_all     mean weighted by local validation size
                                (this is what Flower's FedAvg reports)
          mean_benign_clients   mean over benign clients only
          mean_malicious_clients mean over malicious clients only
        """
        rows = []
        for server_round, group in eval_df.groupby("round"):
            weights = group["num_examples"].to_numpy(dtype=float)
            values = group[value].to_numpy(dtype=float)
            benign = group[group["malicious"] == 0][value]
            malicious = group[group["malicious"] == 1][value]
            rows.append({
                "round": int(server_round),
                "mean_all_clients": float(values.mean()),
                "weighted_mean_all": float(np.average(values, weights=weights)) if weights.sum() > 0 else float("nan"),
                "mean_benign_clients": float(benign.mean()) if len(benign) else float("nan"),
                "mean_malicious_clients": float(malicious.mean()) if len(malicious) else float("nan"),
                "num_clients": int(len(group)),
            })
        return pd.DataFrame(rows)

    def save(self, out_dir, tag, config):
        """
        Write every CSV and the config JSON.
        :param out_dir: results/no_attack or results/attack
        :param tag: prefix for all file names, e.g. "clean" or "p70_flip50"
        :param config: dict of run settings
        :return: dict of written file paths
        """
        csv_prefix = os.path.join(run_subdir(out_dir, CSV_DIR, create=True), tag + "_")
        json_prefix = os.path.join(run_subdir(out_dir, JSON_DIR, create=True), tag + "_")

        eval_df = pd.DataFrame(self.eval_rows).sort_values(["round", "client"])

        # The four required metric tables.
        self._wide_table(eval_df, "loss").to_csv(csv_prefix + "loss_per_client.csv", index=False)
        self._wide_table(eval_df, "accuracy").to_csv(csv_prefix + "accuracy_per_client.csv", index=False)
        self._average_table(eval_df, "loss").to_csv(csv_prefix + "avg_loss.csv", index=False)
        self._average_table(eval_df, "accuracy").to_csv(csv_prefix + "avg_accuracy.csv", index=False)

        # Supporting records.
        eval_df.to_csv(csv_prefix + "per_client_eval_long.csv", index=False)
        if self.train_rows:
            pd.DataFrame(self.train_rows).sort_values(["round", "client"]).to_csv(
                csv_prefix + "train_per_client_long.csv", index=False)
        if self.global_rows:
            pd.DataFrame(self.global_rows).sort_values("round").to_csv(
                csv_prefix + "global_test.csv", index=False)
        if self.aggregation_rows:
            pd.DataFrame(self.aggregation_rows).sort_values("round").to_csv(
                csv_prefix + "aggregation_log.csv", index=False)

        # Run settings.
        config = dict(config)
        config["malicious_clients"] = sorted(self.malicious_ids)
        with open(json_prefix + "config.json", "w", encoding="utf-8") as file:
            json.dump(config, file, indent=2)

        written = {}
        for kind in (CSV_DIR, JSON_DIR):
            folder = run_subdir(out_dir, kind)
            for name in os.listdir(folder):
                if name.startswith(tag + "_"):
                    written[name] = os.path.join(folder, name)
        return written
