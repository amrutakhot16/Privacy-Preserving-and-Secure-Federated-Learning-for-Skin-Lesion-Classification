"""
Plots

For every run (tag) it draws the four required history plots into the
run's plots/ folder:
  T_1_loss_per_client.png       loss of the global model on each client's
                                local validation subset, per round
  T_2_avg_loss.png              average loss among clients, per round
  T_3_accuracy_per_client.png   accuracy per client, per round
  T_4_avg_accuracy.png          average accuracy among clients, per round
plus an extra T_global_test.png (clean held-out test set).

For results/attack/ it also makes the attack-severity results
(plots in plots/, the table in csv/):
  5a_severity_final_loss.png       final loss vs flipping rate and vs % poisoned clients
  5b_severity_final_accuracy.png   final accuracy vs the same two axes
  severity_metrics.csv             the numbers behind both plots

Benign clients are drawn as solid blue lines and malicious clients as
dashed red lines, so poisoned clients can be told apart without relying
on colour alone.


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

from metrics_logger import CSV_DIR, JSON_DIR, PLOTS_DIR, run_subdir


BENIGN_COLOR = "#2a6fdb"
MALICIOUS_COLOR = "#d6402b"
ALL_COLOR = "#222222"
TEST_COLOR = "#2e9d5b"


def _style(axis, title, ylabel):
    axis.set_title(title, fontsize=11)
    axis.set_xlabel("Aggregation round")
    axis.set_ylabel(ylabel)
    axis.grid(True, color="#dddddd", linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def _run_label(tag):
    """
    Human-readable description of a run tag for plot titles.
    """
    if tag == "clean":
        return "no attack"
    if tag.startswith("p") and "_flip" in tag:
        poison, flip = tag[1:].split("_flip")
        return poison + "% poisoned clients, " + flip + "% labels flipped"
    return tag


def plot_per_client(out_dir, tag, value, ylabel, filename, title):
    """
    One line per client. Benign = solid blue shades, malicious = dashed red shades.
    """
    table = pd.read_csv(os.path.join(run_subdir(out_dir, CSV_DIR), tag + "_" + value + "_per_client.csv"))
    client_columns = [c for c in table.columns if c.startswith("client_")]
    benign = [c for c in client_columns if c.endswith("_benign")]
    malicious = [c for c in client_columns if c.endswith("_malicious")]

    figure, axis = plt.subplots(figsize=(9, 5))
    blues = plt.cm.Blues(np.linspace(0.45, 0.95, max(1, len(benign))))
    reds = plt.cm.Reds(np.linspace(0.45, 0.95, max(1, len(malicious))))

    for colour, column in zip(blues, benign):
        axis.plot(table["round"], table[column], color=colour, linewidth=1.4,
                  label=column.replace("client_", "c").replace("_benign", " benign"))
    for colour, column in zip(reds, malicious):
        axis.plot(table["round"], table[column], color=colour, linewidth=1.4, linestyle="--",
                  label=column.replace("client_", "c").replace("_malicious", " MALICIOUS"))

    _style(axis, title + " - " + _run_label(tag), ylabel)
    if value == "accuracy":
        axis.set_ylim(-0.02, 1.02)
    axis.legend(fontsize=7, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    figure.tight_layout()
    figure.savefig(os.path.join(run_subdir(out_dir, PLOTS_DIR, create=True), tag + "_" + filename), dpi=180)
    plt.close(figure)


def plot_average(out_dir, tag, value, ylabel, filename, title):
    """
    Average among clients. The attack runs also show benign-only and
    malicious-only averages so the effect of poisoning is visible.
    """
    table = pd.read_csv(os.path.join(run_subdir(out_dir, CSV_DIR), tag + "_avg_" + value + ".csv"))
    figure, axis = plt.subplots(figsize=(8, 4.5))

    axis.plot(table["round"], table["mean_all_clients"], color=ALL_COLOR, linewidth=2.2,
              marker="o", markersize=3, label="Average over all clients")

    has_malicious = table["mean_malicious_clients"].notna().any()
    if has_malicious:
        axis.plot(table["round"], table["mean_benign_clients"], color=BENIGN_COLOR, linewidth=1.8,
                  label="Average over benign clients")
        axis.plot(table["round"], table["mean_malicious_clients"], color=MALICIOUS_COLOR,
                  linewidth=1.8, linestyle="--", label="Average over malicious clients")

    _style(axis, title + " - " + _run_label(tag), ylabel)
    if value == "accuracy":
        axis.set_ylim(-0.02, 1.02)
    axis.legend(fontsize=8, frameon=False)
    figure.tight_layout()
    figure.savefig(os.path.join(run_subdir(out_dir, PLOTS_DIR, create=True), tag + "_" + filename), dpi=180)
    plt.close(figure)


def plot_global_test(out_dir, tag):
    """
    Extra plot: the global model on the clean held-out test set.
    """
    path = os.path.join(run_subdir(out_dir, CSV_DIR), tag + "_global_test.csv")
    if not os.path.isfile(path):
        return
    table = pd.read_csv(path)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(table["round"], table["loss"], color=ALL_COLOR, linewidth=2)
    _style(axes[0], "Global model loss on clean test set", "Cross-entropy loss")
    axes[1].plot(table["round"], table["accuracy"], color=ALL_COLOR, linewidth=2, label="Accuracy")
    axes[1].plot(table["round"], table["balanced_accuracy"], color=TEST_COLOR, linewidth=2,
                 linestyle="--", label="Balanced accuracy")
    _style(axes[1], "Global model accuracy on clean test set", "Score")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend(fontsize=8, frameon=False)
    figure.suptitle(_run_label(tag), fontsize=11)
    figure.tight_layout()
    figure.savefig(os.path.join(run_subdir(out_dir, PLOTS_DIR, create=True), tag + "_global_test.png"), dpi=180)
    plt.close(figure)


def plot_run(out_dir, tag):
    """
    Make all plots for one run.
    """
    plot_per_client(out_dir, tag, "loss", "Evaluation loss (cross-entropy)",
                    "1_loss_per_client.png", "Loss per client over rounds")
    plot_average(out_dir, tag, "loss", "Average evaluation loss",
                 "2_avg_loss.png", "Average loss among clients over rounds")
    plot_per_client(out_dir, tag, "accuracy", "Evaluation accuracy",
                    "3_accuracy_per_client.png", "Evaluation accuracy per client over rounds")
    plot_average(out_dir, tag, "accuracy", "Average evaluation accuracy",
                 "4_avg_accuracy.png", "Average evaluation accuracy among clients over rounds")
    plot_global_test(out_dir, tag)


def final_metrics(run_dir, tag):
    """
    Final-round metrics of one run (used for the severity analysis).
    """
    csv_dir = run_subdir(run_dir, CSV_DIR)
    loss = pd.read_csv(os.path.join(csv_dir, tag + "_avg_loss.csv")).iloc[-1]
    accuracy = pd.read_csv(os.path.join(csv_dir, tag + "_avg_accuracy.csv")).iloc[-1]
    row = {
        "final_round": int(loss["round"]),
        "final_avg_loss_all": loss["mean_all_clients"],
        "final_avg_loss_benign": loss["mean_benign_clients"],
        "final_avg_loss_malicious": loss["mean_malicious_clients"],
        "final_avg_accuracy_all": accuracy["mean_all_clients"],
        "final_avg_accuracy_benign": accuracy["mean_benign_clients"],
        "final_avg_accuracy_malicious": accuracy["mean_malicious_clients"],
    }
    test_path = os.path.join(run_subdir(run_dir, CSV_DIR), tag + "_global_test.csv")
    if os.path.isfile(test_path):
        test = pd.read_csv(test_path).iloc[-1]
        row.update({
            "final_test_loss": test["loss"],
            "final_test_accuracy": test["accuracy"],
            "final_test_balanced_accuracy": test["balanced_accuracy"],
            "final_test_macro_f1": test["macro_f1"],
        })
    return row


def build_severity_table(clean_dir, attack_dir, num_clients=20, main_poison=70, main_flip=100):
    """
    Collect final metrics for both severity sweeps:
      sweep "flip_rate":          70% poisoned clients, flip 0/25/50/100%
      sweep "poisoned_clients":   100% flip, 0/10/30/50/70% poisoned clients
    The clean run is the 0% point of both sweeps.
    """
    rows = []
    
    json_dir = run_subdir(attack_dir, JSON_DIR)
    for name in sorted(os.listdir(json_dir)):
        if not name.startswith("p") or not name.endswith("_config.json"):
            continue
        tag = name[: -len("_config.json")]
        config_path = os.path.join(json_dir, name)
        poison, flip = (int(v) for v in tag[1:].split("_flip"))
        with open(config_path, encoding="utf-8") as file:
            n_malicious = len(json.load(file)["malicious_clients"])
        base = {"tag": tag, "poisoned_percent": poison, "poisoned_clients": n_malicious, "flip_percent": flip}
        metrics = final_metrics(attack_dir, tag)
        if poison == main_poison:
            rows.append({"sweep": "flip_rate", **base, **metrics})
        if flip == main_flip:
            rows.append({"sweep": "poisoned_clients", **base, **metrics})

    clean = final_metrics(clean_dir, "clean")
    rows.append({"sweep": "flip_rate", "tag": "clean", "poisoned_percent": main_poison,
                 "poisoned_clients": 0, "flip_percent": 0, **clean})
    rows.append({"sweep": "poisoned_clients", "tag": "clean", "poisoned_percent": 0,
                 "poisoned_clients": 0, "flip_percent": main_flip, **clean})

    table = pd.DataFrame(rows)
    table = table.sort_values(["sweep", "flip_percent", "poisoned_percent"]).reset_index(drop=True)
    return table


def plot_severity(clean_dir, attack_dir):
    """
    Final loss and accuracy vs attack severity (two figures, two panels each).
    """
    table = build_severity_table(clean_dir, attack_dir)
    table.to_csv(os.path.join(run_subdir(attack_dir, CSV_DIR, create=True), "severity_metrics.csv"), index=False)
    plots_dir = run_subdir(attack_dir, PLOTS_DIR, create=True)

    flip = table[table["sweep"] == "flip_rate"].sort_values("flip_percent")
    poison = table[table["sweep"] == "poisoned_clients"].sort_values("poisoned_percent")

    for metric, ylabel, filename in (
        ("loss", "Final loss", "5a_severity_final_loss.png"),
        ("accuracy", "Final accuracy", "5b_severity_final_accuracy.png"),
    ):
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.3))
        for axis, data, xcol, xlabel, title in (
            (axes[0], flip, "flip_percent", "Label flipping rate on malicious clients (%)",
             "70% of clients poisoned, varying flip rate"),
            (axes[1], poison, "poisoned_percent", "Poisoned clients (% of 20)",
             "100% flip rate, varying number of poisoned clients"),
        ):
            axis.plot(data[xcol], data["final_avg_" + metric + "_all"], color=ALL_COLOR,
                      marker="o", linewidth=2, label="Avg over all clients (local val)")
            axis.plot(data[xcol], data["final_avg_" + metric + "_benign"], color=BENIGN_COLOR,
                      marker="s", linewidth=1.8, label="Avg over benign clients")
            if not data["final_avg_" + metric + "_malicious"].isna().all():
                axis.plot(data[xcol], data["final_avg_" + metric + "_malicious"], color=MALICIOUS_COLOR,
                          marker="^", linewidth=1.8, linestyle="--", label="Avg over malicious clients")
            if "final_test_" + metric in data:
                axis.plot(data[xcol], data["final_test_" + metric], color=TEST_COLOR, marker="D",
                          linewidth=1.8, linestyle=":", label="Global model, clean test set")
            if metric == "accuracy" and "final_test_balanced_accuracy" in data:
                
                axis.plot(data[xcol], data["final_test_balanced_accuracy"], color="#7b3fb5",
                          marker="P", markersize=8, linewidth=2.2,
                          label="Global model, clean test set (balanced acc.)")
            axis.set_xticks(sorted(data[xcol].unique()))
            _style(axis, title, ylabel)
            axis.set_xlabel(xlabel)
            if metric == "accuracy":
                axis.set_ylim(-0.02, 1.02)
        axes[0].legend(fontsize=8, frameon=False)
        figure.suptitle(ylabel + " after the last round vs attack severity (conventional FedAvg)", fontsize=11)
        figure.tight_layout()
        figure.savefig(os.path.join(plots_dir, filename), dpi=180)
        plt.close(figure)

    print("Wrote severity_metrics.csv, 5a_severity_final_loss.png, 5b_severity_final_accuracy.png in " + attack_dir)


def main():
    project_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser(description="Re-draw plots from saved CSVs.")
    parser.add_argument("--out", help="results/no_attack or results/attack")
    parser.add_argument("--tag", help="run tag, e.g. clean or p70_flip50")
    parser.add_argument("--severity", action="store_true")
    parser.add_argument("--clean-dir", default=os.path.join(project_folder, "results", "no_attack"))
    parser.add_argument("--attack-dir", default=os.path.join(project_folder, "results", "attack"))
    args = parser.parse_args()

    if args.tag:
        plot_run(args.out, args.tag)
    if args.severity:
        plot_severity(args.clean_dir, args.attack_dir)


if __name__ == "__main__":
    main()
