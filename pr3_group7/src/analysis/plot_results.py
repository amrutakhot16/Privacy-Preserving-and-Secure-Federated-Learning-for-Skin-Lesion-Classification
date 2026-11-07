"""
Plots for Part 2 (conventional FL) and Part 3 (FL with PID).

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

Part 3 (PID runs, tags start with "pid_") additionally get:
  T_6_pid_scores.png            PID score of every client per round with the
                                threshold; X marks the round a client is excluded
  T_7_client_removal.png        client removal dynamics: clients x rounds map
                                (active / excluded, benign / malicious) and the
                                number of active benign and malicious clients
  pid_5a_severity_final_loss.png, pid_5b_severity_final_accuracy.png,
  pid_severity_metrics.csv      attack severity for the PID runs
  compare_<run>_avg_loss.png, compare_<run>_avg_accuracy.png
                                conventional FedAvg vs FL with PID on one plot

Benign clients are drawn as solid blue lines and malicious clients as
dashed red lines, so poisoned clients can be told apart without relying
on colour alone.

Usage:
  python src/analysis/plot_results.py --out results/no_attack --tag clean
  python src/analysis/plot_results.py --out results/attack --tag p70_flip50
  python src/analysis/plot_results.py --severity     (after all runs are finished)
  python src/analysis/plot_results.py --out results/attack --tag pid_p70_flip100 --pid
  python src/analysis/plot_results.py --severity --prefix pid_
  python src/analysis/plot_results.py --compare p70_flip100

@author Amruta Kirtikumar Khot, ak3792@rit.edu
@author Cazal Tirkey, ct9857@rit.edu
"""

import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.colors
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
    method = ""
    if tag.startswith("pid_"):
        method = "FL with PID: "
        tag = tag[len("pid_"):]
        for layers, name in (("last_", "output layer"), ("classifier_", "classifier layers")):
            if tag.startswith(layers):
                method = "FL with PID (" + name + "): "
                tag = tag[len(layers):]
    if tag == "clean":
        return method + "no attack"
    if tag.startswith("p") and "_flip" in tag:
        poison, flip = tag[1:].split("_flip")
        return method + poison + "% poisoned clients, " + flip + "% labels flipped"
    return method + tag


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


def build_severity_table(clean_dir, attack_dir, num_clients=20, main_poison=70, main_flip=100, prefix=""):
    """
    Collect final metrics for both severity sweeps:
      sweep "flip_rate":          70% poisoned clients, flip 0/25/50/100%
      sweep "poisoned_clients":   100% flip, 0/10/30/50/70% poisoned clients
    The clean run is the 0% point of both sweeps.
    :param prefix: "" for conventional FedAvg runs, "pid_" for the PID runs
    """
    rows = []
    # All attack runs share results/attack/; each run's config is
    # results/attack/json/<tag>_config.json.
    json_dir = run_subdir(attack_dir, JSON_DIR)
    pattern = re.compile("^" + re.escape(prefix) + r"p(\d+)_flip(\d+)_config\.json$")
    for name in sorted(os.listdir(json_dir)):
        match = pattern.match(name)
        if not match:
            continue
        tag = name[: -len("_config.json")]
        config_path = os.path.join(json_dir, name)
        poison, flip = int(match.group(1)), int(match.group(2))
        with open(config_path, encoding="utf-8") as file:
            config = json.load(file)
        n_malicious = len(config["malicious_clients"])
        base = {"tag": tag, "poisoned_percent": poison, "poisoned_clients": n_malicious, "flip_percent": flip}
        metrics = final_metrics(attack_dir, tag)
        metrics.update(config.get("detection", {}))
        if poison == main_poison:
            rows.append({"sweep": "flip_rate", **base, **metrics})
        if flip == main_flip:
            rows.append({"sweep": "poisoned_clients", **base, **metrics})

    clean_tag = prefix + "clean"
    clean = final_metrics(clean_dir, clean_tag)
    clean_config = os.path.join(run_subdir(clean_dir, JSON_DIR), clean_tag + "_config.json")
    if os.path.isfile(clean_config):
        with open(clean_config, encoding="utf-8") as file:
            clean.update(json.load(file).get("detection", {}))
    rows.append({"sweep": "flip_rate", "tag": clean_tag, "poisoned_percent": main_poison,
                 "poisoned_clients": 0, "flip_percent": 0, **clean})
    rows.append({"sweep": "poisoned_clients", "tag": clean_tag, "poisoned_percent": 0,
                 "poisoned_clients": 0, "flip_percent": main_flip, **clean})

    table = pd.DataFrame(rows)
    table = table.sort_values(["sweep", "flip_percent", "poisoned_percent"]).reset_index(drop=True)
    return table


def plot_severity(clean_dir, attack_dir, prefix=""):
    """
    Final loss and accuracy vs attack severity (two figures, two panels each).
    :param prefix: "" for conventional FedAvg, "pid_" for FL with PID
    """
    method = _run_label(prefix + "clean").replace(": no attack", "") if prefix.startswith("pid_") else "conventional FedAvg"
    table = build_severity_table(clean_dir, attack_dir, prefix=prefix)
    table.to_csv(os.path.join(run_subdir(attack_dir, CSV_DIR, create=True), prefix + "severity_metrics.csv"), index=False)
    plots_dir = run_subdir(attack_dir, PLOTS_DIR, create=True)

    flip = table[table["sweep"] == "flip_rate"].sort_values("flip_percent")
    poison = table[table["sweep"] == "poisoned_clients"].sort_values("poisoned_percent")

    for metric, ylabel, filename in (
        ("loss", "Final loss", prefix + "5a_severity_final_loss.png"),
        ("accuracy", "Final accuracy", prefix + "5b_severity_final_accuracy.png"),
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
                # For PID runs this is NaN when every malicious client was excluded.
                axis.plot(data[xcol], data["final_avg_" + metric + "_malicious"], color=MALICIOUS_COLOR,
                          marker="^", linewidth=1.8, linestyle="--", label="Avg over malicious clients")
            if "final_test_" + metric in data:
                axis.plot(data[xcol], data["final_test_" + metric], color=TEST_COLOR, marker="D",
                          linewidth=1.8, linestyle=":", label="Global model, clean test set")
            if metric == "accuracy" and "final_test_balanced_accuracy" in data:
                # Balanced accuracy (mean per-class recall) is not inflated by the
                # majority class "nv", so it shows the real damage of the attack.
                axis.plot(data[xcol], data["final_test_balanced_accuracy"], color="#7b3fb5",
                          marker="P", markersize=8, linewidth=2.2,
                          label="Global model, clean test set (balanced acc.)")
            axis.set_xticks(sorted(data[xcol].unique()))
            _style(axis, title, ylabel)
            axis.set_xlabel(xlabel)
            if metric == "accuracy":
                axis.set_ylim(-0.02, 1.02)
        axes[0].legend(fontsize=8, frameon=False)
        figure.suptitle(ylabel + " after the last round vs attack severity (" + method + ")", fontsize=11)
        figure.tight_layout()
        figure.savefig(os.path.join(plots_dir, filename), dpi=180)
        plt.close(figure)

    print("Wrote " + prefix + "severity_metrics.csv, " + prefix + "5a_severity_final_loss.png, "
          + prefix + "5b_severity_final_accuracy.png in " + attack_dir)


# ----------------------------------------------------------------------
# Part 3: PID defense
# ----------------------------------------------------------------------

def plot_pid_scores(out_dir, tag):
    """
    PID score of every client over rounds, with the threshold used in each
    round. An X marks the round in which a client was excluded; after that
    the client no longer takes part, so its line stops.
    """
    pid = pd.read_csv(os.path.join(run_subdir(out_dir, CSV_DIR), tag + "_pid_scores.csv"))
    figure, axis = plt.subplots(figsize=(9, 5))
    for client_id, rows in pid.groupby("client"):
        malicious = bool(rows["malicious"].iloc[0])
        axis.plot(rows["round"], rows["pid_score"],
                  color=MALICIOUS_COLOR if malicious else BENIGN_COLOR,
                  linestyle="--" if malicious else "-", linewidth=1.1, alpha=0.75)
        excluded = rows[rows["excluded"] == 1]
        axis.scatter(excluded["round"], excluded["pid_score"], marker="x", s=55, linewidths=2,
                     color=MALICIOUS_COLOR if malicious else BENIGN_COLOR, zorder=3)

    thresholds = pid.groupby("round")["threshold"].first()
    if np.isfinite(thresholds).any():
        axis.plot(thresholds.index, thresholds.values, color=ALL_COLOR, linewidth=2.2, label="Threshold")
    axis.plot([], [], color=BENIGN_COLOR, label="Benign client")
    axis.plot([], [], color=MALICIOUS_COLOR, linestyle="--", label="Malicious client")
    axis.scatter([], [], marker="x", color=ALL_COLOR, label="Excluded in this round")
    _style(axis, "PID score per client over rounds - " + _run_label(tag), "PID score u(t)")
    axis.legend(fontsize=8, frameon=False)
    figure.tight_layout()
    figure.savefig(os.path.join(run_subdir(out_dir, PLOTS_DIR, create=True), tag + "_6_pid_scores.png"), dpi=180)
    plt.close(figure)


def plot_client_removal(out_dir, tag):
    """
    Client removal dynamics.
    Left: map of clients x rounds. Each cell shows whether the client was
    aggregated in that round, and once excluded whether that exclusion was
    correct (malicious client removed) or a false alarm (benign client removed).
    Right: number of benign and malicious clients still aggregated per round.
    """
    pid = pd.read_csv(os.path.join(run_subdir(out_dir, CSV_DIR), tag + "_pid_scores.csv"))
    rounds = sorted(pid["round"].unique())
    clients = sorted(pid["client"].unique())
    malicious = pid.groupby("client")["malicious"].first()
    excluded_round = pid[pid["excluded"] == 1].groupby("client")["round"].min()

    # 0 aggregated benign, 1 aggregated malicious, 2 excluded malicious, 3 excluded benign
    state = np.zeros((len(clients), len(rounds)))
    for i, client_id in enumerate(clients):
        for j, server_round in enumerate(rounds):
            removed = client_id in excluded_round.index and server_round >= excluded_round[client_id]
            if removed:
                state[i, j] = 2 if malicious[client_id] else 3
            else:
                state[i, j] = 1 if malicious[client_id] else 0

    colours = ["#cfe0fa", "#f6c9c2", "#7a1c10", "#e3a21a"]
    labels = ["Aggregated (benign)", "Aggregated (malicious)",
              "Excluded - malicious (correct)", "Excluded - benign (false alarm)"]
    cmap = matplotlib.colors.ListedColormap(colours)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.5, 1]})
    axes[0].imshow(state, aspect="auto", cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="nearest")
    axes[0].set_yticks(range(len(clients)))
    axes[0].set_yticklabels(["c" + format(c, "02d") + (" M" if malicious[c] else "") for c in clients], fontsize=7)
    axes[0].set_xticks(range(len(rounds)))
    axes[0].set_xticklabels(rounds, fontsize=7)
    axes[0].set_xlabel("Aggregation round")
    axes[0].set_ylabel("Client (M = malicious)")
    axes[0].set_title("Client removal map", fontsize=11)
    handles = [matplotlib.patches.Patch(color=c, label=l) for c, l in zip(colours, labels)]
    axes[0].legend(handles=handles, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12),
                   ncol=2, frameon=False)

    active_benign = [(state[:, j] == 0).sum() for j in range(len(rounds))]
    active_malicious = [(state[:, j] == 1).sum() for j in range(len(rounds))]
    axes[1].step(rounds, active_benign, where="mid", color=BENIGN_COLOR, linewidth=2, label="Benign clients aggregated")
    axes[1].step(rounds, active_malicious, where="mid", color=MALICIOUS_COLOR, linewidth=2, linestyle="--",
                 label="Malicious clients aggregated")
    _style(axes[1], "Clients still aggregated", "Number of clients")
    axes[1].set_ylim(-0.5, len(clients) + 0.5)
    axes[1].legend(fontsize=8, frameon=False)

    figure.suptitle("Client removal dynamics - " + _run_label(tag), fontsize=11)
    figure.tight_layout()
    figure.savefig(os.path.join(run_subdir(out_dir, PLOTS_DIR, create=True), tag + "_7_client_removal.png"), dpi=180)
    plt.close(figure)


def plot_pid_run(out_dir, tag):
    """
    All plots of one PID run: the four required history plots, the clean
    test-set plot, the PID scores and the client removal dynamics.
    """
    plot_run(out_dir, tag)
    plot_pid_scores(out_dir, tag)
    plot_client_removal(out_dir, tag)


def plot_comparison(conventional_dir, pid_dir, base_tag, out_dir=None, pid_prefix="pid_"):
    """
    Conventional FedAvg vs FL with PID on one plot (needed again in Part 4).
    Draws average loss and average accuracy among clients for both methods.
    :param conventional_dir: folder of the conventional run (results/attack)
    :param pid_dir: folder of the PID run
    :param base_tag: scenario tag without prefix, e.g. "p70_flip100" or "clean"
    :param pid_prefix: "pid_", "pid_last_" or "pid_classifier_"
    """
    out_dir = out_dir or pid_dir
    for value, ylabel in (("loss", "Average evaluation loss"), ("accuracy", "Average evaluation accuracy")):
        conventional_path = os.path.join(run_subdir(conventional_dir, CSV_DIR), base_tag + "_avg_" + value + ".csv")
        pid_path = os.path.join(run_subdir(pid_dir, CSV_DIR), pid_prefix + base_tag + "_avg_" + value + ".csv")
        if not (os.path.isfile(conventional_path) and os.path.isfile(pid_path)):
            print("Skipping comparison for " + base_tag + " (" + value + "): missing CSV")
            continue
        conventional = pd.read_csv(conventional_path)
        pid = pd.read_csv(pid_path)
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.plot(conventional["round"], conventional["mean_all_clients"], color="#888888", linewidth=2.2,
                  marker="o", markersize=3, label="Conventional FedAvg - all clients")
        pid_name = _run_label(pid_prefix + "clean").replace(": no attack", "")
        axis.plot(pid["round"], pid["mean_all_clients"], color=ALL_COLOR, linewidth=2.2,
                  marker="s", markersize=3, label=pid_name + " - clients still in the federation")
        if conventional["mean_benign_clients"].notna().any() and conventional["mean_malicious_clients"].notna().any():
            axis.plot(conventional["round"], conventional["mean_benign_clients"], color="#8fb3ee",
                      linewidth=1.6, linestyle=":", label="Conventional FedAvg - benign clients")
            axis.plot(pid["round"], pid["mean_benign_clients"], color=BENIGN_COLOR, linewidth=1.6,
                      linestyle="--", label=pid_name + " - benign clients")
        _style(axis, ylabel + ": conventional vs PID - " + _run_label(base_tag), ylabel)
        if value == "accuracy":
            axis.set_ylim(-0.02, 1.02)
        axis.legend(fontsize=8, frameon=False)
        figure.tight_layout()
        name = "compare_" + ("" if pid_prefix == "pid_" else pid_prefix[len("pid_"):]) + base_tag
        figure.savefig(os.path.join(run_subdir(out_dir, PLOTS_DIR, create=True),
                                    name + "_avg_" + value + ".png"), dpi=180)
        plt.close(figure)


def main():
    # Scripts live in src/<group>/, so the project root is three levels up.
    project_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser(description="Re-draw Part 2 / Part 3 plots from saved CSVs.")
    parser.add_argument("--out", help="results/no_attack or results/attack")
    parser.add_argument("--tag", help="run tag, e.g. clean or p70_flip50")
    parser.add_argument("--severity", action="store_true")
    parser.add_argument("--prefix", default="", help='"" for conventional runs, "pid_" for PID runs')
    parser.add_argument("--pid", action="store_true", help="the --tag run is a PID run: also draw PID plots")
    parser.add_argument("--compare", nargs="*", help="scenario tags to compare, e.g. clean p70_flip100")
    parser.add_argument("--clean-dir", default=os.path.join(project_folder, "results", "no_attack"))
    parser.add_argument("--attack-dir", default=os.path.join(project_folder, "results", "attack"))
    args = parser.parse_args()

    if args.tag:
        if args.pid:
            plot_pid_run(args.out, args.tag)
        else:
            plot_run(args.out, args.tag)
    if args.severity:
        plot_severity(args.clean_dir, args.attack_dir, prefix=args.prefix)
    for base_tag in args.compare or []:
        run_dir = args.clean_dir if base_tag == "clean" else args.attack_dir
        plot_comparison(run_dir, run_dir, base_tag, pid_prefix=args.prefix or "pid_")


if __name__ == "__main__":
    main()
