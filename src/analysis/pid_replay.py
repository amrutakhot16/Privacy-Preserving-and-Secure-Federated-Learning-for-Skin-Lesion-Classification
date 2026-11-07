"""
Fast tuning of the PID coefficients and threshold.

The spec says Kp, Ki, Kd "must be investigated empirically". Running the
full simulation for every combination would take hours, so we tune in two
steps:

  1. Run each scenario ONCE in observe-only mode. The PID strategy then
     computes and logs every client's distance to the centroid in every
     round but excludes nobody:

       python src/federated/run_fl.py --tag pid_observe_p70_flip50 --out results/tuning \
           --poisoned-dir data/poisoned/p70_flip50 --defense pid --threshold-mode none

  2. Replay the logged distances with many (Kp, Ki, Kd, threshold)
     combinations here, in seconds, and count how many malicious clients
     each combination would catch and how many benign clients it would
     wrongly remove:

       python src/analysis/pid_replay.py results/tuning/csv/pid_observe_p70_flip50_pid_scores.csv

     Several files can be given at once (e.g. one per scenario); the
     ranking then adds up the errors over all of them.

  3. Confirm the best combination with a real run (run_all.py --defense pid ...).


Output: <first input folder>/pid_replay_results.csv (one row per
combination and scenario) and a printed top-10 ranking.

@author Amruta Kirtikumar Khot, ak3792@rit.edu
@author Cazal Tirkey, ct9857@rit.edu
"""

import argparse
import itertools
import os

import numpy as np
import pandas as pd


def replay(scores_table, kp, ki, kd, threshold_mode, threshold_value, min_keep=2, column="distance"):
    """
    Re-run the PID decision rule on logged distances.
    :param scores_table: DataFrame with round, client, malicious, distance
    :param threshold_mode: "median", "fixed" or "std"
    :param threshold_value: k for k*median, the fixed threshold, or k for mean + k*std
    :param column: which logged distance to use: distance (the one used in the run),
                   distance_all, distance_classifier or distance_last
    :return: dict with detection counts
    """
    rounds = sorted(scores_table["round"].unique())
    by_round = {r: rows for r, rows in scores_table.groupby("round")}
    malicious = scores_table.groupby("client")["malicious"].first().to_dict()
    history = {}
    excluded_round = {}

    for server_round in rounds:
        rows = by_round[server_round]
        rows = rows[~rows["client"].isin(excluded_round.keys())]
        if rows.empty:
            break
        clients = rows["client"].to_numpy()
        distances = rows[column].to_numpy(dtype=float)
        scores = []
        for client_id, distance in zip(clients, distances):
            past = history.get(client_id, [])
            score = kp * distance
            if past:
                score += ki * float(np.sum(past)) + kd * (distance - past[-1])
            scores.append(score)
        scores = np.array(scores)
        if threshold_mode == "median":
            threshold = threshold_value * np.median(scores)
        elif threshold_mode == "fixed":
            threshold = threshold_value
        else:
            threshold = scores.mean() + threshold_value * scores.std()
        exclude = scores > threshold
        if (~exclude).sum() < min_keep:
            exclude = np.ones(len(scores), dtype=bool)
            exclude[np.argsort(scores)[:min_keep]] = False
        for client_id, distance, is_excluded in zip(clients, distances, exclude):
            history.setdefault(client_id, []).append(float(distance))
            if is_excluded:
                excluded_round[client_id] = server_round

    malicious_ids = [c for c, m in malicious.items() if m]
    benign_ids = [c for c, m in malicious.items() if not m]
    true_positives = [c for c in malicious_ids if c in excluded_round]
    false_positives = [c for c in benign_ids if c in excluded_round]
    detection_rounds = [excluded_round[c] for c in true_positives]
    return {
        "malicious_clients": len(malicious_ids),
        "true_positive": len(true_positives),
        "false_negative": len(malicious_ids) - len(true_positives),
        "false_positive": len(false_positives),
        "true_negative": len(benign_ids) - len(false_positives),
        "mean_detection_round": float(np.mean(detection_rounds)) if detection_rounds else float("nan"),
        "last_detection_round": int(max(detection_rounds)) if detection_rounds else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Replay logged PID distances with many settings.")
    parser.add_argument("scores", nargs="+", help="<tag>_pid_scores.csv files from observe-only runs")
    parser.add_argument("--kp", type=float, nargs="+", default=[1.0])
    parser.add_argument("--ki", type=float, nargs="+", default=[0.0, 0.02, 0.05, 0.1])
    parser.add_argument("--kd", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    parser.add_argument("--median-k", type=float, nargs="*", default=[1.2, 1.3, 1.5, 1.75, 2.0],
                        help="k values to try for threshold = k * median score")
    parser.add_argument("--fixed", type=float, nargs="*", default=[1.2, 1.5, 2.0, 2.5, 3.0],
                        help="fixed thresholds to try")
    parser.add_argument("--std-k", type=float, nargs="*", default=[1.0, 1.5, 2.0, 2.5],
                        help="k values to try for threshold = mean + k*std")
    parser.add_argument("--min-keep", type=int, default=2)
    parser.add_argument("--layers", nargs="+", default=None,
                        help="layer choices to compare: all classifier last "
                             "(default: every distance_* column found in the files)")
    parser.add_argument("--out", default=None, help="output CSV (default: next to the first input)")
    args = parser.parse_args()

    first = pd.read_csv(args.scores[0], nrows=1)
    if args.layers:
        columns = ["distance_" + layers for layers in args.layers]
    else:
        columns = [c for c in first.columns if c.startswith("distance_")] or ["distance"]

    thresholds = [("median", k) for k in args.median_k] + [("fixed", t) for t in args.fixed] + [("std", k) for k in args.std_k]
    rows = []
    for path in args.scores:
        table = pd.read_csv(path)
        scenario = os.path.basename(path).replace("_pid_scores.csv", "")
        for column, kp, ki, kd, (mode, value) in itertools.product(columns, args.kp, args.ki, args.kd, thresholds):
            result = replay(table, kp, ki, kd, mode, value, args.min_keep, column)
            rows.append({"scenario": scenario, "layers": column.replace("distance_", "") if column != "distance" else "run",
                         "kp": kp, "ki": ki, "kd": kd,
                         "threshold_mode": mode, "threshold_value": value, **result})

    results = pd.DataFrame(rows)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.scores[0])), "pid_replay_results.csv")
    results.to_csv(out, index=False)

    keys = ["layers", "kp", "ki", "kd", "threshold_mode", "threshold_value"]
    ranking = (results.assign(errors=results["false_negative"] + results["false_positive"])
               .groupby(keys, as_index=False)
               .agg(errors=("errors", "sum"), false_negative=("false_negative", "sum"),
                    false_positive=("false_positive", "sum"),
                    mean_detection_round=("mean_detection_round", "mean"))
               .sort_values(["errors", "false_positive", "mean_detection_round"]))
    print("Top 10 settings (fewest missed malicious + wrongly removed benign clients, summed over scenarios):")
    print(ranking.head(10).to_string(index=False))
    print("\nAll results: " + out)


if __name__ == "__main__":
    main()
