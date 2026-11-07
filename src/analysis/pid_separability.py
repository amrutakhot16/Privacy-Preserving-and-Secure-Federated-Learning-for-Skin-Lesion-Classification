"""
Separability check for the PID defense.

For every PID score file it measures how well the distance to the centroid
separates malicious from honest clients in each round:

  P(mal further) = share of (malicious, honest) client pairs in which the
                   malicious client is further from the centroid
                   (1.0 = perfect separation, 0.5 = no difference,
                    below 0.5 = honest clients are further away)

It does this for every distance logged in the file (whole model, classifier,
output layer), so it shows which choice of weights carries the attack signal.
Works best on observe-only runs (--threshold-mode none), where every client
is present in every round.

Usage:
  python src/analysis/pid_separability.py results/tuning/csv/pid_observe_*_pid_scores.csv

Output: <folder of the first file>/pid_separability.csv and a printed table.

@author Amruta Kirtikumar Khot, ak3792@rit.edu
@author Cazal Tirkey, ct9857@rit.edu
"""

import argparse
import os

import numpy as np
import pandas as pd


def p_malicious_further(malicious, benign):
    """
    Probability that a random malicious client is further away than a random
    honest one (the area under the ROC curve of the distance as a detector).
    """
    if len(malicious) == 0 or len(benign) == 0:
        return float("nan")
    return float(np.mean(malicious[:, None] > benign[None, :]))


def main():
    parser = argparse.ArgumentParser(description="How well does the PID distance separate malicious clients?")
    parser.add_argument("scores", nargs="+", help="<tag>_pid_scores.csv files")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows = []
    for path in args.scores:
        table = pd.read_csv(path)
        scenario = os.path.basename(path).replace("_pid_scores.csv", "")
        if table["malicious"].sum() == 0:
            continue
        columns = [c for c in table.columns if c.startswith("distance_")] or ["distance"]
        for column in columns:
            for server_round, group in table.groupby("round"):
                malicious = group[group["malicious"] == 1][column].to_numpy()
                benign = group[group["malicious"] == 0][column].to_numpy()
                rows.append({
                    "scenario": scenario,
                    "layers": column.replace("distance_", "") if column != "distance" else "run",
                    "round": str(int(server_round)),
                    "mean_D_benign": float(benign.mean()) if len(benign) else float("nan"),
                    "mean_D_malicious": float(malicious.mean()) if len(malicious) else float("nan"),
                    "p_malicious_further": p_malicious_further(malicious, benign),
                })
            # Summed over all rounds, i.e. what the integral term sees.
            totals = table.groupby(["client", "malicious"])[column].sum().reset_index()
            rows.append({
                "scenario": scenario,
                "layers": column.replace("distance_", "") if column != "distance" else "run",
                "round": "sum",
                "mean_D_benign": float("nan"),
                "mean_D_malicious": float("nan"),
                "p_malicious_further": p_malicious_further(
                    totals[totals["malicious"] == 1][column].to_numpy(),
                    totals[totals["malicious"] == 0][column].to_numpy()),
            })

    result = pd.DataFrame(rows)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.scores[0])), "pid_separability.csv")
    result.to_csv(out, index=False)

    shown = result[result["round"].astype(str).isin(["1", "2", "5", "10", "15", "20", "sum"])]
    summary = shown.pivot_table(index=["scenario", "layers"], columns="round",
                                values="p_malicious_further", aggfunc="first")
    summary = summary[[c for c in ["1", "2", "5", "10", "15", "20", "sum"] if c in summary.columns]
                      or summary.columns]
    summary.columns = [str(c) for c in summary.columns]
    order = [c for c in ["1", "2", "5", "10", "15", "20", "sum"] if c in summary.columns]
    print("P(malicious client further from the centroid than an honest one), per round:")
    print("(1.0 = perfect separation, 0.5 = no difference, < 0.5 = honest clients further away)\n")
    print(summary[order].round(3).to_string())
    print("\nAll values: " + out)


if __name__ == "__main__":
    main()
