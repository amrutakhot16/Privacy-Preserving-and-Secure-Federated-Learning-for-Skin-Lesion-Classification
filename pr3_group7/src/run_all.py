"""
One command that reproduces every Part 2 and Part 3 result.

  python src/run_all.py                 # Part 2: conventional FedAvg (7 runs x 20 rounds)
  python src/run_all.py --defense pid   # Part 3: the same 7 runs with the PID defense
  python src/run_all.py --smoke         # 2-round check that everything works
  python src/run_all.py --defense pid --smoke
  python src/run_all.py --rounds 30     # change the number of rounds

With --defense pid every tag gets the prefix "pid_" (pid_clean,
pid_p70_flip100, ...), or "pid_last_" / "pid_classifier_" with
--distance-layers last / classifier, so PID results sit next to the conventional ones
in results/no_attack/ and results/attack/ without overwriting them. The
PID settings (--kp --ki --kd --threshold-mode --threshold --threshold-k
--distance-norm --min-keep --distance-layers) are passed on to run_fl.py.

Steps:
  1. Create the 20 clean client datasets if data/clients/ does not exist.
  2. Create the poisoned client datasets for every attack scenario
     (written to data/poisoned/<scenario>/, loaded directly by the clients).
  3. Run the simulations (each one in its own process):
       clean           no attack                     -> results/no_attack/
       p70_flip25      70% clients, 25% flipped      -> results/attack/
       p70_flip50      70% clients, 50% flipped      -> results/attack/
       p70_flip100     70% clients, 100% flipped     -> results/attack/
       p10_flip100     10% clients, 100% flipped     -> results/attack/
       p30_flip100     30% clients, 100% flipped     -> results/attack/
       p50_flip100     50% clients, 100% flipped     -> results/attack/
     Both results folders have the same layout: plots/, csv/ and json/.
     Every file name starts with the run tag.
     Group 7's assigned setting is 70% poisoned clients with flip rates
     0/25/50/100%; the extra 10/30/50% runs cover the "number of poisoned
     clients" axis of the attack-severity plot.
  4. Draw the attack-severity plots (results/attack/plots/) and table
     (results/attack/csv/severity_metrics.csv).

Finished runs are skipped, so if the computer stops halfway you can just
run the command again. Use --force to redo everything.

@author Amruta Kirtikumar Khot, ak3792@rit.edu
@author Cazal Tirkey, ct9857@rit.edu
"""

import argparse
import os
import subprocess
import sys
import time

SRC_FOLDER = os.path.dirname(os.path.abspath(__file__))
PROJECT_FOLDER = os.path.dirname(SRC_FOLDER)
for _name in ("data_prep", "attack", "federated", "analysis"):
    sys.path.insert(0, os.path.join(SRC_FOLDER, _name))

from poison_clients import create_scenario, scenario_name  # noqa: E402
from plot_results import plot_comparison, plot_severity  # noqa: E402


MAIN_POISON_FRACTION = 0.70           # Group 7 (Table 1)
FLIP_RATES = [0.25, 0.50, 1.00]       # 0% flip == the clean run
EXTRA_POISON_FRACTIONS = [0.10, 0.30, 0.50]   # at 100% flip, for the severity plot


def run(command):
    print("\n>>> " + " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=PROJECT_FOLDER)


def main():
    parser = argparse.ArgumentParser(description="Run every Part 2 experiment.")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--cpus-per-client", type=float, default=1.0)
    parser.add_argument("--gpus-per-client", type=float, default=0.0)
    parser.add_argument("--class-weights", dest="class_weights", action="store_true", default=True)
    parser.add_argument("--no-class-weights", dest="class_weights", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="2 rounds, 2 runs, into results_smoke/")
    parser.add_argument("--force", action="store_true", help="re-run finished experiments")
    # Part 3: PID defense
    parser.add_argument("--defense", choices=["none", "pid"], default="none")
    parser.add_argument("--kp", type=float, default=1.0)
    parser.add_argument("--ki", type=float, default=0.05)
    parser.add_argument("--kd", type=float, default=0.5)
    parser.add_argument("--threshold-mode", choices=["median", "fixed", "std", "none"], default="median")
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--threshold-k", type=float, default=1.5)
    parser.add_argument("--distance-norm", choices=["mean", "params", "none"], default="mean")
    parser.add_argument("--min-keep", type=int, default=2)
    parser.add_argument("--distance-layers", choices=["all", "classifier", "last"], default="all")
    args = parser.parse_args()
    # PID runs on the whole model: pid_<scenario>; on part of the model:
    # pid_last_<scenario> or pid_classifier_<scenario>, so no run overwrites another.
    prefix = ""
    if args.defense == "pid":
        prefix = "pid_" if args.distance_layers == "all" else "pid_" + args.distance_layers + "_"

    clients_dir = os.path.join(PROJECT_FOLDER, "data", "clients")
    poisoned_root = os.path.join(PROJECT_FOLDER, "data", "poisoned")
    results_root = os.path.join(PROJECT_FOLDER, "results_smoke" if args.smoke else "results")
    no_attack_dir = os.path.join(results_root, "no_attack")
    attack_dir = os.path.join(results_root, "attack")

    # 1. Clean client datasets (Part 1 output).
    if not os.path.isfile(os.path.join(clients_dir, "client_19.npz")):
        run([sys.executable, os.path.join(SRC_FOLDER, "data_prep", "partition_clients.py"), "--seed", str(args.seed)])

    # 2. Poisoned datasets on disk.
    scenarios = [(MAIN_POISON_FRACTION, f) for f in FLIP_RATES]
    scenarios += [(p, 1.0) for p in EXTRA_POISON_FRACTIONS]
    if args.smoke:
        scenarios = [(MAIN_POISON_FRACTION, 1.0)]
    for poison_fraction, flip_rate in scenarios:
        create_scenario(clients_dir, poisoned_root, 20, poison_fraction, flip_rate, args.seed)

    rounds = 2 if args.smoke else args.rounds
    epochs = 1 if args.smoke else args.local_epochs
    common = ["--rounds", str(rounds), "--local-epochs", str(epochs), "--seed", str(args.seed),
              "--cpus-per-client", str(args.cpus_per_client), "--gpus-per-client", str(args.gpus_per_client)]
    common.append("--class-weights" if args.class_weights else "--no-class-weights")
    if args.defense == "pid":
        common += ["--defense", "pid", "--kp", str(args.kp), "--ki", str(args.ki), "--kd", str(args.kd),
                   "--threshold-mode", args.threshold_mode, "--threshold", str(args.threshold),
                   "--threshold-k", str(args.threshold_k), "--distance-norm", args.distance_norm,
                   "--min-keep", str(args.min_keep), "--distance-layers", args.distance_layers]

    # 3. Simulations.
    jobs = [(prefix + "clean", no_attack_dir, None)]
    for poison_fraction, flip_rate in scenarios:
        scenario = scenario_name(poison_fraction, flip_rate)
        jobs.append((prefix + scenario, attack_dir, os.path.join(poisoned_root, scenario)))

    start = time.time()
    for tag, out_dir, poisoned_dir in jobs:
        finished_marker = os.path.join(out_dir, "plots", tag + "_4_avg_accuracy.png")
        if not args.force and os.path.isfile(finished_marker):
            print("Skipping " + tag + " (already finished; use --force to redo)")
            continue
        command = [sys.executable, os.path.join(SRC_FOLDER, "federated", "run_fl.py"), "--tag", tag, "--out", out_dir]
        if poisoned_dir:
            command += ["--poisoned-dir", poisoned_dir]
        run(command + common)

    # 4. Attack-severity plots.
    plot_severity(no_attack_dir, attack_dir, prefix=prefix)

    # 5. Part 3: conventional vs PID on one plot, wherever both runs exist.
    if args.defense == "pid":
        for tag, out_dir, _poisoned in jobs:
            plot_comparison(out_dir, out_dir, tag[len(prefix):], pid_prefix=prefix)
    print("\nAll done in " + format((time.time() - start) / 60, ".1f") + " min. Results in " + results_root)


if __name__ == "__main__":
    main()
