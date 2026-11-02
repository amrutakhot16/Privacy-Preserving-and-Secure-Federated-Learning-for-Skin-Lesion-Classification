"""
One command that reproduces every result.

  python src/run_all.py                 # full experiment set (7 runs x 20 rounds)
  python src/run_all.py --smoke         # 2-round check that everything works
  python src/run_all.py --rounds 30     # change the number of rounds

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
from plot_results import plot_severity  # noqa: E402


MAIN_POISON_FRACTION = 0.70           # Group 7 (Table 1)
FLIP_RATES = [0.25, 0.50, 1.00]       # 0% flip == the clean run
EXTRA_POISON_FRACTIONS = [0.10, 0.30, 0.50]   # at 100% flip, for the severity plot


def run(command):
    print("\n>>> " + " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=PROJECT_FOLDER)


def main():
    parser = argparse.ArgumentParser(description="Run every experiment.")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--cpus-per-client", type=float, default=1.0)
    parser.add_argument("--gpus-per-client", type=float, default=0.0)
    parser.add_argument("--class-weights", dest="class_weights", action="store_true", default=True)
    parser.add_argument("--no-class-weights", dest="class_weights", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="2 rounds, 2 runs, into results_smoke/")
    parser.add_argument("--force", action="store_true", help="re-run finished experiments")
    args = parser.parse_args()

    clients_dir = os.path.join(PROJECT_FOLDER, "data", "clients")
    poisoned_root = os.path.join(PROJECT_FOLDER, "data", "poisoned")
    results_root = os.path.join(PROJECT_FOLDER, "results_smoke" if args.smoke else "results")
    no_attack_dir = os.path.join(results_root, "no_attack")
    attack_dir = os.path.join(results_root, "attack")

    # 1. Clean client datasets 
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

    # 3. Simulations.
    jobs = [("clean", no_attack_dir, None)]
    for poison_fraction, flip_rate in scenarios:
        tag = scenario_name(poison_fraction, flip_rate)
        jobs.append((tag, attack_dir, os.path.join(poisoned_root, tag)))

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
    plot_severity(no_attack_dir, attack_dir)
    print("\nAll done in " + format((time.time() - start) / 60, ".1f") + " min. Results in " + results_root)


if __name__ == "__main__":
    main()
