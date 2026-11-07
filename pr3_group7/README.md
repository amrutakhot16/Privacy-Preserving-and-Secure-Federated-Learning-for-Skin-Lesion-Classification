# Privacy Preserving and Secure Federated Learning for Skin Lesion Classification (Group 7)

Amruta Kirtikumar Khot (ak3792@rit.edu), Cazal Tirkey (ct9857@rit.edu)

A federated learning system built with **Flower + PyTorch** with 20 simulated
clients and a **label-flipping data-poisoning attack** on **70% of the clients
(14 of 20)** with flipping rates of **0%, 25%, 50% and 100%** (Group 7 in
Table 1 of the project specification).

* **Part 2 - conventional FL:** plain FedAvg, every client is aggregated.
* **Part 3 - FL with PID:** the server gives every client a PID score based on
  the distance of its model to the centroid of all client models, and
  permanently excludes clients whose score is above the threshold
  (see [Section 8](#8-part-3-fl-with-pid)).

This README is the user's guide for the code in `src/`. All commands below are
run from the project folder (the folder that contains `README.md`, `requirements.txt`, `src/` and `results/`).
```
README.md                  <- user's guide
requirements.txt           <- Python dependencies
src/
  run_all.py               <- one command that reproduces every result
  data_prep/               <- cleaning, resizing, normalization, partitioning, augmentation
  attack/                  <- writes label-flipped client datasets to disk
  federated/               <- DermCNN model, Flower client, FedAvg and PID strategies, run_fl.py
  analysis/                <- metric logging, plotting, PID tuning (pid_replay.py)
results/
  validation/{plots,csv}       <- PID check on synthetic updates (Part 3)
  no_attack/{plots,csv,json}   <- clean runs: conventional ("clean_*") and PID ("pid_clean_*")
  attack/{plots,csv,json}      <- attack runs: conventional ("p70_flip25_*", ...) and PID ("pid_p70_flip25_*", ...)
```

---

## 1. Setup

The exact library versions are listed in `requirements.txt`.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Data

The `data/` folder and the virtual environment are **not** included in the
submission archive (they are large and can be fully regenerated). Download
HAM10000 (Harvard Dataverse, https://doi.org/10.7910/DVN/DBW86T, or Kaggle
"Skin Cancer MNIST: HAM10000") and place it as:

```
data/raw/HAM10000_metadata.csv
data/raw/HAM10000_images_part_1/*.jpg
data/raw/HAM10000_images_part_2/*.jpg
```

The processed outputs must exist in these locations before `run_all.py` is used.

```
data/processed/pool.npz          federated training pool 
data/processed/test.npz          held-out clean test set 
data/processed/norm_stats.json   per-channel mean/std from the pool
data/clients/client_00..19.npz   20 non-IID clients (created automatically if missing)
```

To rebuild them from the raw HAM10000 download, run:

```bash
python src/data_prep/prepare_ham10000.py      # reads data/raw/ (see above)
python src/data_prep/partition_clients.py     # 20 clients, Dirichlet alpha = 0.5, seed 42
```

## 3. Run everything (one command)

First, run a quick 2-round check (a few minutes) to confirm the setup works:

```bash
python src/run_all.py --smoke
```

Then run the full experiment set:

```bash
python src/run_all.py                    # 7 runs x 20 rounds, 3 local epochs, class-weighted loss
python src/run_all.py --gpus-per-client 0.1   
```

`run_all.py` does the following:

1. Creates the poisoned client datasets **on disk** (`data/poisoned/<scenario>/`).
   Poisoning never happens at runtime.
2. Runs one simulation per scenario, each in its own process:

| Tag | Attack | Output folder |
|---|---|---|
| `clean` | none | `results/no_attack/` |
| `p70_flip25` | 14 clients, 25% of labels flipped | `results/attack/` |
| `p70_flip50` | 14 clients, 50% of labels flipped | `results/attack/` |
| `p70_flip100` | 14 clients, 100% of labels flipped | `results/attack/` |
| `p10_flip100` | 2 clients, 100% flipped | `results/attack/` |
| `p30_flip100` | 6 clients, 100% flipped | `results/attack/` |
| `p50_flip100` | 10 clients, 100% flipped | `results/attack/` |

   The last 3 runs give the number of poisoned clients
   axis of the attack-severity plot, while the `p70_*` runs give the
   "flipping rate" axis. The clean run is the 0% point on both axes.
3. Draws the attack-severity plots.

Finished runs are skipped, so if the machine stops halfway, run the same
command again. Use `--force` to redo everything.

To run a single experiment by hand:

```bash
python src/federated/run_fl.py --tag clean --out results/no_attack
python src/attack/poison_clients.py --poison-frac 0.7 --flip-rate 0.5
python src/federated/run_fl.py --tag p70_flip50 --out results/attack --poisoned-dir data/poisoned/p70_flip50
python src/analysis/plot_results.py --severity       # redraw severity plots from CSVs
```

## 4. Output files

`results/no_attack/` and `results/attack/` have the same layout:

```
results/
├── no_attack/          (the clean run, tag "clean")
│   ├── plots/          all .png plots
│   ├── csv/            all .csv metric files
│   └── json/           the run configuration (.json)
└── attack/             (all six attack runs)
    ├── plots/          all .png plots, incl. the 5a/5b severity plots
    ├── csv/            all .csv metric files, incl. severity_metrics.csv
    └── json/           one configuration (.json) per run
```

Every file name starts with the run tag. For example, `attack/csv/p70_flip50_avg_loss.csv`
is the average loss with 70% of clients poisoned and 50% of their labels flipped.

| Required artifact (spec) | Plot (in `plots/`) | CSV (in `csv/`) |
|---|---|---|
| Loss per client over rounds | `<tag>_1_loss_per_client.png` | `<tag>_loss_per_client.csv` |
| Average loss among clients | `<tag>_2_avg_loss.png` | `<tag>_avg_loss.csv` |
| Evaluation accuracy per client | `<tag>_3_accuracy_per_client.png` | `<tag>_accuracy_per_client.csv` |
| Average evaluation accuracy | `<tag>_4_avg_accuracy.png` | `<tag>_avg_accuracy.csv` |
| Loss & accuracy vs attack severity | `5a_severity_final_loss.png`, `5b_severity_final_accuracy.png` | `severity_metrics.csv` |

Additional files for each run:

* `csv/<tag>_per_client_eval_long.csv` – the same per-client metrics in long format (round, client, malicious, loss, accuracy, n).
* `csv/<tag>_aggregation_log.csv` – the number and IDs of the clients aggregated and **excluded** each round. Conventional FedAvg excludes nobody.
* `csv/<tag>_train_per_client_long.csv` – local training loss and accuracy reported by each client.
* `csv/<tag>_global_test.csv` – the global model on the clean held-out test set (accuracy, balanced accuracy, macro-F1); plotted in `plots/<tag>_global_test.png`.
* `json/<tag>_config.json` – every setting of the run, the malicious client IDs, and the library versions.

How the columns are defined:

* In the per-client tables, the columns are named `client_XX_benign` or `client_XX_malicious`.
* In the average tables, `mean_all_clients` is the simple mean over the 20 clients (the "average among clients"). `weighted_mean_all` is weighted by local validation size, which is what Flower reports. `mean_benign_clients` and `mean_malicious_clients` split the average by group.

## 5. What the code does

```
src/
├── run_all.py            reproduces every result with one command
├── data_prep/            data preparation 
├── attack/               label-flipping data poisoning
├── federated/            model and Flower FL system
└── analysis/             metric logging and plots
```

| File | Purpose |
|---|---|
| `run_all.py` | Reproduces every result with one command |
| `data_prep/prepare_ham10000.py` | cleaning, lesion-safe pool/test split, 64×64 resize, normalization stats |
| `data_prep/partition_clients.py` | non-IID Dirichlet split into 20 clients, each with a local train/val split |
| `data_prep/augment.py` | Training-time augmentation and normalization |
| `attack/poison_clients.py` | Writes label-flipped copies of the malicious clients' data to disk |
| `federated/model.py` | DermCNN (3 conv blocks, GroupNorm, 304,871 parameters) |
| `federated/fl_client.py` | Flower `NumPyClient`: local training (`fit`) and local evaluation (`evaluate`) |
| `federated/fl_strategy.py` | `LoggingFedAvg`: standard FedAvg plus per-client metric logging |
| `federated/pid_strategy.py` | `PIDFedAvg` (Part 3): PID scores, threshold, permanent client exclusion |
| `federated/run_fl.py` | Runs one Flower simulation and saves its CSVs and plots |
| `analysis/metrics_logger.py` | Collects the metrics and writes the CSV files |
| `analysis/plot_results.py` | All plots, including attack severity, PID scores, client removal and conventional-vs-PID comparison |
| `analysis/pid_replay.py` | Fast tuning of Kp, Ki, Kd, the threshold and the layer choice from logged distances (Part 3) |
| `analysis/pid_separability.py` | How well each distance separates malicious from honest clients, per round (Part 3) |
| `tests/validate_pid_synthetic.py` | Checks the PID code on synthetic updates where the correct answer is known (Part 3) |

### FL setup

* **Initialization:** the server builds DermCNN and sends the same weights to all clients.
* **Each round:** all 20 clients train locally (`fraction_fit = 1.0`) and send their weights back. The server aggregates them with **FedAvg**, weighted by the number of local training images.
* **Evaluation:** after every aggregation, all 20 clients evaluate the new global model on their own local validation subset (`fraction_evaluate = 1.0`). The server also evaluates it on the clean test set.
* **Class-weighted loss (on by default):** each client weights its cross-entropy loss by inverse class frequency, computed from its own labels only and capped at 10. `--no-class-weights` turns it off.
* **Local training settings:** Adam, learning rate 1e-3, weight decay 1e-4, batch size 32, **3 local epochs, 20 rounds**, with flips, ±20° rotation, zoom/shift and mild colour jitter. 
* **Reproducibility:** all randomness is seeded, using seed + 1000·round + client ID per client per round.

### Why these settings

Our first clean run (1 local epoch, plain cross-entropy, 30 rounds) never got past predicting the majority class. Test accuracy stayed at the nv share of the test set and balanced accuracy stayed at chance for all 30 rounds. With only about 10 mini-batches per client per epoch, one epoch per round is too little training. Plain cross-entropy also rewards always answering "nv", which is 67% of the data. Three local epochs and a class-weighted loss fixed this. In the clean run, balanced accuracy on the test set rose from 0.14 to 0.32, and average client loss fell from 1.75 to 1.23. Plain accuracy first drops (the model stops predicting only nv) and then rises again.

### Attack

* **Malicious clients:** chosen from one fixed random ordering (seed 42), so smaller attack sets are subsets of larger ones. The malicious IDs are stored in `data/poisoned/<scenario>/scenario.json`.
* **Flipping rule:** a flipped label `y` becomes `(y + 1) mod 7`, so every flipped label is wrong.
* **What is flipped:** both the local **training** and **validation** labels of a malicious client. The spec requires each client to be evaluated on its own poisoned subset, which is what makes malicious clients visibly different in the per-client plots. The clean global test set is never modified.

### Changes since Part 1

1. `prepare_ham10000.py` computed the normalization mean in float32, which gave the same value (0.480) for all three channels. It now uses float64. The correct values are mean = [0.764, 0.538, 0.562] and std = [0.137, 0.158, 0.176].
2. Label corruption was planned for Part 3 in the Part 1 report. Part 2 already requires attacked runs, so it is implemented here, and validation labels of malicious clients are flipped as well.

## 6. Run time

There are 7 runs of 20 rounds each. Each round trains on about 6,800 images
across all clients (3 local epochs). On our MacBook Air (Apple silicon, 10 CPU
cores) one round took 1 to 2 minutes and one run took 22 to 40 minutes; the
full set of 7 runs took about 3 hours. The `round_seconds` column of `csv/<tag>_aggregation_log.csv`
shows the time per round. Use the `--smoke` run to estimate the total time
before starting the full set. On a GPU, use `--gpus-per-client 0.1`.

## 7. Troubleshooting

* **`ModuleNotFoundError` inside Ray workers:** run commands from the project root, as shown above. `run_fl.py` adds the four `src/` subfolders to `PYTHONPATH` for the workers.
* **Out of memory / Ray workers killed:** use `--cpus-per-client 2` so fewer clients run in parallel.
* **Ray fails to start (e.g. on Windows):** create a fresh virtual environment with Python 3.11, or run under WSL.
* **A client failed:** the run stops with an error rather than silently averaging over fewer clients.

## 8. Part 3: FL with PID

### How the defense works

`PIDFedAvg` (`src/federated/pid_strategy.py`) is `LoggingFedAvg` with one extra
step before aggregation. Client training, evaluation, logging and plotting are
the same code as in Part 2, so the two methods are directly comparable.

In every round the server:

1. flattens every received client model into one vector and computes the
   centroid (the mean of all received models);
2. computes each client's L2 distance to the centroid and normalizes it
   (`--distance-norm mean`: divided by the average distance of that round,
   so a typical client has D ≈ 1);
3. computes the PID score

   `u(t) = Kp·D(t) + Ki·Σ_{i<t} D(i) + Kd·(D(t) − D(t−1))`

   using only `Kp·D(t)` in a client's first round (no history yet);
4. excludes every client whose score is **above** the threshold. The spec's
   pseudocode says "PID < threshold", but its step-by-step text says clients
   with a PID greater than the threshold are discarded; a larger distance means
   a more unusual model, so we use ">";
5. runs normal FedAvg (weighted by local training size) over the remaining clients.

**Which weights are compared** (`--distance-layers`): `all` (whole model,
default), `classifier` (the two fully connected layers) or `last` (only the
output layer). Label flipping changes which class an image is mapped to, and
that mapping lives mostly in the last layers; the convolutional layers learn
general image features and their many parameters can hide the difference
(this follows Tolpegin et al., 2020). Every PID score file logs all three
distances (`distance_all`, `distance_classifier`, `distance_last`), so one
observe-only run is enough to compare them. PID runs with `classifier` or
`last` are saved as `pid_classifier_<run>` / `pid_last_<run>` so they never
overwrite the whole-model results.

**Exclusion is permanent.** An excluded client no longer receives the global
model for training or evaluation, so it disappears from the per-client plots
from that round on. A safeguard (`--min-keep 2`) keeps the federation from
excluding everyone.

**Threshold modes** (`--threshold-mode`):

| Mode | Rule | Notes |
|---|---|---|
| `median` (default) | exclude if `u > k · median(u)` (`--threshold-k`, default 1.5) | Stays on the same scale every round although the integral term keeps growing; excludes nobody when all clients behave alike |
| `fixed` | exclude if `u > --threshold` | The integral grows every round, so a fixed value eventually catches honest clients too |
| `std` | exclude if `u > mean(u) + k · std(u)` | Tends to remove some client in almost every round |
| `none` | never exclude | Observe-only, used for tuning |

### Running Part 3

```bash
python src/run_all.py --defense pid --smoke     # 2-round check, writes to results_smoke/
python src/run_all.py --defense pid             # 7 PID runs x 20 rounds (same scenarios as Part 2)
python src/run_all.py --defense pid --kp 1 --ki 0.05 --kd 0.5 --threshold-mode median --threshold-k 1.75
```

Every PID run uses the tag of the matching Part 2 run with the prefix `pid_`
(`pid_clean`, `pid_p70_flip100`, ...), the same seed and the same settings, so
the conventional and PID results sit side by side without overwriting each other.

A single run by hand:

```bash
python src/federated/run_fl.py --tag pid_p70_flip100 --out results/attack \
    --poisoned-dir data/poisoned/p70_flip100 --defense pid --kp 1 --ki 0.05 --kd 0.5
```

### Tuning Kp, Ki, Kd and the threshold

The spec asks for the coefficients to be investigated empirically. A full run
takes 20 to 40 minutes, so tuning is done in three steps:

```bash
# 1. one observe-only run per scenario (logs distances, excludes nobody)
python src/federated/run_fl.py --tag pid_observe_p70_flip50 --out results/tuning \
    --poisoned-dir data/poisoned/p70_flip50 --defense pid --threshold-mode none

# 2. replay many settings on the logged distances (seconds)
python src/analysis/pid_replay.py results/tuning/csv/pid_observe_*_pid_scores.csv

# 2b. which weights separate malicious clients best (whole model, classifier, output layer)
python src/analysis/pid_separability.py results/tuning/csv/pid_observe_*_pid_scores.csv

# 3. confirm the chosen settings with real runs
python src/run_all.py --defense pid --kp ... --ki ... --kd ... --threshold-k ...
```

The replay is exact up to the first exclusion. After that it is approximate,
because removing a client would change the centroid in later rounds; step 3
gives the real result.

### Part 3 output files

In addition to the four required plots and CSVs (same names as Part 2 with the
`pid_` prefix, e.g. `pid_p70_flip100_avg_loss.csv`), each PID run writes:

| File | Content |
|---|---|
| `csv/pid_<run>_pid_scores.csv` | one row per (round, client): raw and normalized distance, P, I and D terms, score, threshold, excluded |
| `csv/pid_<run>_detection_summary.csv` | per client: malicious or not, excluded or not, round of exclusion, outcome (true/false positive/negative) |
| `csv/pid_<run>_aggregation_log.csv` | clients aggregated and excluded in each round |
| `plots/pid_<run>_6_pid_scores.png` | PID score of every client per round with the threshold; X = round of exclusion |
| `plots/pid_<run>_7_client_removal.png` | **client removal dynamics**: clients × rounds map (aggregated / correctly excluded / wrongly excluded) and the number of benign and malicious clients still aggregated |
| `csv/pid_severity_metrics.csv`, `plots/pid_5a_severity_final_loss.png`, `plots/pid_5b_severity_final_accuracy.png` | final loss and accuracy vs attack severity, plus detection counts |
| `plots/compare_<run>_avg_loss.png`, `plots/compare_<run>_avg_accuracy.png` | conventional FedAvg vs FL with PID on one plot |
| `json/pid_<run>_config.json` | all settings, including Kp, Ki, Kd, threshold and the detection counts |

Plots can be redrawn from the CSVs:

```bash
python src/analysis/plot_results.py --out results/attack --tag pid_p70_flip100 --pid
python src/analysis/plot_results.py --severity --prefix pid_
python src/analysis/plot_results.py --compare clean p70_flip100
```

### Experiments reported in the Part 3 report

All commands are run from the project folder. The main results use the
spec's coefficients with a threshold chosen by tuning:

```bash
# Main Part 3 results (7 runs)
python src/run_all.py --defense pid --kp 1 --ki 0.05 --kd 0.5 --threshold-mode median --threshold-k 1.75

# Tuning: observe-only runs + replay (results/tuning/, not part of the submission archive)
python src/federated/run_fl.py --tag pid_observe_clean --out results/tuning --defense pid --threshold-mode none
for s in p70_flip25 p70_flip50 p70_flip100 p30_flip100; do
  python src/federated/run_fl.py --tag pid_observe_$s --out results/tuning \
      --poisoned-dir data/poisoned/$s --defense pid --threshold-mode none
done
python src/analysis/pid_replay.py results/tuning/csv/pid_observe_*_pid_scores.csv

# Stricter threshold (k = 1.3)
for s in p70_flip100 p30_flip100; do
  python src/federated/run_fl.py --tag pid_k1.3_$s --out results/attack --poisoned-dir data/poisoned/$s \
      --defense pid --kp 1 --ki 0.05 --kd 0.5 --threshold-mode median --threshold-k 1.3
done

# IID control: same attack (30% clients, 100% flipped) on an IID client split
python src/data_prep/partition_clients.py --strategy iid --out data/clients_iid
python src/attack/poison_clients.py --clients-dir data/clients_iid --out-root data/poisoned_iid --poison-frac 0.3 --flip-rate 1.0
python src/federated/run_fl.py --tag pid_iid_p30_flip100 --out results/attack --clients-dir data/clients_iid \
    --poisoned-dir data/poisoned_iid/p30_flip100 --defense pid --kp 1 --ki 0.05 --kd 0.5 --threshold-mode median --threshold-k 1.75
python src/federated/run_fl.py --tag pid_iid_tuned_p30_flip100 --out results/attack --clients-dir data/clients_iid \
    --poisoned-dir data/poisoned_iid/p30_flip100 --defense pid --kp 1 --ki 0.02 --kd 0.5 --threshold-mode median --threshold-k 1.15
```

### Checking the implementation on synthetic updates

```bash
python src/tests/validate_pid_synthetic.py      # a few seconds, no dataset needed
```

This feeds the real `PIDFedAvg` selection code with synthetic client updates
in which malicious clients clearly differ from honest ones, for 0, 10, 30, 50
and 70% malicious clients. With the same settings as the main runs it removes
nobody at 0%, all malicious clients at 10% and 30%, nobody at 50% (both groups
are equally far from the centroid) and the six honest clients at 70% (the
attackers form the centroid). Results: `results/validation/csv/pid_synthetic_validation.csv`
and the `synthetic_p*_6_pid_scores.png` / `synthetic_p*_7_client_removal.png` plots.
