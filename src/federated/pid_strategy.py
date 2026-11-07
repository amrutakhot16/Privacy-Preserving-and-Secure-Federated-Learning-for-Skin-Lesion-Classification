"""
Enhanced FL with a PID-based client filter.

PIDFedAvg extends LoggingFedAvg. Training, client code, logging,
CSV output and plotting are unchanged, so conventional FL and FL with PID
are directly comparable. Only two things are added:

  1. select_clients_for_aggregation() computes a PID score for every
     client from the weights it sent and excludes clients whose score is
     above the threshold. The remaining clients are aggregated with the
     normal FedAvg rule (weighted by local training size).
  2. configure_fit() / configure_evaluate() stop sending the global model
     to excluded clients. As required by the spec, an excluded client is
     removed from aggregation, from global model distribution and from all
     further communication (exclusion is permanent).

PID score (project spec, Part 3):

    u_c(t) = Kp * D_c(t)
           + Ki * sum_{i < t} D_c(i)              (integral: client history)
           + Kd * (D_c(t) - D_c(t-1))              (derivative: change)

where D_c(t) is the normalized L2 distance between client c's model and
the centroid (plain mean) of all models received in round t. In the first
round a client takes part in, only Kp * D is used (no history yet).
With Kp, Ki, Kd = 1, 0, 0 the score is just the distance.

spec: Pseudocode says "if PID < threshold then exclude",
but the step-by-step text says clients whose PID is GREATER than the
threshold are discarded. A larger distance means a more unusual model,
so we exclude clients with score > threshold.

Which weights are compared (--distance-layers):
  all         every parameter of the model (default, as in the spec)
  classifier  only the two fully connected layers at the end of DermCNN
  last        only the final output layer (7 x 128 weights + 7 biases)
  Label flipping changes which class each image is mapped to, and that
  mapping is mostly stored in the last layer(s). The convolutional layers
  learn general image features that look similar for honest and malicious
  clients, and their many parameters can hide the difference. Comparing
  only the last layer(s) follows the observation of Tolpegin et al. (2020)
  that label-flipping clients separate best in the output-layer weights.
  In every round all three distances are also written to the PID score
  CSV (distance_all, distance_classifier, distance_last), so one
  observe-only run is enough to compare them with analysis/pid_replay.py.

Distance normalization (--distance-norm):
  mean    D = ||w_c - mu|| / mean_k ||w_k - mu||   (default; scale-free,
          D is about 1 for a typical client in every round)
  params  D = ||w_c - mu|| / sqrt(number of parameters)
  none    D = ||w_c - mu||

Threshold (--threshold-mode):
  median  exclude if u > k * median(u) over this round's clients (default).
          The score of a typical client grows every round because the
          integral term adds up past distances, so a fixed number would
          sooner or later exclude everyone. Comparing with the median keeps
          the threshold on the same scale in every round, and unlike
          mean + k*std it does not have to exclude anyone when all clients
          behave alike. It relies on honest clients being the majority.
  fixed   exclude if u > threshold
  std     exclude if u > mean(u) + k * std(u) over this round's clients
  none    never exclude (observe-only: logs PID scores for tuning, see
          analysis/pid_replay.py)

@author Amruta Kirtikumar Khot, ak3792@rit.edu
@author Cazal Tirkey, ct9857@rit.edu
"""

import numpy as np
from flwr.common import parameters_to_ndarrays

from fl_strategy import LoggingFedAvg


# Which arrays of the model (in state_dict order) each option uses. DermCNN
# ends with classifier.2 (Linear 128->128) and classifier.5 (Linear 128->7),
# each with a weight and a bias array, so the last 4 arrays are the
# classifier and the last 2 arrays are the output layer.
LAYER_START = {"all": 0, "classifier": -4, "last": -2}


class PIDFedAvg(LoggingFedAvg):
    """
    FedAvg with PID-based detection and permanent exclusion of clients.
    """

    def __init__(self, logger, kp=1.0, ki=0.05, kd=0.5, threshold_mode="median",
                 threshold=2.0, threshold_k=1.5, distance_norm="mean", min_keep=2,
                 distance_layers="all", **kwargs):
        """
        :param logger: MetricsLogger (records PID scores as well)
        :param kp, ki, kd: PID coefficients
        :param threshold_mode: "median", "fixed", "std" or "none"
        :param threshold: fixed threshold (used when threshold_mode == "fixed")
        :param threshold_k: k in k*median or mean + k*std
        :param distance_norm: "mean", "params" or "none"
        :param min_keep: safeguard, never aggregate fewer than this many clients
        :param distance_layers: "all", "classifier" or "last" (which weights are compared)
        """
        super().__init__(logger, **kwargs)
        if threshold_mode not in ("median", "fixed", "std", "none"):
            raise ValueError("threshold_mode must be median, fixed, std or none")
        if distance_norm not in ("mean", "params", "none"):
            raise ValueError("distance_norm must be mean, params or none")
        if distance_layers not in LAYER_START:
            raise ValueError("distance_layers must be all, classifier or last")
        self.distance_layers = distance_layers
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.threshold_mode = threshold_mode
        self.threshold = float(threshold)
        self.threshold_k = float(threshold_k)
        self.distance_norm = distance_norm
        self.min_keep = int(min_keep)

        # client id -> list of past normalized distances D(0..t-1)
        self.distance_history = {}
        # Flower proxy cid of every excluded client, so the server stops
        # communicating with it (Flower's proxy cid is not our client id).
        self.excluded_proxy_cids = set()

    @staticmethod
    def _flatten(arrays, layers="all"):
        """
        Model parameters of one client as one float64 vector.
        :param arrays: list of numpy arrays in state_dict order
        :param layers: "all", "classifier" or "last"
        """
        selected = arrays[LAYER_START[layers]:]
        return np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in selected])

    def normalized_distances(self, vectors):
        """
        L2 distance of every client model to the centroid (mean) of all
        client models received this round, normalized.
        :param vectors: 2-D array, one row per client
        :return: (normalized distances, raw distances)
        """
        centroid = vectors.mean(axis=0)
        raw = np.linalg.norm(vectors - centroid, axis=1)
        if self.distance_norm == "mean":
            scale = raw.mean() if raw.mean() > 0 else 1.0
        elif self.distance_norm == "params":
            scale = np.sqrt(vectors.shape[1])
        else:
            scale = 1.0
        return raw / scale, raw

    def pid_score(self, client_id, distance):
        """
        PID score of one client for the current round.
        Uses the history BEFORE the current round, as in the spec formula.
        :return: (score, proportional part, integral part, derivative part)
        """
        history = self.distance_history.get(client_id, [])
        proportional = self.kp * distance
        if not history:
            # First round for this client: distance only (spec, step 2).
            return proportional, proportional, 0.0, 0.0
        integral = self.ki * float(np.sum(history))
        derivative = self.kd * (distance - history[-1])
        return proportional + integral + derivative, proportional, integral, derivative

    def round_threshold(self, scores):
        """
        Threshold used in this round.
        """
        if self.threshold_mode == "median":
            return float(self.threshold_k * np.median(scores))
        if self.threshold_mode == "fixed":
            return self.threshold
        if self.threshold_mode == "std":
            return float(np.mean(scores) + self.threshold_k * np.std(scores))
        return float("inf")

   
    def select_clients_for_aggregation(self, server_round, results):
        """
        Compute every client's PID score and exclude the ones above the
        threshold. Called by LoggingFedAvg.aggregate_fit before FedAvg.
        :return: (kept results, list of client ids excluded in this round)
        """
        client_ids = [int(fit_res.metrics["client_id"]) for _proxy, fit_res in results]
        arrays = [parameters_to_ndarrays(fit_res.parameters) for _proxy, fit_res in results]

        # Normalized distance for every layer choice; the chosen one drives the decision,
        # all of them are logged so they can be compared later with pid_replay.py.
        all_distances = {}
        for layers in LAYER_START:
            vectors = np.stack([self._flatten(a, layers) for a in arrays])
            all_distances[layers] = self.normalized_distances(vectors)
        distances, raw = all_distances[self.distance_layers]

        parts = [self.pid_score(c, d) for c, d in zip(client_ids, distances)]
        scores = np.array([p[0] for p in parts])
        threshold = self.round_threshold(scores)
        exclude = scores > threshold

        # never drop so many clients that fewer than min_keep remain.
        if (~exclude).sum() < self.min_keep:
            keep_order = np.argsort(scores)[: self.min_keep]
            exclude = np.ones(len(scores), dtype=bool)
            exclude[keep_order] = False

        kept, excluded_ids = [], []
        for index, (proxy, fit_res) in enumerate(results):
            client_id = client_ids[index]
            self.distance_history.setdefault(client_id, []).append(float(distances[index]))
            proportional, integral, derivative = parts[index][1:]
            self.logger.add_pid(
                server_round, client_id, float(raw[index]), float(distances[index]),
                proportional, integral, derivative, float(scores[index]), threshold,
                bool(exclude[index]),
                extra={"distance_" + layers: float(all_distances[layers][0][index]) for layers in LAYER_START},
            )
            if exclude[index]:
                excluded_ids.append(client_id)
                self.excluded_proxy_cids.add(proxy.cid)
            else:
                kept.append((proxy, fit_res))

        print("[round " + str(server_round) + "] PID threshold " + format(threshold, ".3f")
              + ", excluded now: " + str(sorted(excluded_ids)))
        return kept, excluded_ids

    def _drop_excluded(self, instructions):
        return [(proxy, ins) for proxy, ins in instructions if proxy.cid not in self.excluded_proxy_cids]

    def configure_fit(self, server_round, parameters, client_manager):
        """
        Excluded clients no longer receive the global model for training.
        """
        return self._drop_excluded(super().configure_fit(server_round, parameters, client_manager))

    def configure_evaluate(self, server_round, parameters, client_manager):
        """
        Excluded clients no longer receive the global model for evaluation.
        """
        return self._drop_excluded(super().configure_evaluate(server_round, parameters, client_manager))
