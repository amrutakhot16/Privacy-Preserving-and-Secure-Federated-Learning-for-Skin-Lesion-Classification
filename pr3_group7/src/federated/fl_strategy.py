"""
Conventional FedAvg strategy with metric logging.

LoggingFedAvg is Flower's standard FedAvg (weighted by the number of local
training examples) plus logging hooks. The aggregation itself is NOT
changed. Every client update is accepted, which is exactly why
conventional FL is vulnerable to the label-flipping clients.


@author Amruta Kirtikumar Khot, ak3792@rit.edu
@author Cazal Tirkey, ct9857@rit.edu
"""

import time
from flwr.server.strategy import FedAvg


def weighted_average(metrics, key):
    """
    Weighted mean of one metric over clients.
    :param metrics: list of (num_examples, metrics_dict) tuples from Flower
    :param key: metric name
    :return: float
    """
    total = sum(n for n, m in metrics if key in m)
    if total == 0:
        return 0.0
    return sum(n * float(m[key]) for n, m in metrics if key in m) / total


def fit_metrics_aggregation(metrics):
    return {
        "train_loss": weighted_average(metrics, "train_loss"),
        "train_accuracy": weighted_average(metrics, "train_accuracy"),
    }


def evaluate_metrics_aggregation(metrics):
    return {"accuracy": weighted_average(metrics, "accuracy")}


class LoggingFedAvg(FedAvg):
    """
    FedAvg that records per-client metrics after every aggregation round.
    """

    def __init__(self, logger, **kwargs):
        kwargs.setdefault("fit_metrics_aggregation_fn", fit_metrics_aggregation)
        kwargs.setdefault("evaluate_metrics_aggregation_fn", evaluate_metrics_aggregation)
        super().__init__(**kwargs)
        self.logger = logger
        self.excluded_clients = set()
        self._round_start = None

    def configure_fit(self, server_round, parameters, client_manager):
        self._round_start = time.time()
        return super().configure_fit(server_round, parameters, client_manager)

    def select_clients_for_aggregation(self, server_round, results):
        """
        Decide which client updates are aggregated.
        Conventional FL: keep everyone, exclude no one.
        :return: (kept results, list of excluded client ids)
        """
        return results, []

    def aggregate_fit(self, server_round, results, failures):
        """
        Log every client's training metrics, then run normal FedAvg.
        """
        if failures:
            # A crashed client would silently disappear from the averages,
            # so stop loudly instead of producing misleading results.
            raise RuntimeError(
                "Round " + str(server_round) + ": " + str(len(failures))
                + " client(s) failed during fit: " + repr(failures[:3])
            )

        for _proxy, fit_res in results:
            m = fit_res.metrics
            self.logger.add_train(
                server_round,
                int(m["client_id"]),
                m.get("train_loss", float("nan")),
                m.get("train_accuracy", float("nan")),
                fit_res.num_examples,
            )

        kept, excluded = self.select_clients_for_aggregation(server_round, results)
        self.excluded_clients.update(excluded)

        aggregated_ids = [int(fit_res.metrics["client_id"]) for _p, fit_res in kept]
        seconds = None if self._round_start is None else time.time() - self._round_start
        self.logger.add_aggregation(server_round, aggregated_ids, excluded, len(failures), seconds)

        print(
            "[round " + str(server_round) + "] aggregated " + str(len(aggregated_ids))
            + " clients, excluded " + str(len(excluded)) + ", failures " + str(len(failures))
        )
        return super().aggregate_fit(server_round, kept, failures)

    def aggregate_evaluate(self, server_round, results, failures):
        """
        Log every client's evaluation loss and accuracy (global model on
        the client's local validation subset), then let FedAvg compute
        its weighted average.
        """
        for _proxy, eval_res in results:
            m = eval_res.metrics
            self.logger.add_eval(
                server_round,
                int(m["client_id"]),
                eval_res.loss,
                m.get("accuracy", float("nan")),
                eval_res.num_examples,
            )
        return super().aggregate_evaluate(server_round, results, failures)
