from .base import AggregationResult, Aggregator, ClientUpdate, normalise, weighted_average
from .fedavg import FedAvg


class StrategyNotImplemented(RuntimeError):
    pass


REGISTRY = {"fedavg": FedAvg}
_PLANNED_STEP = {"diversity": 6, "clustered": 7, "hybrid": 7, "fedsa": 8}


def get_aggregator(name: str) -> Aggregator:
    if name in REGISTRY:
        return REGISTRY[name]()
    if name in _PLANNED_STEP:
        raise StrategyNotImplemented(
            f"strategy '{name}' is planned for Step {_PLANNED_STEP[name]}; implemented now: {sorted(REGISTRY)}")
    raise StrategyNotImplemented(f"unknown strategy '{name}'")
