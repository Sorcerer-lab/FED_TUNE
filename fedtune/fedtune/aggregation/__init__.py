from .base import AggContext, AggregationResult, Aggregator, ClientUpdate, normalise, weighted_average
from .diversity import DiversityAware
from .fedavg import FedAvg


class StrategyNotImplemented(RuntimeError):
    pass


REGISTRY = {"fedavg": lambda cfg: FedAvg(), "diversity": lambda cfg: DiversityAware(cfg)}
_PLANNED_STEP = {"clustered": 7, "hybrid": 7, "fedsa": 8}


def get_aggregator(name: str, cfg: dict | None = None) -> Aggregator:
    if name in REGISTRY:
        return REGISTRY[name](cfg)
    if name in _PLANNED_STEP:
        raise StrategyNotImplemented(
            f"strategy '{name}' is planned for Step {_PLANNED_STEP[name]}; implemented now: {sorted(REGISTRY)}")
    raise StrategyNotImplemented(f"unknown strategy '{name}'")
