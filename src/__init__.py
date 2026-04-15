"""Project package exports."""

__all__ = ["WeightedTokenClassificationTrainer"]


def __getattr__(name):
    if name == "WeightedTokenClassificationTrainer":
        from .hf_trainer import WeightedTokenClassificationTrainer

        return WeightedTokenClassificationTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
