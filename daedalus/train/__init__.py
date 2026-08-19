"""Training: supervised pretraining, then the verifier-guided loop."""

from .loop import (
    Accepted,
    LoopConfig,
    RepairPair,
    RoundReport,
    TrainingSet,
    collapse_warning,
    run,
    run_round,
)
from .pretrain import (
    EVAL_RATIOS,
    TrainConfig,
    describe_device,
    evaluate,
    load_checkpoint,
    pick_device,
    save_checkpoint,
    smoke,
    train,
)

__all__ = [
    "EVAL_RATIOS",
    "Accepted",
    "LoopConfig",
    "RepairPair",
    "RoundReport",
    "TrainConfig",
    "TrainingSet",
    "collapse_warning",
    "describe_device",
    "evaluate",
    "load_checkpoint",
    "pick_device",
    "run",
    "run_round",
    "save_checkpoint",
    "smoke",
    "train",
]
