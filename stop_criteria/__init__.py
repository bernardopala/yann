"""Stop criteria available to the training API."""

from .max_epochs import MaxEpochs
from .max_training_time import MaxTrainingTime
from .stop_criterion import StopCriterion
from .target_loss import TargetLoss

__all__ = ["StopCriterion", "MaxEpochs", "MaxTrainingTime", "TargetLoss"]
