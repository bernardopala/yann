"""Loss-based training stop criterion."""

from .stop_criterion import StopCriterion


class TargetLoss(StopCriterion):
    """Stop when the latest epoch loss reaches the target value."""

    def __init__(self, target_loss: float) -> None:
        if target_loss < 0:
            raise ValueError("target_loss must be non-negative")

        self.target_loss = float(target_loss)

    def reached(self, epochs_completed: int, elapsed_seconds: float, loss: float | None = None) -> bool:
        return loss is not None and loss <= self.target_loss
