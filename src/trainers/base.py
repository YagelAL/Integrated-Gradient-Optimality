"""Training loop for any BaseRecommender."""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.config.schemas import TrainerCfg
from src.recommenders.base import BaseRecommender
from src.trainers.metrics import Metric


class Trainer:
    """Generic training loop.

    Delegates to model.calculate_loss for training and
    model.full_sort_predict for evaluation.

    Responsibilities:

    * Forward / backward / optimizer step per batch.
    * Optional LR scheduler step per epoch.
    * Streaming evaluation via Metric objects.
    * Early stopping on a primary metric.
    * Best-state-dict snapshotting and restore at end-of-training, so
      the returned model is the best-on-validation, not the last epoch.
    * Optional W&B step + epoch logging via a passed-in wandb_run.
    """

    def __init__(
        self,
        *,
        model: BaseRecommender,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        cfg: TrainerCfg,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        metrics: Iterable[Metric] = (),
        device: torch.device | str = "cpu",
        wandb_run: Any = None,
        primary_metric: str = "hr@10",
        primary_metric_goal: str = "maximize",
        metric_prefix: str = "val",
    ) -> None:
        """Configure the trainer.

        Args:
            model: Recommender to train; moved to device in-place.
            train_loader: Yields training batches consumable by
                model.calculate_loss.
            val_loader: Yields validation batches consumable by
                model.full_sort_predict.
            optimizer: Optimizer over model.parameters().
            cfg: Training schedule (epochs, eval cadence, early-stopping).
            scheduler: Optional LR scheduler stepped once per epoch.
            metrics: Streaming Metric objects evaluated on val_loader
                after each eval epoch.
            device: Target device for model and batches.
            wandb_run: Optional live wandb.Run for step / epoch logs.
            primary_metric: Metric name used for best-model selection
                and early stopping (must match a metric's name).
            primary_metric_goal: "maximize" or "minimize".
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.metrics = list(metrics)
        self.device = torch.device(device)
        self.wandb_run = wandb_run
        self.primary_metric = primary_metric
        self.primary_metric_goal = primary_metric_goal
        self.metric_prefix = metric_prefix

        self._best_value: float = float("-inf") if primary_metric_goal == "maximize" else float("inf")
        self._best_state_dict: dict[str, torch.Tensor] | None = None
        self._epochs_no_improve: int = 0
        self._global_step: int = 0

    def train_one_epoch(self) -> float:
        """Run one training pass over train_loader.

        Returns:
            Mean per-batch training loss across the epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in self.train_loader:
            batch = self._to_device(batch)
            loss = self.model.calculate_loss(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1
            self._global_step += 1
            if self.wandb_run is not None and self._global_step % 50 == 0:
                self.wandb_run.log(
                    {"train/loss_step": float(loss.item()), "step": self._global_step}
                )
        if self.scheduler is not None:
            self.scheduler.step()
        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Run one validation pass and collect all configured metrics.

        Returns:
            Mapping of metric name (Metric.name) to its scalar value.
        """
        self.model.eval()
        for m in self.metrics:
            m.reset()
        for batch in self.val_loader:
            batch = self._to_device(batch)
            scores, labels = self.model.full_sort_predict(batch)
            for m in self.metrics:
                m.update(scores, labels)
        return {m.name: m.compute() for m in self.metrics}

    def fit(self, *, restore_best: bool = True) -> dict[str, float]:
        """Run the full training loop with eval, early stopping, and best-state restore.

        Trains for up to cfg.num_epochs, evaluates every
        cfg.eval_every_n_epochs, and restores the best-on-validation
        state-dict before returning.

        Returns:
            Validation metrics from the best epoch, or an empty dict if no
            evaluation ever ran.
        """
        best_metrics: dict[str, float] = {}
        for epoch in range(1, self.cfg.num_epochs + 1):
            t0 = time.time()
            train_loss = self.train_one_epoch()

            if epoch % self.cfg.eval_every_n_epochs != 0:
                continue

            val_metrics = self.evaluate()
            if self.wandb_run is not None:
                self.wandb_run.log(
                    {
                        **{f"{self.metric_prefix}/{k}": v for k, v in val_metrics.items()},
                        "train/loss_epoch": train_loss,
                        "epoch": epoch,
                    }
                )

            improved = self._is_improvement(val_metrics)
            if improved:
                self._best_value = val_metrics[self.primary_metric]
                if restore_best:
                    self._best_state_dict = {
                        k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                    }
                self._epochs_no_improve = 0
                best_metrics = dict(val_metrics)
            else:
                self._epochs_no_improve += 1

            metric_str = "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            print(
                f"epoch {epoch:3d}  loss={train_loss:.4f}  {metric_str}  "
                f"({time.time() - t0:.1f}s)" + ("  *" if improved else "")
            )

            if (
                epoch >= self.cfg.early_stopping_min_epochs
                and self._epochs_no_improve >= self.cfg.early_stopping_patience
            ):
                print(
                    f"early stopping at epoch {epoch} "
                    f"(no improvement for {self._epochs_no_improve} evals)"
                )
                break

        if self._best_state_dict is not None:
            self.model.load_state_dict(self._best_state_dict)
        return best_metrics

    def _is_improvement(self, metrics: dict[str, float]) -> bool:
        """Check whether the latest metrics beat the running best.

        Args:
            metrics: Validation metrics from the most recent eval epoch.

        Returns:
            True if metrics[primary_metric] improves on the running best
            (per primary_metric_goal); False otherwise or if the primary
            metric is absent.
        """
        value = metrics.get(self.primary_metric)
        if value is None:
            return False
        if self.primary_metric_goal == "maximize":
            return value > self._best_value
        return value < self._best_value

    def _to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move tensor values in a batch dict onto the trainer's device.

        Args:
            batch: Dict whose values may be tensors or arbitrary objects.

        Returns:
            New dict with tensor values transferred to self.device;
            non-tensor values are passed through unchanged.
        """
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def save_checkpoint(self, path: str | Path) -> None:
        """Save the best (or current) model state to a .pt file.

        Args:
            path: Filesystem destination for torch.save.
        """
        state = self._best_state_dict or self.model.state_dict()
        torch.save(state, path)


__all__ = ["Trainer"]
