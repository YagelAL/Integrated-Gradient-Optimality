"""Typed Pydantic schemas for Hydra-supplied configuration.

Hydra hands over nested dicts at runtime; they are validated through these
schemas at the entry points. All schemas are frozen and reject unknown keys
(``extra="forbid"``), so YAML typos fail at parse time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.capabilities import Family


# ---------------------------------------------------------------------------
# Split, filtering, dataset.
# ---------------------------------------------------------------------------


class RandomFractionSplit(BaseModel):
    """User-level random holdout: test/val users are entirely held out."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    strategy: Literal["random_fraction"] = "random_fraction"
    test_fraction: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.2
    val_fraction: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.1
    seed: Annotated[int, Field(ge=0)] = 42

    @model_validator(mode="after")
    def _fractions_sum(self) -> RandomFractionSplit:
        if self.test_fraction + self.val_fraction >= 1.0:
            raise ValueError(
                f"test_fraction ({self.test_fraction}) + val_fraction "
                f"({self.val_fraction}) must be < 1.0"
            )
        return self


class FilterCfg(BaseModel):
    """Iterative N-core sparsity filter applied during preprocessing."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    min_items_per_user: Annotated[int, Field(ge=1)] = 5
    min_users_per_item: Annotated[int, Field(ge=1)] = 5
    drop_consecutive_repeats: bool = False


class DatasetCfg(BaseModel):
    """Raw dataset location and column mapping.

    ``user_col`` / ``item_col`` / ``timestamp_col`` map the raw file into the
    canonical ``(user_id, item_id, timestamp)`` schema. ``timestamp_col`` is
    optional (Pinterest, Yahoo! Music and the synthetic data have none).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    raw_path: Path
    user_col: str
    item_col: str
    timestamp_col: str | None = None
    rating_col: str | None = None
    rating_threshold: float | None = None  # e.g., Yahoo keeps ratings >= 70.
    format: Literal["csv", "tsv", "dat", "json"] = "csv"
    csv_sep: str = ","

    @property
    def supports_temporal(self) -> bool:
        """True iff the dataset has a timestamp column."""
        return self.timestamp_col is not None


class PreprocessCfg(BaseModel):
    """Top-level preprocess invocation: dataset x filters x split."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    dataset: DatasetCfg
    filters: FilterCfg = Field(default_factory=FilterCfg)
    split: RandomFractionSplit = Field(default_factory=RandomFractionSplit)
    output_root: Path


# ---------------------------------------------------------------------------
# Recommender architecture config.
# ---------------------------------------------------------------------------


class RecommenderCfg(BaseModel):
    """Architecture-level recommender config."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    name: Literal["mlp", "mlp_deep"]
    family: Family
    hidden_dim: Annotated[int, Field(ge=8)]
    dropout: Annotated[float, Field(ge=0.0, le=1.0)] = 0.2
    beta: Annotated[float, Field(ge=0.0)] | None = None  # negative-loss weight


# ---------------------------------------------------------------------------
# Optimizer / scheduler / trainer.
# ---------------------------------------------------------------------------


class OptimizerCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: Literal["adam", "adamw"] = "adamw"
    lr: Annotated[float, Field(gt=0.0)] = 1e-3
    weight_decay: Annotated[float, Field(ge=0.0)] = 1e-3


class SchedulerCfg(BaseModel):
    """LR scheduler. ``none`` keeps LR constant."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    name: Literal["none", "cosine_warmup", "step"] = "cosine_warmup"
    warmup_steps: Annotated[int, Field(ge=0)] = 0
    # Used only by ``step``.
    decay_epochs: Annotated[int, Field(ge=1)] = 10
    decay_gamma: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.1


class TrainerCfg(BaseModel):
    """Training-loop hyperparameters."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    batch_size: Annotated[int, Field(ge=1)] = 256
    num_epochs: Annotated[int, Field(ge=1)] = 100
    optimizer: OptimizerCfg = Field(default_factory=OptimizerCfg)
    scheduler: SchedulerCfg = Field(default_factory=SchedulerCfg)
    early_stopping_patience: Annotated[int, Field(ge=1)] = 3
    early_stopping_min_epochs: Annotated[int, Field(ge=0)] = 5
    eval_every_n_epochs: Annotated[int, Field(ge=1)] = 1
    deterministic: bool = True
    seed: Annotated[int, Field(ge=0)] = 42
    num_workers: Annotated[int, Field(ge=0)] = 4


class WandbCfg(BaseModel):
    """W&B run settings. ``entity`` falls back to the ``WANDB_ENTITY`` env var."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    project: str
    entity: str | None = None
    tags: tuple[str, ...] = ()
    artifact_alias: str = "best"


# ---------------------------------------------------------------------------
# Explainer eval.
# ---------------------------------------------------------------------------


class FidelityMetricCfg(BaseModel):
    """One fidelity metric to evaluate (registry name + builder params)."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    params: dict = Field(default_factory=dict)


class FidelityCfg(BaseModel):
    """Fidelity-evaluator wiring: grids, metrics, item ranks.

    ``item_ranks`` picks which recommendations to attribute (0 = top-1).
    Metrics with ``depends_on_kr=False`` ignore ``kr_grid``.

    ``monotonic``: when ``True``, every slice emits the monotonic-corrected
    value (per-user ``cummin`` / ``cummax`` along ``ke``) plus a
    ``_raw``-suffixed sibling with the uncorrected mean. ``save_artifacts``
    writes a per-user CSV under ``hydra.run.dir/details``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    metrics: tuple[FidelityMetricCfg, ...]
    ke_grid: tuple[Annotated[int, Field(ge=1)], ...] = (1, 5, 10)
    kr_grid: tuple[Annotated[int, Field(ge=1)], ...] = (10, 20)
    item_ranks: tuple[Annotated[int, Field(ge=0)], ...] = (0,)
    monotonic: bool = True
    save_artifacts: bool = False


class RecommenderArtifactCfg(BaseModel):
    """Where to load a trained checkpoint from for eval.

    Either ``wandb_path`` (``"entity/project/name:alias"``) or
    ``local_path`` must be set; ``wandb_path`` takes precedence.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    wandb_path: str | None = None
    local_path: Path | None = None

    @model_validator(mode="after")
    def _at_least_one_source(self) -> RecommenderArtifactCfg:
        if self.wandb_path is None and self.local_path is None:
            raise ValueError(
                "RecommenderArtifactCfg requires either wandb_path or local_path"
            )
        return self


__all__ = [
    "DatasetCfg",
    "FidelityCfg",
    "FidelityMetricCfg",
    "FilterCfg",
    "OptimizerCfg",
    "PreprocessCfg",
    "RandomFractionSplit",
    "RecommenderArtifactCfg",
    "RecommenderCfg",
    "SchedulerCfg",
    "TrainerCfg",
    "WandbCfg",
]
