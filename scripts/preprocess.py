"""Hydra entry point for the unified preprocessing pipeline.

Composes ``configs/preprocess.yaml`` (defaults ``dataset=ml1m``,
``split=random_fraction``, ``filters=default``), validates the resolved config
through :class:`PreprocessCfg`, and runs :func:`run_preprocess_pipeline`.

Examples::

    RAW_DATA_DIR=/data python scripts/preprocess.py dataset=ml1m split=random_fraction
    RAW_DATA_DIR=/data python scripts/preprocess.py dataset=yahoo split=random_fraction \\
        filters.drop_consecutive_repeats=false
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from src.config.schemas import PreprocessCfg  # noqa: E402
from src.preprocess.pipeline import run_preprocess_pipeline  # noqa: E402


@hydra.main(version_base=None, config_path="../configs", config_name="preprocess")
def main(cfg: DictConfig) -> None:
    print("Resolved config:")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    container = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(container, dict)
    # Strip Hydra-internal keys before Pydantic validation (`extra="forbid"`).
    container.pop("hydra", None)
    preprocess_cfg = PreprocessCfg.model_validate(container)

    result = run_preprocess_pipeline(preprocess_cfg)

    print("Preprocessing complete:")
    print(f"  dataset:    {result.dataset_name}")
    print(f"  num_users:  {result.num_users}")
    print(f"  num_items:  {result.num_items}")
    print(f"  filter:     {result.filter_report}")
    print(f"  train:      {result.train_path}")
    print(f"  val:        {result.val_path}")
    print(f"  test:       {result.test_path}")
    print(f"  meta:       {result.meta_path}")


if __name__ == "__main__":
    main()
