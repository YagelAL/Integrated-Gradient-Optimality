from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import pandas as pd

from src.config.paths import SplitPaths
from src.config.schemas import DatasetCfg, PreprocessCfg
from src.preprocess.base import PipelineResult
from src.preprocess.filters import IterativeNCoreFilter
from src.preprocess.indexes import LabelEncoder
from src.preprocess.splits import SplitResult, build_splitter

# Canonical column names written to CSV; the raw per-dataset names are
# dropped after preprocessing.
USER_COL = "user_id"
ITEM_COL = "item_id"
TIMESTAMP_COL = "timestamp"
RATING_COL = "rating"


def run_preprocess_pipeline(cfg: PreprocessCfg) -> PipelineResult:
    """Execute the unified preprocessing pipeline end-to-end.

    Loads the raw interaction table, applies the optional rating filter and
    the iterative N-core filter, label-encodes users and items, runs the
    random user-level split, and writes canonical CSVs together with the
    encoder and meta JSON sidecars.

    Returns a :class:`PipelineResult` describing the artifacts produced.
    """
    dataset = cfg.dataset

    df = _load_raw(dataset)
    df = _apply_rating_threshold(df, dataset)

    filt = IterativeNCoreFilter(
        cfg.filters,
        user_col=dataset.user_col,
        item_col=dataset.item_col,
        timestamp_col=dataset.timestamp_col,
    )
    df, filter_report = filt.apply(df)

    # Split BEFORE encoding items, so the encoder can order ids by the earliest
    # split each item appears in. Splitting only reads user ids, so raw item
    # values are fine here.
    raw_cols: dict[str, Any] = {
        USER_COL: df[dataset.user_col].to_numpy(),
        ITEM_COL: df[dataset.item_col].to_numpy(),
    }
    if dataset.supports_temporal:
        raw_cols[TIMESTAMP_COL] = df[dataset.timestamp_col].to_numpy()
    if dataset.rating_col is not None:
        raw_cols[RATING_COL] = df[dataset.rating_col].to_numpy()
    raw_canonical = pd.DataFrame(raw_cols)

    splitter = build_splitter(cfg.split, user_col=USER_COL)
    split_result = splitter.split(raw_canonical)

    # Tiered item encoding: train items first, then val-only, then test-only.
    # Each phase's warm vocabulary is therefore a contiguous prefix of the id
    # space (sweeps: 1..num_items_train; finals: 1..num_items_trainval), so a
    # recommender is built over the warm items without any id remapping. The
    # trailing raw tier keeps the encoder total.
    # Items offset by 1 so id 0 is reserved as the PAD slot.
    item_encoder = LabelEncoder.fit_tiered(
        [
            split_result.train[ITEM_COL],
            split_result.val[ITEM_COL],
            split_result.test[ITEM_COL],
            raw_canonical[ITEM_COL],
        ],
        offset=1,
    )
    user_encoder = LabelEncoder.fit(raw_canonical[USER_COL], offset=0)

    def _encode(frame: pd.DataFrame) -> pd.DataFrame:
        encoded = frame.copy()
        encoded[USER_COL] = user_encoder.transform(frame[USER_COL])
        encoded[ITEM_COL] = item_encoder.transform(frame[ITEM_COL])
        return encoded

    split_result = SplitResult(
        train=_encode(split_result.train),
        val=_encode(split_result.val),
        test=_encode(split_result.test),
        meta=split_result.meta,
    )

    # Contiguous by construction; recorded so training can size the model
    # per phase without rescanning the CSVs.
    num_items_train = int(split_result.train[ITEM_COL].nunique())
    num_items_trainval = int(
        pd.concat([split_result.train[ITEM_COL], split_result.val[ITEM_COL]]).nunique()
    )

    paths = SplitPaths(
        output_root=cfg.output_root,
        dataset_name=dataset.name,
        split=cfg.split,
    )
    paths.ensure_dirs()
    split_result.train.to_csv(paths.train_path, index=False)
    split_result.val.to_csv(paths.val_path, index=False)
    split_result.test.to_csv(paths.test_path, index=False)

    item_encoder.to_json(paths.split_dir / "item_encoder.json")
    user_encoder.to_json(paths.split_dir / "user_encoder.json")

    meta = {
        "dataset": dataset.name,
        "num_users": user_encoder.vocab_size,
        "num_items": item_encoder.vocab_size,
        # Warm prefixes: items reachable by gradient in each training phase.
        "num_items_train": num_items_train,
        "num_items_trainval": num_items_trainval,
        "supports_temporal": dataset.supports_temporal,
        "rows": {
            "train": len(split_result.train),
            "val": len(split_result.val),
            "test": len(split_result.test),
        },
        "split": split_result.meta,
        "filter_report": asdict(filter_report),
    }
    paths.meta_path.write_text(json.dumps(meta, indent=2))

    return PipelineResult(
        dataset_name=dataset.name,
        train_path=paths.train_path,
        val_path=paths.val_path,
        test_path=paths.test_path,
        meta_path=paths.meta_path,
        num_users=user_encoder.vocab_size,
        num_items=item_encoder.vocab_size,
        filter_report=filter_report,
    )


def _load_raw(dataset: DatasetCfg) -> pd.DataFrame:
    """Load raw interaction data per the :class:`DatasetCfg` format."""
    if dataset.format in ("csv", "tsv", "dat"):
        sep = "\t" if dataset.format == "tsv" else dataset.csv_sep
        return pd.read_csv(dataset.raw_path, sep=sep)
    if dataset.format == "json":
        return pd.read_json(dataset.raw_path, lines=True)
    raise ValueError(f"unknown dataset format: {dataset.format!r}")


def _apply_rating_threshold(df: pd.DataFrame, dataset: DatasetCfg) -> pd.DataFrame:
    """Drop interactions below ``rating_threshold`` when both rating columns are configured."""
    if dataset.rating_col is None or dataset.rating_threshold is None:
        return df
    return df[df[dataset.rating_col] >= dataset.rating_threshold]


__all__ = [
    "ITEM_COL",
    "RATING_COL",
    "TIMESTAMP_COL",
    "USER_COL",
    "run_preprocess_pipeline",
]
