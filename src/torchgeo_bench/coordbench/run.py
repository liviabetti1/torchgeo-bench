"""Runner for the CoordBench location-encoder track.

Driven by a :class:`CoordConfig`: instantiate a coordinate encoder from its
model preset, embed each benchmark's points once, then probe
with KNN and/or a ridge linear head under random and/or spatial-block
cross-validation. One CSV row per (benchmark, task, method, split) is appended
to ``output.file`` via the shared atomic writer, with resume support.
"""

import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from rich.progress import Progress
import torch

from torchgeo_bench.config.presets import ModelPreset, build_model
from torchgeo_bench.coordbench.config import (
    CoordConfig,
    CoordEvaluationConfig,
    resolve_coord_preset,
)
from torchgeo_bench.coordbench.datasets import CoordBenchmark, load_benchmarks
from torchgeo_bench.coordbench.temporal_aggregation import TEMPORAL_AGGREGATION_METHODS, temporal_aggregation
from torchgeo_bench.coordbench.models import LocationEncoder
from torchgeo_bench.coordbench.probe import (
    knn_probe_score,
    linear_probe_score
)
from torchgeo_bench.coordbench.splits import spatial_fold_ids
from torchgeo_bench.devices import resolve_device
from torchgeo_bench.results import append_rows_atomic

logger = logging.getLogger(__name__)

RESUME_KEY_COLS = ("dataset", "task", "method", "model_name", "split", "embedding_aggregation")


@dataclass
class CoordResult:
    """A single CoordBench evaluation result row."""

    dataset: str  # benchmark name (e.g. "sustainbench-asset")
    task: str  # label/column name within the benchmark
    task_type: str  # "regression" | "classification"
    method: str  # "linear" | "knn{k}"
    split: str  # "random" | "spatial" | "official"
    metric_name: str  # "r2" | "accuracy"
    metric_value: float  # mean over folds (or the held-out score)
    ci_lower: float  # metric_value - std over folds
    ci_upper: float  # metric_value + std over folds
    n_folds: int
    cell_deg: float
    feature_dim: int
    n_samples: int
    n_test: int
    seed: int
    model_name: str
    model_target: str
    embedding_aggregation: str  # "none" | "yearly:{year}:{freq}" | "summer:{year}:{freq}"

    def to_row(self) -> dict[str, Any]:
        """Convert to a flat dict suitable for CSV/DataFrame export."""
        return self.__dict__.copy()


def _instantiate_encoder(preset: ModelPreset, device: str) -> LocationEncoder:
    """Build a coordinate encoder using only its constructor options."""
    encoder = build_model(preset, device=device)
    if not isinstance(encoder, LocationEncoder):
        raise TypeError(
            f"coord requires model.target to be a LocationEncoder subclass; "
            f"got {type(encoder).__name__}. Pick a coordinate model, e.g. --model sincos."
        )
    return encoder


def _resolve_splits(split: str) -> list[str]:
    """Expand the selected CV modes to run."""
    if split == "both":
        return ["random", "spatial"]
    if split not in ("random", "spatial"):
        raise ValueError(f"evaluation.split must be random|spatial|both; got {split!r}")
    return [split]


def _completed_keys(output_path: str) -> set[tuple[str, ...]]:
    """Existing (dataset, task, method, model_name, split) keys for resume."""
    if not os.path.exists(output_path):
        return set()
    df = pd.read_csv(output_path)
    for col in RESUME_KEY_COLS:
        if col not in df.columns:
            return set()
    rows = df[list(RESUME_KEY_COLS)].fillna("").astype(str).to_numpy()
    return {tuple(r) for r in rows}


def _methods_for(task_type: str, requested: Sequence[str], knn_k: int) -> list[tuple[str, str]]:
    """Resolve (method-label, kind) pairs applicable to a task type.

    KNN is classification-only (there is no KNN-regression head here); the ridge
    linear probe handles both regression (R^2) and classification (accuracy).
    """
    methods: list[tuple[str, str]] = []
    if "knn" in requested and task_type == "classification":
        methods.append((f"knn{knn_k}", "knn"))
    if "linear" in requested:
        methods.append(("linear", "linear"))
    return methods


def _score_one(
    kind: str,
    features: np.ndarray,
    labels: np.ndarray,
    task_type: str,
    *,
    folds: int,
    seed: int,
    device: str,
    knn_device: str,
    knn_k: int,
    test_mask: np.ndarray | None,
    fold_assign: np.ndarray | None,
) -> tuple[float, list[float]]:
    """KNN or linear probe for one (dataset, task, method) combination."""
    if kind == "knn":
        return knn_probe_score(
            features,
            labels,
            folds=folds,
            seed=seed,
            k=knn_k,
            device=knn_device,
            test_mask=test_mask,
            fold_assign=fold_assign,
        )
    return linear_probe_score(
        features,
        labels,
        task_type,
        folds=folds,
        seed=seed,
        device=device,
        test_mask=test_mask,
        fold_assign=fold_assign,
    )


def _evaluation_split(
    bench: CoordBenchmark, split: str, coord: CoordEvaluationConfig, seed: int
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """Use an official holdout when present, otherwise the requested CV split."""
    if bench.test_mask is not None:
        return bench.test_mask, None, "official"
    if split == "spatial":
        assignment = spatial_fold_ids(bench.lat, bench.lon, coord.folds, coord.cell_deg, seed)
        return None, assignment, "spatial"
    return None, None, "random"


def _expand_temporal(
    benchmarks: Sequence[CoordBenchmark],
    methods: Sequence[str],
    *,
    encoder: LocationEncoder | None = None,
) -> list[tuple[CoordBenchmark, np.ndarray | None]]:
    """Replace each benchmark with one variant per temporal method applicable to its resolution.
    """
    expanded: list[tuple[CoordBenchmark, np.ndarray | None]] = []
    for bench in benchmarks:
        resolution = bench.temporal_resolution or "none"
        applicable_methods = [m for m in methods if m in TEMPORAL_AGGREGATION_METHODS.get(resolution, [])]
        for method in applicable_methods:
            windowed, emb = temporal_aggregation(bench, method, encoder=encoder)
            windowed.name = f"{bench.name}-{method}"
            expanded.append((windowed, emb))
    return expanded


def run_coordbench(cfg: CoordConfig) -> None:
    """Run the CoordBench location-encoder benchmark for the configured model."""
    preset = resolve_coord_preset(cfg)
    device = str(resolve_device(cfg.runtime.device))
    torch.manual_seed(cfg.runtime.seed)
    if cfg.runtime.device != device:
        cfg = cfg.model_copy(update={"runtime": cfg.runtime.model_copy(update={"device": device})})
    splits = _resolve_splits(cfg.evaluation.split)

    coord = cfg.evaluation
    seed = cfg.runtime.seed
    folds = coord.folds
    cell_deg = coord.cell_deg
    knn_k = coord.knn_k
    knn_device = coord.knn_device
    methods = coord.methods
    temporal_aggregation_methods = list(coord.temporal_aggregation_methods)
    aggregate_embeddings = coord.aggregate_embeddings
    model_name = preset.name
    model_target = preset.target

    output_path = cfg.output.file
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    encoder = _instantiate_encoder(preset, device)
    logger.info("CoordBench: model=%s device=%s splits=%s", preset.name, device, splits)

    completed = _completed_keys(output_path) if cfg.output.resume else set()
    if completed:
        logger.info("Resume mode: %d existing coord results in %s", len(completed), output_path)

    t0 = time.perf_counter()
    names = "all" if cfg.datasets == ["all"] else cfg.datasets
    benchmarks = load_benchmarks(names)
    logger.info("CoordBench: loaded %d benchmark(s) in %.1fs", len(benchmarks), time.perf_counter() - t0)

    t0 = time.perf_counter()
    if temporal_aggregation_methods:
        all_benchmarks = _expand_temporal(
            benchmarks,
            temporal_aggregation_methods,
            encoder=encoder if aggregate_embeddings else None,
        )
    else:
        all_benchmarks = [(b, None) for b in benchmarks]
    logger.info(
        "CoordBench: %d benchmarks selected (temporal expansion took %.1fs)",
        len(all_benchmarks),
        time.perf_counter() - t0,
    )

    if bool(coord.skip_no_timestamp):
        skipped = [b.name for b, emb in all_benchmarks if b.posix_timestamp is None and emb is None]
        if skipped:
            logger.info("Skipping %d benchmark(s) with no posix_timestamp: %s", len(skipped), skipped)
        all_benchmarks = [(b, emb) for b, emb in all_benchmarks if b.posix_timestamp is not None or emb is not None]

    with Progress() as progress:
        task_id = progress.add_task("CoordBench", total=len(all_benchmarks))
        for bench, emb in all_benchmarks:
            progress.update(task_id, description=f"CoordBench: {bench.name}")
            bench_t0 = time.perf_counter()
            rows, encode_s, probe_s = _evaluate_benchmark(
                bench,
                encoder,
                methods=methods,
                splits=splits,
                folds=folds,
                cell_deg=cell_deg,
                knn_k=knn_k,
                knn_device=knn_device,
                seed=seed,
                device=device,
                model_name=model_name,
                model_target=model_target,
                completed=completed if cfg.output.resume else None,
                precomputed_features=emb,
            )
            if rows:
                append_rows_atomic(output_path, rows)
            logger.info(
                "CoordBench: %-40s total=%.1fs (encode=%.1fs probe=%.1fs) -> %d row(s)",
                bench.name,
                time.perf_counter() - bench_t0,
                encode_s,
                probe_s,
                len(rows),
            )
            progress.advance(task_id)

    logger.info("CoordBench complete. Results appended to %s", output_path)


def test_sample_count(labels: np.ndarray, task_type: str, test_mask: np.ndarray | None) -> int:
    """Count held-out samples, or finite regression labels for cross-validation."""
    if test_mask is not None:
        return int(np.asarray(test_mask, dtype=bool).sum())
    if task_type == "regression":
        return int(np.isfinite(np.asarray(labels, dtype=np.float64)).sum())
    return len(labels)


def _evaluate_benchmark(
    bench: CoordBenchmark,
    encoder: LocationEncoder,
    *,
    methods: Sequence[str],
    splits: Sequence[str],
    folds: int,
    cell_deg: float,
    knn_k: int,
    knn_device: str,
    seed: int,
    device: str,
    model_name: str,
    model_target: str,
    completed: set[tuple[str, ...]] | None,
    precomputed_features: np.ndarray | None = None,
) -> tuple[list[dict], float, float]:
    """Embed one benchmark once and probe every (task, method, split) combination.

    Returns ``(rows, encode_seconds, probe_seconds)`` so the caller can log where
    time went for this benchmark.
    """
    metric_name = "r2" if bench.task_type == "regression" else "accuracy"
    method_kinds = _methods_for(bench.task_type, methods, knn_k)
    if not method_kinds:
        return [], 0.0, 0.0

    t0 = time.perf_counter()
    if precomputed_features is not None:
        features = precomputed_features
        embedding_aggregation = "embedding_mean"
    else:
        features = encoder.encode(bench.lon, bench.lat, bench.posix_timestamp)
        embedding_aggregation = ""
    encode_s = time.perf_counter() - t0

    feature_dim = int(features.shape[1])

    probe_s = 0.0
    rows: list[dict] = []
    for split in splits:
        # Official held-out split wins when present; else the requested CV mode.
        if bench.test_mask is not None:
            test_mask, fold_assign, split_label = bench.test_mask, None, "official"
        elif split == "spatial":
            test_mask, fold_assign, split_label = (
                None,
                spatial_fold_ids(bench.lat, bench.lon, folds, cell_deg, seed),
                "spatial",
            )
        else:
            test_mask, fold_assign, split_label = None, None, "random"

        for task, labels in bench.tasks.items():
            for method_label, kind in method_kinds:
                key = (bench.name, task, method_label, model_name, split_label, embedding_aggregation)
                if completed is not None and tuple(map(str, key)) in completed:
                    continue
                probe_t0 = time.perf_counter()
                score, fold_scores = _score_one(
                    kind,
                    features,
                    np.asarray(labels),
                    bench.task_type,
                    folds=folds,
                    seed=seed,
                    device=device,
                    knn_device=knn_device,
                    knn_k=knn_k,
                    test_mask=test_mask,
                    fold_assign=fold_assign,
                )
                probe_s += time.perf_counter() - probe_t0
                std = float(np.std(fold_scores)) if len(fold_scores) > 1 else 0.0
                if test_mask is not None:
                    n_test = int(np.asarray(test_mask, dtype=bool).sum())
                elif bench.task_type == "regression":
                    n_test = int(np.isfinite(np.asarray(labels, dtype=np.float64)).sum())
                else:
                    n_test = int(len(labels))
                rows.append(
                    CoordResult(
                        dataset=bench.name,
                        task=task,
                        task_type=bench.task_type,
                        method=method_label,
                        split=split_label,
                        metric_name=metric_name,
                        metric_value=score,
                        ci_lower=score - std,
                        ci_upper=score + std,
                        n_folds=1 if split_label == "official" else folds,
                        cell_deg=cell_deg,
                        feature_dim=feature_dim,
                        n_samples=len(labels),
                        n_test=n_test,
                        seed=seed,
                        model_name=model_name,
                        model_target=model_target,
                        embedding_aggregation=embedding_aggregation,
                    ).to_row()
                )
        # A benchmark with an official split is split-invariant; don't re-run per CV mode.
        if bench.test_mask is not None:
            break
    return rows, encode_s, probe_s
