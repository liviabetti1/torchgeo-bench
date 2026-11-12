"""Runner for the CoordBench location-encoder track.

Driven by a :class:`CoordConfig`: instantiate a coordinate encoder from its
model preset, embed each benchmark's points once, then probe
with linear, KNN, and/or IDW heads under an official test mask or nested CV
(see splits.py). One CSV row per (benchmark, task, method) is appended
to ``output.file`` via the shared atomic writer, with resume support.
"""

import itertools
import logging
import os
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.progress import Progress
import torch

from torchgeo_bench.config.presets import ModelPreset, build_model
from torchgeo_bench.config.schema import resolve_output_path
from torchgeo_bench.coordbench.config import (
    CoordConfig,
    CoordEvaluationConfig,
    resolve_coord_preset
)
from torchgeo_bench.coordbench.config_enums import TemporalResolution
from torchgeo_bench.coordbench.datasets import CoordBenchmark, load_benchmarks
from torchgeo_bench.coordbench.spatial_aggregation import SPATIAL_AGGREGATION_METHODS, spatial_aggregation
from torchgeo_bench.coordbench.temporal_aggregation import (
    discretize_dataset_and_aggregate_embeddings,
    pool_embeddings,
    temporal_sample_and_aggregate_embeddings,
)
from torchgeo_bench.coordbench.models import LocationEncoder
from torchgeo_bench.coordbench.probe import (
    ProbeData,
    _valid_mask,
    evaluate,
    idw_predictor,
    knn_predictor,
    ridge_predictor,
)
from torchgeo_bench.coordbench.splits import make_folds, split_variants
from torchgeo_bench.devices import resolve_device
from torchgeo_bench.results import append_rows_atomic

logger = logging.getLogger(__name__)

RESUME_KEY_COLS = (
    "dataset",
    "task",
    "method",
    "model_name",
    "split_test_type",
    "split",
    "inner_split",
    "split_params",
    "temporal_embedding_aggregation",
)


def _split_params(
    coord: CoordEvaluationConfig, splits: tuple[str, str], time_unit: tuple[str, int]
) -> str:
    """Split settings that identify a result row, in the key order stored by earlier runs."""
    # test_size joins the key only for temporal splits, so random/spatial results stored before
    # it existed still resume.
    temporal = any(s.startswith("temporal") for s in splits)
    tail = {"reset_year", "max_train_size"} | ({"test_size"} if temporal else set())
    return str({
        **coord.model_dump(include={"folds", "inner_folds", "cell_deg"}),
        "time_unit": time_unit[0],
        "time_unit_size": time_unit[1],
        **coord.model_dump(include=tail),
    })


@dataclass
class CoordResult:
    """A single CoordBench evaluation result row."""

    dataset: str  # benchmark name (e.g. "sustainbench-asset")
    task: str  # label/column name within the benchmark
    task_type: str  # "regression" | "classification"
    method: str  # "linear" | "knn" | "idw"
    split_test_type: str  # "official" | "nested_cv"
    split: str
    inner_split: str
    split_params: str
    metric_name: str  # "r2" | "accuracy"
    metric_value: float  # mean over outer folds
    ci_lower: float  # metric_value - std over folds
    ci_upper: float  # metric_value + std over folds
    n_folds: int
    best_params: str  # tuned grid point per outer fold
    feature_dim: int
    n_samples: int
    n_test: int
    seed: int
    model_name: str
    model_target: str
    temporal_embedding_aggregation: str # "" (encoded per point) | "<window>/mean" (precomputed features)


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


def _predictor(method: str, data: ProbeData, bench: CoordBenchmark, cfg: CoordConfig) -> tuple:
    """Build one probe's ``predict`` function and its hyperparameter grid."""
    coord = cfg.evaluation
    if method == "linear":
        return ridge_predictor(data, cfg.runtime.device), coord.ridge_alphas
    if method == "knn":
        return knn_predictor(data, coord.knn_device), coord.knn_ks
    timestamp = bench.posix_timestamp
    predict = idw_predictor(data, bench.lat, bench.lon, timestamp, coord.knn_device)
    return predict, list(itertools.product(coord.idw_ks, coord.idw_cs))


def _pooled(
    bench: CoordBenchmark, embeddings: dict[str, torch.Tensor], aggregation: str
) -> tuple[CoordBenchmark, np.ndarray]:
    """Pool the embeddings and name the benchmark so each aggregation keeps its own result rows."""
    bench.name = f"{bench.name}_{aggregation}"
    return bench, pool_embeddings(embeddings, aggregation)


def _expand_temporal(
    benchmarks: Sequence[CoordBenchmark],
    discretization_methods: Sequence[str],
    temporal_sampling_methods: Sequence[str],
    embedding_aggregation_methods: Sequence[str],
    *,
    encoder: LocationEncoder,
) -> Iterator[tuple[CoordBenchmark, np.ndarray | None]]:
    """Discretize each timestamped benchmark per (discretization, aggregation) pair."""
    for bench in benchmarks:
        if bench.posix_timestamp is None:
            if temporal_sampling_methods is not None and len(temporal_sampling_methods) > 0:
                for sampling, discretization, aggregation in itertools.product(
                    temporal_sampling_methods, discretization_methods, embedding_aggregation_methods
                ):
                    sampled, embeddings = temporal_sample_and_aggregate_embeddings(
                        bench, encoder, sampling, discretization, aggregation
                    )
                    yield _pooled(sampled, embeddings, aggregation)
            else:
                yield bench, None
            continue

        elif bench.temporal_resolution != TemporalResolution.DAILY:
            for season, discretization, aggregation in itertools.product(
                temporal_sampling_methods or ["year"], discretization_methods, embedding_aggregation_methods
            ):
                sampled, embeddings = temporal_sample_and_aggregate_embeddings(
                    bench, encoder, season, discretization, aggregation
                )
                yield _pooled(sampled, embeddings, aggregation)
            continue


        label_stat = "mode" if bench.task_type == "classification" else "mean"
        for discretization in discretization_methods:
            for aggregation in embedding_aggregation_methods:
                discretized, embeddings = discretize_dataset_and_aggregate_embeddings(
                    bench, encoder, discretization, aggregation
                )
                discretized.tasks = {
                    name: values
                    for name, values in discretized.tasks.items()
                    if name.endswith(f"_{label_stat}")
                }
                yield _pooled(discretized, embeddings, aggregation)
                del discretized, embeddings


def _expected_benchmark_count(
    benchmarks: Sequence[CoordBenchmark], coord: CoordEvaluationConfig
) -> int:
    """Number of (benchmark, embeddings) pairs the expansion pipeline will yield."""
    with_timestamp = sum(b.posix_timestamp is not None for b in benchmarks)
    without_timestamp = 0 if coord.skip_no_timestamp else len(benchmarks) - with_timestamp
    methods_per_timestamp = 1
    if coord.temporal_discretization_methods:
        methods_per_timestamp = len(coord.temporal_discretization_methods) * len(
            coord.temporal_embedding_aggregation_methods
        )
    sampled = sum(b.posix_timestamp is None and b.year is not None for b in benchmarks)
    sampled *= (
        len(coord.temporal_sampling_methods)
        * len(coord.temporal_discretization_methods)
        * len(coord.temporal_embedding_aggregation_methods)
    )
    coarse = sum(
        b.posix_timestamp is not None and b.temporal_resolution != TemporalResolution.DAILY
        for b in benchmarks
    )
    seasons_per_coarse = max(1, len(coord.temporal_sampling_methods))
    count = (
        with_timestamp * methods_per_timestamp
        + coarse * methods_per_timestamp * (seasons_per_coarse - 1)
        + without_timestamp
        + sampled
    )
    if coord.from_polygon:
        count *= sum(m in SPATIAL_AGGREGATION_METHODS for m in coord.spatial_aggregation_methods)
    return count


def _expand_spatial(
        benchmarks: Sequence[tuple[CoordBenchmark, np.ndarray]],
        methods: Sequence[str],
        encoder: LocationEncoder | None = None,
) -> Iterator[tuple[CoordBenchmark, np.ndarray]]:
    for bench, embeddings in benchmarks:
        applicable_methods = [m for m in methods if m in SPATIAL_AGGREGATION_METHODS]
        for method in applicable_methods:
            windowed, emb = spatial_aggregation(bench, method, embeddings, encoder=encoder)
            windowed.name = f"{bench.name}-{method}"
            yield windowed, emb
            del windowed, emb


def run_coordbench(cfg: CoordConfig) -> None:
    """Run the CoordBench location-encoder benchmark for the configured model."""
    preset = resolve_coord_preset(cfg)
    device = str(resolve_device(cfg.runtime.device))
    torch.manual_seed(cfg.runtime.seed)
    if cfg.runtime.device != device:
        cfg = cfg.model_copy(update={"runtime": cfg.runtime.model_copy(update={"device": device})})

    coord = cfg.evaluation
    temporal_discretization_methods = list(coord.temporal_discretization_methods)
    temporal_sampling_methods = list(coord.temporal_sampling_methods)
    temporal_embedding_aggregation_methods = list(coord.temporal_embedding_aggregation_methods)
    from_polygon = coord.from_polygon
    should_subsample = coord.subsample
    spatial_aggregation_methods = list(coord.spatial_aggregation_methods)
    save_predictions = coord.save_predictions

    output_path = resolve_output_path(
        cfg.output.directory, cfg.output.file, "coordbench_results.csv"
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    encoder = _instantiate_encoder(preset, device)
    logger.info(
        "CoordBench: model=%s device=%s nested_splits=%s", preset.name, device, coord.nested_splits
    )

    completed = _completed_keys(output_path) if cfg.output.resume else set()
    if completed:
        logger.info("Resume mode: %d existing coord results in %s", len(completed), output_path)

    t0 = time.perf_counter()
    names = "all" if cfg.datasets == ["all"] else cfg.datasets
    benchmarks = load_benchmarks(names, from_polygon, should_subsample)

    if coord.temporal_sampling_year is not None:
        for bench in benchmarks:
            if bench.posix_timestamp is None:
                bench.year = coord.temporal_sampling_year


    logger.info("CoordBench: loaded %d benchmark(s) in %.1fs", len(benchmarks), time.perf_counter() - t0)

    if coord.exclude_antarctica:
        for bench in benchmarks:
            n_before = len(bench.lat)
            bench.filter_by_latitude(-60.0)
            if len(bench.lat) < n_before:
                bench.name = f"{bench.name}_no_antarctica"
                logger.info("Exclude Antarctica: %s %d -> %d points", bench.name, n_before, len(bench.lat))

    if temporal_discretization_methods or temporal_sampling_methods:
        all_benchmarks = _expand_temporal(
            benchmarks,
            temporal_discretization_methods,
            temporal_sampling_methods,
            temporal_embedding_aggregation_methods,
            encoder=encoder,
        )
    else:
        all_benchmarks = ((b, None) for b in benchmarks)

    if from_polygon:
        all_benchmarks = _expand_spatial(
            all_benchmarks,
            spatial_aggregation_methods,
            encoder=encoder
        )

    if bool(coord.skip_no_timestamp):
        all_benchmarks = (
            (b, emb)
            for b, emb in all_benchmarks
            if b.posix_timestamp is not None or emb is not None
        )

    with Progress() as progress:
        task_id = progress.add_task(
            "CoordBench", total=_expected_benchmark_count(benchmarks, coord)
        )
        for bench, emb in all_benchmarks:
            progress.update(task_id, description=f"CoordBench: {bench.name}")
            bench_t0 = time.perf_counter()
            n_rows = 0
            for row in _evaluate_benchmark(bench, encoder, cfg, preset, completed, save_predictions, output_path, emb):
                append_rows_atomic(output_path, [row])
                n_rows += 1
            logger.info(
                "CoordBench: %-40s total=%.1fs -> %d row(s)",
                bench.name,
                time.perf_counter() - bench_t0,
                n_rows,
            )
            progress.advance(task_id)
            del bench, emb


    logger.info("CoordBench complete. Results appended to %s", output_path)


def _evaluate_benchmark(
    bench: CoordBenchmark,
    encoder: LocationEncoder,
    cfg: CoordConfig,
    preset: ModelPreset,
    completed: set[tuple[str, ...]],
    save_predictions: bool,
    output_path: str,
    precomputed_features: np.ndarray | None = None,
) -> Iterator[dict[str, Any]]:
    """Embed one benchmark once and probe every (task, outer split, method) combination."""
    coord, seed = cfg.evaluation, cfg.runtime.seed
    needs_timestamps = any(s.startswith("temporal") for pair in coord.nested_splits for s in pair)
    if (coord.split_test_type == "official" and bench.test_mask is None) or (
        needs_timestamps and bench.posix_timestamp is None
    ):
        logger.info("%s: skipping, no test_mask or timestamps for this split", bench.name)
        return
    methods = [m for m in coord.methods if m != "knn" or bench.task_type == "classification"]
    features = precomputed_features
    temporal_embedding_aggregation = (
        f"{bench.temporal_resolution}/mean" if precomputed_features is not None else ""
    )

    for task, labels in bench.tasks.items():
        todo = {
            (outer, inner, unit): [
                m
                for m in methods
                if (bench.name, task, m, preset.name, coord.split_test_type, outer, inner,
                    _split_params(coord, (outer, inner), unit), temporal_embedding_aggregation)
                not in completed
            ]
            for outer, inner, unit in split_variants(coord)
        }  # fmt: skip
        todo = {variant: ms for variant, ms in todo.items() if ms}
        if not todo:
            continue
        if features is None:
            features = encoder.encode(bench.lon, bench.lat, bench.posix_timestamp)
        labels = np.asarray(labels)
        valid_rows = np.flatnonzero(_valid_mask(features, labels, bench.task_type))
        folds_by_split = make_folds(coord, bench, valid_rows, seed)
        if bench.task_type == "regression":
            data = ProbeData(features, labels.astype(np.float64), bench.task_type, None)
        else:
            class_indices = np.full(len(labels), -1)
            class_indices[valid_rows] = np.unique(labels[valid_rows], return_inverse=True)[1]
            data = ProbeData(features, labels, bench.task_type, class_indices)

        for (outer_split, inner_split, unit), split_methods in todo.items():
            folds = folds_by_split[outer_split, inner_split, unit]
            for method in split_methods:
                result = evaluate(*_predictor(method, data, bench, cfg), folds, data)
                if save_predictions:
                    pred_dir = Path(output_path).parent / "predictions" / preset.name
                    pred_dir.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(
                        {"labels": labels, "predictions": result.predictions, "lat": bench.lat,
                         "lon": bench.lon, "timestamp": bench.posix_timestamp}
                    ).to_parquet(
                        pred_dir / f"{bench.name}_{task}_{method}_{outer_split}_{inner_split}_{unit[0]}{unit[1]}.parquet"
                    )  # fmt: skip
                std = float(np.std(result.fold_scores))
                yield CoordResult(
                    dataset=bench.name,
                    task=task,
                    task_type=bench.task_type,
                    method=method,
                    split_test_type=coord.split_test_type,
                    split=outer_split,
                    inner_split=inner_split,
                    split_params=_split_params(coord, (outer_split, inner_split), unit),
                    metric_name="r2" if bench.task_type == "regression" else "accuracy",
                    metric_value=result.test_score,
                    ci_lower=result.test_score - std,
                    ci_upper=result.test_score + std,
                    n_folds=len(folds),
                    best_params=str(result.best_params),
                    feature_dim=int(features.shape[1]),
                    n_samples=len(labels),
                    n_test=sum(len(fold.test) for fold in folds),
                    seed=seed,
                    model_name=preset.name,
                    model_target=preset.target,
                    temporal_embedding_aggregation=temporal_embedding_aggregation,
                ).to_row()
