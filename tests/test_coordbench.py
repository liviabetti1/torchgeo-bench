"""Tests for the CoordBench location-encoder track (network-free)."""

from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch
from pydantic import ValidationError

from tests.support.numerical import isolated_torch_rng as isolated_torch_rng
from torchgeo_bench.config.presets import ModelPreset
from torchgeo_bench.coordbench import (
    CoordBenchmark,
    SinCosLocationEncoder,
    load_benchmarks,
    run_coordbench,
)
from torchgeo_bench.coordbench import datasets as cb_datasets
from torchgeo_bench.coordbench.config import CoordConfig, CoordEvaluationConfig
from torchgeo_bench.coordbench.probe import (
    ProbeData,
    ProbeResult,
    evaluate,
    knn_predictor,
    ridge_predictor,
)
from torchgeo_bench.coordbench.run import _instantiate_encoder
from torchgeo_bench.coordbench.splits import make_folds, spatial_cell_ids, time_unit_ids

pytestmark = pytest.mark.usefixtures("isolated_torch_rng")


def _probe_data(features: np.ndarray, labels: np.ndarray, task_type: str) -> ProbeData:
    if task_type == "regression":
        return ProbeData(features, labels.astype(np.float64), task_type, None)
    return ProbeData(features, labels, task_type, labels)


def _folds(lon: np.ndarray, lat: np.ndarray, **evaluation: Any) -> list:
    bench = CoordBenchmark("synthetic", lat, lon)
    cfg = CoordEvaluationConfig(**evaluation)
    (folds,) = make_folds(cfg, bench, np.arange(len(lat)), seed=0).values()
    return folds


def _ridge_score(features: np.ndarray, labels: np.ndarray, *, device: str = "cpu") -> ProbeResult:
    lon, lat = features[:, 0], features[:, 1]
    data = _probe_data(features, labels, "regression")
    return evaluate(ridge_predictor(data, device), [0.1], _folds(lon, lat), data)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def points(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    lon = rng.uniform(-180, 180, size=400)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, size=400)))
    return lon, lat


@pytest.mark.parametrize("batch_size", [1, 137, 8192])
def test_sincos_encode_preserves_coordinate_order_across_batches(
    points: tuple[np.ndarray, np.ndarray], batch_size: int
) -> None:
    lon, lat = points
    feats = SinCosLocationEncoder(device="cpu", batch_size=batch_size).encode(lon, lat)
    assert feats.dtype == np.float32
    lat_r, lon_r = np.deg2rad(lat), np.deg2rad(lon)
    expected = np.column_stack([np.sin(lat_r), np.cos(lat_r), np.sin(lon_r), np.cos(lon_r)])
    np.testing.assert_allclose(feats, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    ("device", "expected"),
    [("auto", "cuda:1"), ("cuda", "cuda:1"), ("cuda:0", "cuda:0"), ("meta", "meta")],
)
def test_encoder_resolves_devices_without_changing_features(
    points: tuple[np.ndarray, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    expected: str,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    encoder = SinCosLocationEncoder(device=device)
    assert encoder.device == expected
    np.testing.assert_array_equal(encoder.encode(*points), SinCosLocationEncoder().encode(*points))


@pytest.mark.parametrize(
    ("device", "available", "message"),
    [
        ("cuda", False, "CUDA is unavailable"),
        ("cuda:0", False, "CUDA is unavailable"),
        ("cuda:2", True, "CUDA index 2"),
        ("cuda:-1", True, "Invalid device request"),
    ],
)
def test_encoder_and_linear_probe_reject_invalid_devices(
    monkeypatch: pytest.MonkeyPatch, device: str, *, available: bool, message: str
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match=message):
        SinCosLocationEncoder(device=device)
    with (
        mock.patch.object(torch, "as_tensor") as allocate,
        pytest.raises(ValueError, match=message),
    ):
        ridge_predictor(_probe_data(np.ones((10, 2)), np.arange(10), "regression"), device)
    allocate.assert_not_called()


def test_coordinate_cpu_paths_do_not_query_cuda(
    points: tuple[np.ndarray, np.ndarray], monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_query() -> bool:
        pytest.fail("CPU encoders and probes must not query CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected_query)
    features = SinCosLocationEncoder().encode(*points)
    assert _ridge_score(features, features[:, 0]).test_score > 0.99


@pytest.mark.parametrize(("available", "expected"), [(False, "cpu"), (True, "cuda:1")])
def test_coordinate_auto_probe_selects_device_and_preserves_solution(
    points: tuple[np.ndarray, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
    *,
    available: bool,
    expected: str,
) -> None:
    features = SinCosLocationEncoder().encode(*points)
    expected_score = _ridge_score(features, features[:, 0]).test_score
    devices = []
    as_tensor = torch.as_tensor

    def cpu_tensor(data: object, **kwargs: Any) -> torch.Tensor:
        devices.append(str(kwargs.pop("device")))
        return as_tensor(data, **kwargs)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch, "as_tensor", cpu_tensor)
    score = _ridge_score(features, features[:, 0], device="auto").test_score
    assert devices
    assert set(devices) == {expected}
    assert score == expected_score


def test_documented_fourier_encoder_example(points: tuple[np.ndarray, np.ndarray]) -> None:
    from docs.examples.coordbench_location_encoder import FourierLocationEncoder

    lon, lat = points
    feats = FourierLocationEncoder(num_frequencies=4).encode(lon, lat)
    assert feats.shape == (len(lon), 16)
    assert feats.dtype == np.float32
    assert np.isfinite(feats).all()


def test_linear_probe_regression_recovers_smooth_target(
    points: tuple[np.ndarray, np.ndarray],
) -> None:
    # Ridge regression should recover a target built directly from the features.
    lon, lat = points
    feats = SinCosLocationEncoder(device="cpu").encode(lon, lat)
    labels = 3.0 * feats[:, 0] - 2.0 * feats[:, 3] + 0.5  # 3*sin(lat) - 2*cos(lon) + b
    data = _probe_data(feats, labels, "regression")
    result = evaluate(ridge_predictor(data, "cpu"), [0.01, 1.0], _folds(lon, lat), data)
    assert len(result.fold_scores) == 5
    assert result.test_score > 0.98


def test_linear_and_knn_classification(points: tuple[np.ndarray, np.ndarray]) -> None:
    lon, lat = points
    feats = SinCosLocationEncoder(device="cpu").encode(lon, lat)
    labels = (lat > 0).astype(np.int64)  # The sign of sin(latitude) separates the hemispheres.
    data = _probe_data(feats, labels, "classification")
    folds = _folds(lon, lat)
    lin = evaluate(ridge_predictor(data, "cpu"), [0.01, 1.0], folds, data)
    knn = evaluate(knn_predictor(data, "cpu"), [5, 15], folds, data)
    assert lin.test_score > 0.9
    assert knn.test_score > 0.9
    assert len(knn.fold_scores) == 5


def _timestamped_bench(n: int = 400) -> CoordBenchmark:
    rng = np.random.default_rng(0)
    lat, lon = rng.uniform(-60, 60, n), rng.uniform(-180, 180, n)
    t = rng.uniform(1.5e9, 1.5e9 + 6 * 365 * 86400, n)
    return CoordBenchmark("synthetic", lat, lon, posix_timestamp=t)


@pytest.mark.parametrize(
    "evaluation",
    [
        {"nested_splits": [("random", "random")]},
        {"nested_splits": [("spatial", "spatial")], "cell_deg": 20.0},
        {"nested_splits": [("temporal_checkerboard",) * 2], "time_units": [("month", 2)]},
        {"nested_splits": [("temporal_contiguous_block",) * 2], "time_units": [("month", 6)]},
        {"nested_splits": [("temporal_forecast",) * 2], "test_size": 2},
    ],
)
def test_nested_folds_are_disjoint_and_keep_groups_intact(evaluation: dict[str, Any]) -> None:
    bench = _timestamped_bench()
    ((name, _),) = evaluation["nested_splits"]
    cfg = CoordEvaluationConfig(folds=3, inner_folds=3, **evaluation)
    (folds,) = make_folds(cfg, bench, np.arange(len(bench.lat)), seed=0).values()
    if name == "spatial":
        groups = spatial_cell_ids(bench.lat, bench.lon, cfg.cell_deg)
    elif name != "random":
        groups = time_unit_ids(bench.posix_timestamp, "month")
    for fold in folds:
        assert not np.intersect1d(fold.train, fold.test).size
        if name != "random":
            assert not np.intersect1d(groups[fold.train], groups[fold.test]).size
        for train, val in fold.inner:
            # Tuning must never see the outer test rows.
            assert np.isin(train, fold.train).all()
            assert np.isin(val, fold.train).all()


@pytest.mark.parametrize(
    ("unit", "start", "end", "expected"),
    [
        # 2017-01-01 is a Sunday: weeks counted from it keep Jan 1-7 together.
        (("week", 1), "2017-01-01", "2017-01-08", [0] * 7 + [1]),
        # Odd-sized blocks start at the first month, not at a fixed calendar offset.
        (("month", 3), "2017-02-01", "2017-05-01", [0, 0, 0, 1]),
    ],
)
def test_time_units_start_at_the_first_timestamp(
    unit: tuple[str, int], start: str, end: str, expected: list[int]
) -> None:
    freq = "D" if unit[0] == "week" else "MS"
    times = pd.date_range(start, end, freq=freq, tz="UTC")
    seconds = (times - pd.Timestamp(0, tz="UTC")).total_seconds().to_numpy()
    np.testing.assert_array_equal(time_unit_ids(seconds, *unit), expected)


@pytest.mark.parametrize(("max_train_size", "test_size"), [(None, 2), (6, 1)])
def test_temporal_forecast_trains_on_the_past(
    max_train_size: int | None, test_size: int | None
) -> None:
    bench = _timestamped_bench()
    split = "temporal_forecast" if max_train_size is None else "temporal_forecast_block"
    cfg = CoordEvaluationConfig(
        nested_splits=[(split, split)], max_train_size=max_train_size,
        test_size=test_size, folds=3, inner_folds=2,
    )  # fmt: skip
    months = time_unit_ids(bench.posix_timestamp, "month")
    (folds,) = make_folds(cfg, bench, np.arange(len(bench.lat)), seed=0).values()
    for fold in folds:
        assert bench.posix_timestamp[fold.train].max() < bench.posix_timestamp[fold.test].min()
        for train, val in fold.inner:
            assert bench.posix_timestamp[train].max() < bench.posix_timestamp[val].min()
        if max_train_size is not None:
            n_train, n_test = (len(np.unique(months[i])) for i in (fold.train, fold.test))
            assert n_test <= n_train <= max_train_size
        if test_size is not None:
            assert len(np.unique(months[fold.test])) == test_size


def test_split_variants_repeat_only_temporal_pairs_per_time_unit() -> None:
    cfg = CoordEvaluationConfig(
        nested_splits=[("random", "random"), ("temporal_checkerboard", "random")],
        time_units=[("month", 1), ("year", 1)], test_size=1, folds=2, inner_folds=2,
    )  # fmt: skip
    folds = make_folds(cfg, _timestamped_bench(), np.arange(400), seed=0)
    assert set(folds) == {
        ("random", "random", ("month", 1)),
        ("temporal_checkerboard", "random", ("month", 1)),
        ("temporal_checkerboard", "random", ("year", 1)),
    }


def test_official_split_tests_exactly_the_mask(points: tuple[np.ndarray, np.ndarray]) -> None:
    lon, lat = points
    bench = CoordBenchmark("synthetic", lat, lon, test_mask=np.arange(len(lat)) % 4 == 0)
    cfg = CoordEvaluationConfig(split_test_type="official")
    ((fold,),) = make_folds(cfg, bench, np.arange(len(lat)), seed=0).values()
    np.testing.assert_array_equal(fold.test, np.flatnonzero(bench.test_mask))
    assert len(fold.inner) == cfg.inner_folds


def _synthetic_benchmarks() -> list[CoordBenchmark]:
    rng = np.random.default_rng(1)
    lon = rng.uniform(-180, 180, size=300)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, size=300)))
    reg = CoordBenchmark(
        name="synthetic-reg",
        lat=lat,
        lon=lon,
        tasks={"target": np.sin(np.deg2rad(lat)) + 0.1 * rng.standard_normal(300)},
        task_type="regression",
    )
    clf = CoordBenchmark(
        name="synthetic-clf",
        lat=lat,
        lon=lon,
        tasks={"label": (lat > 0).astype(np.int64)},
        task_type="classification",
    )
    return [reg, clf]


def _coord_cfg(tmp_path: Path, **coord_overrides: object) -> CoordConfig:
    coord = {
        "methods": ["knn", "linear"],
        "folds": 5,
        "cell_deg": 10.0,
        "knn_ks": [5],
        "skip_no_timestamp": False,
        "exclude_antarctica": False,
    }
    coord.update(coord_overrides)
    return CoordConfig.model_validate(
        {
            "runtime": {"seed": 0, "device": "cpu"},
            "output": {"file": str(tmp_path / "coord.csv"), "resume": False},
            "model": {"name": "sincos"},
            "evaluation": coord,
        }
    )


@pytest.mark.parametrize("explicit_file", [False, True])
def test_run_coordbench_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, explicit_file: bool
) -> None:
    monkeypatch.setattr(
        "torchgeo_bench.coordbench.run.load_benchmarks", lambda *args: _synthetic_benchmarks()
    )
    cfg = _coord_cfg(tmp_path, nested_splits=[("spatial", "random")])
    cfg.output.directory = str(tmp_path / "nested")
    output_path = tmp_path / "coord.csv"
    if not explicit_file:
        cfg.output.file = None
        output_path = tmp_path / "nested" / "coordbench_results.csv"
    run_coordbench(cfg)

    assert set(tmp_path.rglob("*.csv")) == {output_path}
    df = pd.read_csv(output_path)
    assert {"dataset", "task", "method", "split", "metric_name", "metric_value"} <= set(df.columns)
    reg = df[df.dataset == "synthetic-reg"]
    clf = df[df.dataset == "synthetic-clf"]
    assert set(reg.method) == {"linear"}
    assert set(reg.metric_name) == {"r2"}
    assert set(clf.method) == {"linear", "knn"}
    assert set(clf.metric_name) == {"accuracy"}
    assert set(df.split) == {"spatial"}
    assert set(df.inner_split) == {"random"}
    assert (df.metric_value.abs() <= 1.5).all()


def test_run_coordbench_reports_official_test_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = _synthetic_benchmarks()[0]
    test_mask = np.zeros(len(bench.lat), dtype=bool)
    test_mask[::3] = True
    bench.test_mask = test_mask
    monkeypatch.setattr("torchgeo_bench.coordbench.run.load_benchmarks", lambda *args: [bench])

    cfg = _coord_cfg(tmp_path, split_test_type="official")
    run_coordbench(cfg)

    df = pd.read_csv(cfg.output.file)
    assert set(df.split_test_type) == {"official"}
    assert set(df.n_folds) == {1}
    assert set(df.n_test) == {int(test_mask.sum())}


@pytest.mark.parametrize(
    "evaluation",
    [
        {"split_test_type": "official"},
        {"nested_splits": [("temporal_contiguous_block", "random")]},
    ],
)
def test_run_coordbench_skips_benchmarks_missing_split_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, evaluation: dict[str, Any]
) -> None:
    # The synthetic benchmarks have neither a test mask nor timestamps.
    monkeypatch.setattr(
        "torchgeo_bench.coordbench.run.load_benchmarks", lambda *args: _synthetic_benchmarks()
    )
    cfg = _coord_cfg(tmp_path, **evaluation)
    run_coordbench(cfg)
    assert not Path(cfg.output.file).exists()


def test_coord_config_requires_window_for_forecast_block(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="requires max_train_size"):
        _coord_cfg(tmp_path, nested_splits=[("temporal_forecast_block", "random")])


def test_run_coordbench_resume_keeps_distinct_split_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "torchgeo_bench.coordbench.run.load_benchmarks", lambda *args: [_synthetic_benchmarks()[0]]
    )
    for cell_deg in (5.0, 20.0):
        cfg = _coord_cfg(tmp_path, nested_splits=[("spatial", "random")], cell_deg=cell_deg)
        cfg.output.resume = True
        run_coordbench(cfg)

    # The cell_deg=20.0 run must not be skipped as already completed.
    df = pd.read_csv(cfg.output.file)
    assert df.split_params.nunique() == 2
    # test_size only shapes temporal folds, so spatial rows keep the key they had before it existed.
    assert not df.split_params.str.contains("test_size").any()


def test_run_coordbench_encodes_once_for_all_nested_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "torchgeo_bench.coordbench.run.load_benchmarks", lambda *args: [_synthetic_benchmarks()[0]]
    )
    encodes = []

    def counting_encoder(preset: ModelPreset, device: str) -> SinCosLocationEncoder:
        encoder = _instantiate_encoder(preset, device)
        encode = encoder.encode

        def counted(*args: Any) -> torch.Tensor:
            encodes.append(args)
            return encode(*args)

        encoder.encode = counted
        return encoder

    monkeypatch.setattr("torchgeo_bench.coordbench.run._instantiate_encoder", counting_encoder)
    pairs = [("random", "random"), ("spatial", "random"), ("spatial", "spatial")]
    cfg = _coord_cfg(tmp_path, nested_splits=pairs)
    cfg.output.resume = True
    run_coordbench(cfg)

    assert len(encodes) == 1
    df = pd.read_csv(cfg.output.file)
    assert set(zip(df.split, df.inner_split, strict=True)) == set(pairs)

    # A rerun finds every (split, method) row already written and does not encode again.
    run_coordbench(cfg)
    assert len(encodes) == 1
    assert len(pd.read_csv(cfg.output.file)) == len(df)


def test_load_benchmarks_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    # Use local tables instead of downloading Parquet files.
    tables = {
        "country": pd.DataFrame(
            {"lon": [0.0, 1.0, 2.0], "lat": [0.0, 1.0, 2.0], "country": [1, 2, 1]}
        ),
        "worldclim_bio": pd.DataFrame(
            {"lon": [0.0, 1.0], "lat": [0.0, 1.0], "bio1": [10.0, 20.0], "bio12": [1.0, 2.0]}
        ),
    }
    monkeypatch.setattr(cb_datasets, "load_config", lambda cfg: tables[cfg])

    only_country = load_benchmarks("country")
    assert [b.name for b in only_country] == ["country"]

    wc = load_benchmarks("worldclim")
    assert {b.name for b in wc} == {"worldclim-bio1", "worldclim-bio12"}

    single = load_benchmarks("worldclim-bio1")
    assert [b.name for b in single] == ["worldclim-bio1"]

    np.testing.assert_array_equal(single[0].tasks["bio1"], [10.0, 20.0])


def test_mind_load_roundtrip(tmp_path: Path) -> None:
    from safetensors.torch import save_file

    from torchgeo_bench.coordbench.mind import ReSIRENLocationEncoder, load_mind

    model = ReSIRENLocationEncoder(embed_dim=16, out_dim=8, depth=2).eval()
    path = tmp_path / "m.safetensors"
    save_file(model.state_dict(), str(path))

    loaded = load_mind(str(path))
    assert loaded.embed_dim == 16
    assert len(loaded.blocks) == 2
    assert not loaded.use_year  # A two-input checkpoint has coordinates but no year.

    latlon = torch.tensor([[37.77, -122.42], [51.51, -0.13]], dtype=torch.float32)
    with torch.no_grad():
        torch.testing.assert_close(
            loaded(latlon, return_features=True), model(latlon, return_features=True)
        )
        torch.testing.assert_close(loaded(latlon), model(latlon))


def test_mind_encoder_dim_slice(monkeypatch: pytest.MonkeyPatch) -> None:
    from torchgeo_bench.coordbench import mind as mind_mod
    from torchgeo_bench.coordbench.models import MINDLocationEncoder

    model = mind_mod.ReSIRENLocationEncoder(embed_dim=32, out_dim=32, depth=2).eval()
    monkeypatch.setattr(mind_mod, "load_mind", lambda path, device="cpu": model)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **k: "dummy")

    enc = MINDLocationEncoder(dim=8, feature="pooled", device="cpu")
    out = enc.encode(np.array([1.0, 2.0]), np.array([37.0, 51.0]))
    assert out.shape == (2, 8)  # MIND's training permits using a prefix of the 32 features.
    assert out.dtype == np.float32
    with torch.no_grad():
        expected = model(torch.tensor([[37.0, 1.0], [51.0, 2.0]]), return_features=True)
    np.testing.assert_allclose(out, expected[:, :8].numpy())


def test_family_index_matches_loaders() -> None:
    # Static family indexes allow listing and selection without downloads.
    assert set(cb_datasets.FAMILY_BENCHMARKS) == set(cb_datasets.FAMILY_LOADERS)
    all_names = cb_datasets.list_benchmarks()
    assert len(all_names) == len(set(all_names))
    assert set(all_names) == {
        name for family_names in cb_datasets.FAMILY_BENCHMARKS.values() for name in family_names
    }
    assert "pdfm-conus27" in all_names
    assert sum(n.startswith("dm-") for n in all_names) == 15


@pytest.mark.parametrize("methods", [["linear"], ["knn"], ["knn", "linear"]])
def test_runtime_method_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, methods: list[str]
) -> None:
    monkeypatch.setattr(
        "torchgeo_bench.coordbench.run.load_benchmarks", lambda *args: _synthetic_benchmarks()
    )
    grids = []

    def fake_evaluate(predict: Any, grid: list, folds: list, data: ProbeData) -> ProbeResult:
        grids.append(list(grid))
        return ProbeResult(0.8, [0.7, 0.9], [grid[0]] * 2, np.full(len(data.y), np.nan))

    monkeypatch.setattr("torchgeo_bench.coordbench.run.evaluate", fake_evaluate)
    config = _coord_cfg(
        tmp_path,
        methods=methods,
        knn_ks=[7],
        knn_device="cpu",
        ridge_alphas=[0.1, 2.0],
    )
    run_coordbench(config)
    rows = pd.read_csv(config.output.file)
    assert set(rows.method) == set(methods)
    expected_grids = {"knn": [7], "linear": [0.1, 2.0]}
    assert {tuple(grid) for grid in grids} == {tuple(expected_grids[m]) for m in methods}
    assert set(rows.model_target) == {"torchgeo_bench.coordbench.models.SinCosLocationEncoder"}
    assert set(rows.model_name) == {"sincos"}
    assert list(rows.columns) == [
        "dataset",
        "task",
        "task_type",
        "method",
        "split_test_type",
        "split",
        "inner_split",
        "split_params",
        "metric_name",
        "metric_value",
        "ci_lower",
        "ci_upper",
        "n_folds",
        "best_params",
        "feature_dim",
        "n_samples",
        "n_test",
        "seed",
        "model_name",
        "model_target",
        "temporal_embedding_aggregation",
    ]


def test_resume_skips_encoding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "torchgeo_bench.coordbench.run.load_benchmarks", lambda *args: _synthetic_benchmarks()
    )
    config = _coord_cfg(tmp_path)
    run_coordbench(config)
    before = Path(config.output.file).read_bytes()

    def unexpected_encode(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Completed benchmarks must not be encoded again")

    monkeypatch.setattr(SinCosLocationEncoder, "encode", unexpected_encode)
    config.output.resume = True
    run_coordbench(config)
    assert Path(config.output.file).read_bytes() == before


def test_partial_results_survive_failure_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = _synthetic_benchmarks()[0]
    bench.tasks["second"] = bench.tasks["target"].copy()
    monkeypatch.setattr("torchgeo_bench.coordbench.run.load_benchmarks", lambda *args: [bench])
    calls = []

    def failing_evaluate(predict: Any, grid: list, folds: list, data: ProbeData) -> ProbeResult:
        calls.append("linear")
        if len(calls) == 2:
            raise RuntimeError("second task failed")
        return ProbeResult(0.8, [0.8], [grid[0]], np.full(len(data.y), np.nan))

    monkeypatch.setattr("torchgeo_bench.coordbench.run.evaluate", failing_evaluate)
    config = _coord_cfg(tmp_path, methods=["linear"])
    with pytest.raises(RuntimeError, match="second task failed"):
        run_coordbench(config)
    assert pd.read_csv(config.output.file).task.tolist() == ["target"]
    config.output.resume = True
    run_coordbench(config)
    assert pd.read_csv(config.output.file).task.tolist() == ["target", "second"]
    assert len(calls) == 3


def test_knn_only_skips_regression(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "torchgeo_bench.coordbench.run.load_benchmarks",
        lambda *args: [_synthetic_benchmarks()[0]],
    )

    def unexpected_encode(*args: Any, **kwargs: Any) -> None:
        pytest.fail("KNN-only regression must not encode or probe")

    monkeypatch.setattr(SinCosLocationEncoder, "encode", unexpected_encode)
    config = _coord_cfg(tmp_path, methods=["knn"])
    run_coordbench(config)
    assert not Path(config.output.file).exists()


class FixtureEncoder(SinCosLocationEncoder):
    def __init__(self, payload: dict[str, Any], device: str) -> None:
        super().__init__(device=device)
        self.payload = payload


def test_custom_encoder_preserves_ordinary_nested_kwargs() -> None:
    payload = {"_target_": "not_imported.Missing", "options": {"enabled": True}}
    preset = ModelPreset(
        name="my-encoder",
        target=f"{__name__}.FixtureEncoder",
        kwargs={"payload": payload},
    )
    encoder = _instantiate_encoder(preset, "cpu")
    assert isinstance(encoder, FixtureEncoder)
    assert encoder.payload == payload
    assert encoder.device == "cpu"


def test_rejects_non_location_custom_target() -> None:
    preset = ModelPreset(name="not-an-encoder", target="builtins.dict")
    with pytest.raises(TypeError, match="LocationEncoder"):
        _instantiate_encoder(preset, "cpu")


@pytest.mark.parametrize(("available", "expected"), [(False, "cpu"), (True, "cuda:1")])
def test_auto_device_resolves_for_encoder_and_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, available: bool, expected: str
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        "torchgeo_bench.coordbench.run.load_benchmarks",
        lambda *args: [_synthetic_benchmarks()[0]],
    )
    devices = []

    def ridge(data: ProbeData, device: str) -> Any:
        devices.append(device)
        return ridge_predictor(data, "cpu")

    monkeypatch.setattr("torchgeo_bench.coordbench.run.ridge_predictor", ridge)
    config = _coord_cfg(tmp_path)
    config.runtime.device = "auto"
    with mock.patch(
        "torchgeo_bench.coordbench.run._instantiate_encoder", wraps=_instantiate_encoder
    ) as build:
        run_coordbench(config)
    assert build.call_args.args[1] == expected
    assert devices == [expected]
    assert config.runtime.device == "auto"
    assert pd.read_csv(config.output.file)["metric_value"].iloc[0] > 0.9


@pytest.mark.parametrize(
    "device_case",
    [("cuda", False, "CUDA is unavailable"), ("cuda:2", True, "CUDA index 2")],
)
@pytest.mark.parametrize("entrypoint", ["direct", "cli"])
def test_coordinate_run_rejects_invalid_cuda_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
    device_case: tuple[str, bool, str],
) -> None:
    from torchgeo_bench.cli import main

    device, available, message = device_case
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    config = _coord_cfg(tmp_path)
    config.runtime.device = device
    with (
        mock.patch("torchgeo_bench.coordbench.run.load_benchmarks") as data,
        mock.patch("torchgeo_bench.coordbench.run.build_model") as build,
    ):
        if entrypoint == "direct":
            with pytest.raises(ValueError, match=message):
                run_coordbench(config)
        else:
            with pytest.raises(ValueError, match=message):
                main(
                    [
                        "coord",
                        "--model",
                        "sincos",
                        "--dataset",
                        "california_housing",
                        "--device",
                        device,
                        "--output",
                        config.output.file,
                    ]
                )
    data.assert_not_called()
    build.assert_not_called()
    assert not Path(config.output.file).exists()


def test_runtime_seeds_encoder_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("torchgeo_bench.coordbench.run.load_benchmarks", lambda *args: [])
    random_values = []

    def build(preset: ModelPreset, *, device: str) -> SinCosLocationEncoder:
        random_values.append(torch.rand(4))
        return SinCosLocationEncoder(device=device)

    monkeypatch.setattr("torchgeo_bench.coordbench.run.build_model", build)
    config = _coord_cfg(tmp_path)
    for seed in (7, 7, 11):
        config.runtime.seed = seed
        run_coordbench(config)
        expected = torch.rand(4, generator=torch.Generator().manual_seed(seed))
        assert torch.equal(random_values[-1], expected)
