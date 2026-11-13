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
    idw_predictor,
    knn_predictor,
    ridge_predictor,
)
from torchgeo_bench.coordbench.run import _instantiate_encoder
from torchgeo_bench.coordbench.splits import (
    make_folds,
    spatial_cell_ids,
    split_variants,
    time_unit_ids,
)

pytestmark = pytest.mark.usefixtures("isolated_torch_rng")
RUN = "torchgeo_bench.coordbench.run"


def _probe_data(
    features: np.ndarray, labels: np.ndarray, task_type: str, **coords: np.ndarray
) -> ProbeData:
    labels = labels.reshape(len(labels), -1)
    if task_type == "regression":
        return ProbeData(features, labels.astype(np.float64), task_type, None, **coords)
    return ProbeData(features, labels, task_type, labels, **coords)


def _folds(bench: CoordBenchmark, **evaluation: Any) -> list:
    cfg = CoordEvaluationConfig(**evaluation)
    return make_folds(cfg, bench, np.arange(len(bench.lat)), 0, split_variants(cfg)[0])


def _mock_cuda(monkeypatch: pytest.MonkeyPatch, *, available: bool) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)


@pytest.fixture
def points() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    return rng.uniform(-180, 180, size=400), np.degrees(np.arcsin(rng.uniform(-1, 1, size=400)))


def test_sincos_encode_preserves_coordinate_order_across_batches(
    points: tuple[np.ndarray, np.ndarray],
) -> None:
    lon, lat = points
    feats = SinCosLocationEncoder(device="cpu", batch_size=137).encode(lon, lat)
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
    _mock_cuda(monkeypatch, available=True)
    encoder = SinCosLocationEncoder(device=device)
    assert encoder.device == expected
    np.testing.assert_array_equal(encoder.encode(*points), SinCosLocationEncoder().encode(*points))


@pytest.mark.parametrize(
    ("device", "available", "message"),
    [("cuda", False, "CUDA is unavailable"), ("cuda:2", True, "CUDA index 2")],
)
def test_encoder_probe_and_run_reject_invalid_devices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device: str, *, available: bool, message: str
) -> None:
    _mock_cuda(monkeypatch, available=available)
    with pytest.raises(ValueError, match=message):
        SinCosLocationEncoder(device=device)
    with (
        mock.patch.object(torch, "as_tensor") as allocate,
        pytest.raises(ValueError, match=message),
    ):
        ridge_predictor(_probe_data(np.ones((10, 2)), np.arange(10), "regression"), device)
    allocate.assert_not_called()
    config = _coord_cfg(tmp_path)
    config.runtime.device = device
    with (
        mock.patch(f"{RUN}.load_benchmarks") as data,
        mock.patch(f"{RUN}.build_model") as build,
        pytest.raises(ValueError, match=message),
    ):
        run_coordbench(config)
    data.assert_not_called()
    build.assert_not_called()


def test_coordinate_cpu_paths_do_not_query_cuda(
    points: tuple[np.ndarray, np.ndarray], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        torch.cuda, "is_available", lambda: pytest.fail("CPU paths must not query CUDA")
    )
    lon, lat = points
    features = SinCosLocationEncoder().encode(lon, lat)
    data = _probe_data(features, features[:, 0], "regression")
    (result,) = evaluate(
        ridge_predictor(data, "cpu"), [0.1], _folds(CoordBenchmark("synthetic", lat, lon)), data
    )
    assert result.test_score > 0.99


def test_documented_fourier_encoder_example(points: tuple[np.ndarray, np.ndarray]) -> None:
    from docs.examples.coordbench_location_encoder import FourierLocationEncoder

    feats = FourierLocationEncoder(num_frequencies=4).encode(*points)
    assert feats.shape == (400, 16)
    assert feats.dtype == np.float32
    assert np.isfinite(feats).all()


@pytest.mark.filterwarnings("ignore:.*grid edge")
@pytest.mark.parametrize("task_type", ["regression", "classification"])
def test_tasks_probed_together_match_each_task_alone(
    points: tuple[np.ndarray, np.ndarray], task_type: str
) -> None:
    lon, lat = points
    feats = SinCosLocationEncoder(device="cpu").encode(lon, lat)
    labels = np.column_stack([feats[:, 0] - 2 * feats[:, 3], feats[:, 1] * feats[:, 2]])
    if task_type == "classification":
        labels = (labels > np.median(labels, axis=0)).astype(np.int64)
    probes = [
        (lambda d: ridge_predictor(d, "cpu"), [0.01, 1.0, 100.0]),
        (lambda d: idw_predictor(d, "cpu"), [(1, 0.0), (5, 0.0), (20, 0.0)]),
    ]
    if task_type == "classification":
        probes.append((lambda d: knn_predictor(d, "cpu"), [1, 5, 15]))
    folds = _folds(CoordBenchmark("synthetic", lat, lon))
    data = _probe_data(feats, labels, task_type, lat=lat, lon=lon)
    for predictor, grid in probes:
        for i, result in enumerate(evaluate(predictor(data), grid, folds, data)):
            alone = _probe_data(feats, labels[:, i], task_type, lat=lat, lon=lon)
            (expected,) = evaluate(predictor(alone), grid, folds, alone)
            assert result.best_params == expected.best_params
            np.testing.assert_allclose(result.fold_scores, expected.fold_scores, rtol=1e-6)
            np.testing.assert_allclose(result.predictions, expected.predictions, rtol=1e-5)


def _timestamped_bench(n: int = 400) -> CoordBenchmark:
    rng = np.random.default_rng(0)
    lat, lon = rng.uniform(-60, 60, n), rng.uniform(-180, 180, n)
    return CoordBenchmark(
        "synthetic", lat, lon, posix_timestamp=rng.uniform(1.5e9, 1.5e9 + 6 * 365 * 86400, n)
    )


@pytest.mark.parametrize(
    "evaluation",
    [
        {"nested_splits": [("random", "random")]},
        {"nested_splits": [("spatial", "spatial")], "cell_deg": 20.0},
        {"nested_splits": [("temporal_checkerboard",) * 2], "time_units": [("month", 2)]},
        {"nested_splits": [("temporal_contiguous_block",) * 2], "time_units": [("month", 6)]},
        {
            "nested_splits": [("temporal_forecast",) * 2],
            "time_units": [("month", 1)],
            "test_size": 2,
        },
    ],
)
def test_nested_folds_are_disjoint_and_keep_groups_intact(evaluation: dict[str, Any]) -> None:
    bench = _timestamped_bench()
    ((name, _),) = evaluation["nested_splits"]
    if name == "spatial":
        groups = spatial_cell_ids(bench.lat, bench.lon, 20.0)
    elif name != "random":
        groups = time_unit_ids(bench.posix_timestamp, "month")
    for fold in _folds(bench, folds=3, inner_folds=3, **evaluation):
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
        (("quarter", 1), "2017-02-01", "2017-05-01", [0, 0, 1, 1]),
        # December opens the next meteorological year's winter.
        (("season", 1), "2016-11-01", "2017-03-01", [0, 1, 1, 1, 2]),
    ],
)
def test_time_units_start_at_the_first_timestamp(
    unit: tuple[str, int], start: str, end: str, expected: list[int]
) -> None:
    times = pd.date_range(start, end, freq="D" if unit[0] == "week" else "MS", tz="UTC")
    seconds = (times - pd.Timestamp(0, tz="UTC")).total_seconds().to_numpy()
    np.testing.assert_array_equal(time_unit_ids(seconds, *unit), expected)


@pytest.mark.parametrize(("max_train_size", "test_size"), [(None, 2), (6, 1)])
def test_temporal_forecast_trains_on_the_past(max_train_size: int | None, test_size: int) -> None:
    bench = _timestamped_bench()
    split = "temporal_forecast" if max_train_size is None else "temporal_forecast_block"
    folds = _folds(
        bench, nested_splits=[(split, split)], time_units=[("month", 1)], max_train_size=max_train_size,
        test_size=test_size, folds=3, inner_folds=2,
    )  # fmt: skip
    times, months = bench.posix_timestamp, time_unit_ids(bench.posix_timestamp, "month")
    for fold in folds:
        assert times[fold.train].max() < times[fold.test].min()
        for train, val in fold.inner:
            assert times[train].max() < times[val].min()
        n_train, n_test = (len(np.unique(months[i])) for i in (fold.train, fold.test))
        assert n_test == test_size
        if max_train_size is not None:
            assert n_test <= n_train <= max_train_size


def test_split_variants_repeat_only_temporal_pairs_per_time_unit() -> None:
    cfg = CoordEvaluationConfig(
        nested_splits=[("random", "random"), ("temporal_checkerboard", "random")],
        time_units=[("month", 1), ("year", 1)],
    )
    assert set(split_variants(cfg)) == {
        ("random", "random", ("month", 1)),
        ("temporal_checkerboard", "random", ("month", 1)),
        ("temporal_checkerboard", "random", ("year", 1)),
    }


def test_official_split_tests_exactly_the_mask(points: tuple[np.ndarray, np.ndarray]) -> None:
    lon, lat = points
    bench = CoordBenchmark("synthetic", lat, lon, test_mask=np.arange(len(lat)) % 4 == 0)
    (fold,) = _folds(bench, split_test_type="official")
    np.testing.assert_array_equal(fold.test, np.flatnonzero(bench.test_mask))
    assert len(fold.inner) == 5


def _synthetic_benchmarks() -> list[CoordBenchmark]:
    rng = np.random.default_rng(1)
    lon, lat = rng.uniform(-180, 180, size=300), np.degrees(np.arcsin(rng.uniform(-1, 1, size=300)))
    target = np.sin(np.deg2rad(lat)) + 0.1 * rng.standard_normal(300)
    return [
        CoordBenchmark("synthetic-reg", lat, lon, tasks={"target": target}, task_type="regression"),
        CoordBenchmark(
            "synthetic-clf",
            lat,
            lon,
            tasks={"label": (lat > 0).astype(np.int64)},
            task_type="classification",
        ),
    ]


@pytest.fixture
def benches(monkeypatch: pytest.MonkeyPatch) -> list[CoordBenchmark]:
    benchmarks = _synthetic_benchmarks()
    monkeypatch.setattr(f"{RUN}.load_benchmarks", lambda *args: benchmarks)
    return benchmarks


def _coord_cfg(tmp_path: Path, **evaluation: object) -> CoordConfig:
    return CoordConfig.model_validate({
        "runtime": {"seed": 0, "device": "cpu"},
        "output": {"file": str(tmp_path / "coord.csv")},
        "model": {"name": "sincos"},
        "evaluation": {"knn_device": "cpu", "exclude_antarctica": False, **evaluation},
    })  # fmt: skip


@pytest.mark.parametrize("explicit_file", [False, True])
def test_run_coordbench_end_to_end(tmp_path: Path, benches: list, *, explicit_file: bool) -> None:
    cfg = _coord_cfg(tmp_path, nested_splits=[("spatial", "random")])
    cfg.output.directory = str(tmp_path / "nested")
    output_path = tmp_path / "coord.csv"
    if not explicit_file:
        cfg.output.file = None
        output_path = tmp_path / "nested" / "coordbench_results.csv"
    run_coordbench(cfg)
    assert set(tmp_path.rglob("*.csv")) == {output_path}
    df = pd.read_csv(output_path)
    reg, clf = df[df.dataset == "synthetic-reg"], df[df.dataset == "synthetic-clf"]
    assert (set(reg.method), set(reg.metric_name)) == ({"linear"}, {"r2"})
    assert (set(clf.method), set(clf.metric_name)) == ({"linear", "knn"}, {"accuracy"})
    assert (set(df.split), set(df.inner_split)) == ({"spatial"}, {"random"})
    assert (df.metric_value > 0.9).all()


@pytest.mark.parametrize(
    "evaluation",
    [
        {"split_test_type": "official"},
        {"nested_splits": [("temporal_contiguous_block", "random")], "time_units": [("month", 1)]},
    ],
)
def test_run_coordbench_skips_benchmarks_missing_split_inputs(
    tmp_path: Path, benches: list, evaluation: dict[str, Any]
) -> None:
    # The synthetic benchmarks have neither a test mask nor timestamps.
    cfg = _coord_cfg(tmp_path, **evaluation)
    run_coordbench(cfg)
    assert not Path(cfg.output.file).exists()


def test_run_coordbench_resume_keeps_distinct_split_params(tmp_path: Path, benches: list) -> None:
    del benches[1]
    for cell_deg in (5.0, 20.0):
        cfg = _coord_cfg(tmp_path, nested_splits=[("spatial", "random")], cell_deg=cell_deg)
        cfg.output.resume = True
        run_coordbench(cfg)
    df = pd.read_csv(cfg.output.file)
    # The cell_deg=20.0 run must not be skipped as already completed.
    assert df.split_params.nunique() == 2
    # test_size only shapes temporal folds, so spatial rows keep the key they had before it existed.
    assert not df.split_params.str.contains("test_size").any()


def test_run_coordbench_encodes_once_and_resume_skips_encoding(
    tmp_path: Path, benches: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    del benches[1]
    encodes = []
    original_encode = SinCosLocationEncoder.encode
    monkeypatch.setattr(
        SinCosLocationEncoder,
        "encode",
        lambda self, *a: encodes.append(a) or original_encode(self, *a),
    )
    pairs = [("random", "random"), ("spatial", "random"), ("spatial", "spatial")]
    cfg = _coord_cfg(tmp_path, nested_splits=pairs)
    cfg.output.resume = True
    run_coordbench(cfg)
    assert len(encodes) == 1
    before = Path(cfg.output.file).read_bytes()
    df = pd.read_csv(cfg.output.file)
    assert set(zip(df.split, df.inner_split, strict=True)) == set(pairs)
    run_coordbench(cfg)
    assert len(encodes) == 1
    assert Path(cfg.output.file).read_bytes() == before


def test_idw_runs_without_a_model(tmp_path: Path, benches: list) -> None:
    cfg = CoordConfig.model_validate({**_coord_cfg(tmp_path).model_dump(), "model": None,
                                      "evaluation": {"methods": ["idw"], "knn_device": "cpu"}})  # fmt: skip
    with mock.patch(f"{RUN}.build_model") as build:
        run_coordbench(cfg)
    build.assert_not_called()
    df = pd.read_csv(cfg.output.file)
    assert (set(df.method), set(df.model_name), set(df.feature_dim)) == ({"idw"}, {"idw"}, {0})
    assert (df.metric_value > 0.5).all()


@pytest.mark.parametrize(
    "evaluation",
    [{"methods": ["linear", "idw"]}, {"methods": ["idw"], "temporal_sampling_methods": ["year"]}],
)
def test_coord_config_requires_model_for_embeddings(evaluation: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="model is required"):
        CoordConfig.model_validate({"evaluation": evaluation})


def test_load_benchmarks_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    tables = {
        "country": pd.DataFrame(
            {"lon": [0.0, 1.0, 2.0], "lat": [0.0, 1.0, 2.0], "country": [1, 2, 1]}
        ),
        "worldclim_bio": pd.DataFrame(
            {"lon": [0.0, 1.0], "lat": [0.0, 1.0], "bio1": [10.0, 20.0], "bio12": [1.0, 2.0]}
        ),
    }
    monkeypatch.setattr(cb_datasets, "load_config", lambda cfg: tables[cfg])
    assert [b.name for b in load_benchmarks("country")] == ["country"]
    assert {b.name for b in load_benchmarks("worldclim")} == {"worldclim-bio1", "worldclim-bio12"}
    (single,) = load_benchmarks("worldclim-bio1")
    assert single.name == "worldclim-bio1"
    np.testing.assert_array_equal(single.tasks["bio1"], [10.0, 20.0])


def test_family_index_matches_loaders() -> None:
    # Static family indexes allow listing and selection without downloads.
    assert set(cb_datasets.FAMILY_BENCHMARKS) == set(cb_datasets.FAMILY_LOADERS)
    all_names = cb_datasets.list_benchmarks()
    assert sorted(all_names) == sorted(
        n for names in cb_datasets.FAMILY_BENCHMARKS.values() for n in names
    )
    assert "pdfm-conus27" in all_names
    assert sum(n.startswith("dm-") for n in all_names) == 15


def test_mind_load_roundtrip(tmp_path: Path) -> None:
    from safetensors.torch import save_file

    from torchgeo_bench.coordbench.mind import ReSIRENLocationEncoder, load_mind

    model = ReSIRENLocationEncoder(embed_dim=16, out_dim=8, depth=2).eval()
    save_file(model.state_dict(), str(tmp_path / "m.safetensors"))
    loaded = load_mind(str(tmp_path / "m.safetensors"))
    assert (loaded.embed_dim, len(loaded.blocks)) == (16, 2)
    assert not loaded.use_year  # A two-input checkpoint has coordinates but no year.
    latlon = torch.tensor([[37.77, -122.42], [51.51, -0.13]])
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
    out = MINDLocationEncoder(dim=8, feature="pooled", device="cpu").encode(
        np.array([1.0, 2.0]), np.array([37.0, 51.0])
    )
    assert out.dtype == np.float32
    with torch.no_grad():
        expected = model(torch.tensor([[37.0, 1.0], [51.0, 2.0]]), return_features=True)
    # MIND's training permits using a prefix of the 32 features.
    np.testing.assert_allclose(out, expected[:, :8].numpy())


@pytest.mark.parametrize("methods", [["linear"], ["knn"]])
def test_runtime_method_selection(
    tmp_path: Path, benches: list, monkeypatch: pytest.MonkeyPatch, methods: list[str]
) -> None:
    grids = []

    def fake_evaluate(predict: Any, grid: list, folds: list, data: ProbeData) -> list[ProbeResult]:
        grids.append(list(grid))
        return [ProbeResult(0.8, [0.7, 0.9], [grid[0]] * 2, np.full(len(data.y), np.nan))]

    monkeypatch.setattr(f"{RUN}.evaluate", fake_evaluate)
    config = _coord_cfg(tmp_path, methods=methods, knn_ks=[7], ridge_alphas=[0.1, 2.0])
    run_coordbench(config)
    rows = pd.read_csv(config.output.file)
    assert set(rows.method) == set(methods)
    assert grids[0] == {"knn": [7], "linear": [0.1, 2.0]}[methods[0]]
    assert set(rows.model_target) == {"torchgeo_bench.coordbench.models.SinCosLocationEncoder"}
    assert list(rows.columns) == [
        "dataset", "task", "task_type", "method", "split_test_type", "split", "inner_split",
        "split_params", "metric_name", "metric_value", "ci_lower", "ci_upper", "n_folds",
        "best_params", "feature_dim", "n_samples", "n_test", "seed", "model_name", "model_target",
        "temporal_embedding_aggregation",
    ]  # fmt: skip


def test_partial_results_survive_failure_and_resume(
    tmp_path: Path, benches: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    del benches[1]
    benches[0].tasks["second"] = benches[0].tasks["target"].copy()
    calls = []

    def failing_evaluate(predict: Any, grid: list, folds: list, data: ProbeData) -> list:
        calls.append(data.y.shape[1])
        if len(calls) == 2:
            raise RuntimeError("second split failed")
        return [ProbeResult(0.8, [0.8], [grid[0]], np.full(len(data.y), np.nan))] * data.y.shape[1]

    monkeypatch.setattr(f"{RUN}.evaluate", failing_evaluate)
    config = _coord_cfg(
        tmp_path, methods=["linear"], nested_splits=[("random", "random"), ("spatial", "random")]
    )
    with pytest.raises(RuntimeError, match="second split failed"):
        run_coordbench(config)
    rows = pd.read_csv(config.output.file)
    assert list(zip(rows.split, rows.task, strict=True)) == [("random", "target"), ("random", "second")]  # fmt: skip
    config.output.resume = True
    run_coordbench(config)
    assert pd.read_csv(config.output.file).split.tolist() == ["random"] * 2 + ["spatial"] * 2
    # Both tasks share each split's probe; resume reruns only the failed split.
    assert calls == [2, 2, 2]


def test_knn_only_skips_regression(
    tmp_path: Path, benches: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    del benches[1]
    monkeypatch.setattr(
        SinCosLocationEncoder,
        "encode",
        lambda *a: pytest.fail("KNN-only regression must not encode"),
    )
    config = _coord_cfg(tmp_path, methods=["knn"])
    run_coordbench(config)
    assert not Path(config.output.file).exists()


class FixtureEncoder(SinCosLocationEncoder):
    def __init__(self, payload: dict[str, Any], device: str) -> None:
        super().__init__(device=device)
        self.payload = payload


def test_custom_encoder_targets() -> None:
    payload = {"_target_": "not_imported.Missing", "options": {"enabled": True}}
    encoder = _instantiate_encoder(
        ModelPreset(name="mine", target=f"{__name__}.FixtureEncoder", kwargs={"payload": payload}),
        "cpu",
    )
    assert isinstance(encoder, FixtureEncoder)
    assert (encoder.payload, encoder.device) == (payload, "cpu")
    with pytest.raises(TypeError, match="LocationEncoder"):
        _instantiate_encoder(ModelPreset(name="not-an-encoder", target="builtins.dict"), "cpu")


@pytest.mark.parametrize(("available", "expected"), [(False, "cpu"), (True, "cuda:1")])
def test_auto_device_resolves_for_encoder_and_probe(
    tmp_path: Path,
    benches: list,
    monkeypatch: pytest.MonkeyPatch,
    *,
    available: bool,
    expected: str,
) -> None:
    del benches[1]
    _mock_cuda(monkeypatch, available=available)
    devices = []
    monkeypatch.setattr(
        f"{RUN}.ridge_predictor",
        lambda data, device: devices.append(device) or ridge_predictor(data, "cpu"),
    )
    config = _coord_cfg(tmp_path)
    config.runtime.device = "auto"
    with mock.patch(f"{RUN}._instantiate_encoder", wraps=_instantiate_encoder) as build:
        run_coordbench(config)
    assert build.call_args.args[1] == expected
    assert devices == [expected]
    assert config.runtime.device == "auto"
    assert pd.read_csv(config.output.file)["metric_value"].iloc[0] > 0.9


def test_runtime_seeds_encoder_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(f"{RUN}.load_benchmarks", lambda *args: [])
    random_values = []

    def build(preset: ModelPreset, *, device: str) -> SinCosLocationEncoder:
        random_values.append(torch.rand(4))
        return SinCosLocationEncoder(device=device)

    monkeypatch.setattr(f"{RUN}.build_model", build)
    config = _coord_cfg(tmp_path)
    for seed in (7, 11):
        config.runtime.seed = seed
        run_coordbench(config)
        assert torch.equal(
            random_values[-1], torch.rand(4, generator=torch.Generator().manual_seed(seed))
        )
