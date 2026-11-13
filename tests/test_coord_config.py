"""Strict coordinate configuration and lightweight command coverage."""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from torchgeo_bench.cli import main as cli_main
from torchgeo_bench.commands._coord import load_config, run
from torchgeo_bench.commands.coord_arguments import add_coord_arguments
from torchgeo_bench.coordbench.config import CoordConfig, load_coord_config, resolve_coord_preset


def _parse(command: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_coord_arguments(parser)
    return parser.parse_args(command.split())


def test_coord_defaults() -> None:
    config = CoordConfig.model_validate({"model": {"name": "sincos"}})
    assert config.datasets == ["all"]
    assert config.evaluation.knn_device == "cuda"
    assert config.output.file is None
    assert not config.output.resume
    preset = resolve_coord_preset(config)
    assert (preset.name, preset.track, preset.kwargs) == ("sincos", "coord", {})


@pytest.mark.parametrize(
    "values",
    [
        {"unexpected": True},
        {"schema_version": 2},
        {"model": {"name": "sincos", "extra": 3}},
        {"datasets": []},
        {"datasets": ["country", "country"]},
        {"datasets": ["all", "country"]},
        {"datasets": ["does-not-exist"]},
        {"evaluation": {"methods": ["ridge"]}},
        {"evaluation": {"methods": ["linear", "linear"]}},
        {"evaluation": {"nested_splits": [["official", "random"]]}},
        {"evaluation": {"nested_splits": [["random", "random"], ["random", "random"]]}},
        {"evaluation": {"split_test_type": "official", "nested_splits": [["random", "random"], ["spatial", "random"]]}},
        {"evaluation": {"nested_splits": [["temporal_checkerboard", "random"]]}},
        {"evaluation": {"nested_splits": [["temporal_forecast_block", "random"]], "time_units": [["month", 1]]}},
        {"evaluation": {"nested_splits": [["temporal_forecast_block", "random"]], "time_units": [["month", 1]], "max_train_size": 1, "test_size": 2}},
        {"evaluation": {"test_size": 2}},
        {"evaluation": {"time_units": [["month", 0]]}},
        {"evaluation": {"time_units": [["month", 1], ["month", 1]]}},
        {"evaluation": {"folds": "5"}},
        {"evaluation": {"folds": 1}},
        {"evaluation": {"cell_deg": 0}},
        {"evaluation": {"ridge_alphas": []}},
        {"evaluation": {"ridge_alphas": [0]}},
        {"evaluation": {"knn_device": "cuda:bad"}},
        {"runtime": {"device": "cuda:-1"}},
        {"runtime": {"seed": -1}},
        {"output": {"file": " "}},
        {"output": {"resume": "false"}},
    ],
)  # fmt: skip
def test_coord_rejects_invalid_values(values: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        CoordConfig.model_validate({"model": {"name": "sincos"}, **values})


@pytest.mark.parametrize("flag", ["--dataset", "--datasets", "--name", "--names"])
def test_dataset_flag_aliases(flag: str) -> None:
    config = load_config(_parse(f"--model sincos {flag} worldclim worldclim-bio1"))
    assert config.datasets == ["worldclim", "worldclim-bio1"]


def test_all_typed_flags() -> None:
    config = load_config(_parse(
        "--model sincos --dataset country --methods linear knn --split-test-type official "
        "--nested-splits temporal_forecast_block,spatial --folds 3 --inner-folds 4 --cell-deg 2.5 "
        "--time-units week,2 month,1 --reset-year --no-exclude-antarctica --max-train-size 6 "
        "--test-size 2 --ridge-alphas 0.1 1 10 --knn-ks 7 9 --knn-device cpu --idw-ks 5 20 "
        "--idw-cs 0 1e-12 --device cpu --seed 42 --output runs/coord.csv --resume"
    ))  # fmt: skip
    expected = {
        "methods": ["linear", "knn"], "split_test_type": "official",
        "nested_splits": [("temporal_forecast_block", "spatial")], "folds": 3, "inner_folds": 4,
        "cell_deg": 2.5, "time_units": [("week", 2), ("month", 1)], "reset_year": True,
        "exclude_antarctica": False, "max_train_size": 6, "test_size": 2,
        "ridge_alphas": [0.1, 1.0, 10.0], "knn_ks": [7, 9], "knn_device": "cpu",
        "idw_ks": [5, 20], "idw_cs": [0.0, 1e-12],
    }  # fmt: skip
    assert config.evaluation.model_dump(include=set(expected)) == expected
    assert config.runtime.model_dump() == {"device": "cpu", "seed": 42}
    assert config.output.model_dump() == {"directory": "results", "file": "runs/coord.csv", "resume": True}  # fmt: skip


def test_yaml_flag_precedence(tmp_path: Path) -> None:
    path = tmp_path / "coord.yaml"
    path.write_text(
        "model: {name: sincos, kwargs: {batch_size: 16}}\ndatasets: [country]\n"
        "evaluation: {methods: [linear], cell_deg: 1e1}\nruntime: {seed: 7}\n"
        "output: {file: custom.csv, resume: true}\n"
    )
    config = load_config(_parse(f"--config {path} --no-resume --seed 0"))
    assert config.model.kwargs == {"batch_size": 16}
    assert config.evaluation.methods == ["linear"]
    assert config.evaluation.cell_deg == 10.0
    assert config.runtime.seed == 0
    assert config.output.file == "custom.csv"
    assert not config.output.resume
    assert load_coord_config(path).output.resume


def test_model_flag_replaces_yaml_constructor(tmp_path: Path) -> None:
    path = tmp_path / "coord.yaml"
    path.write_text("model: {name: custom, target: custom.Encoder, kwargs: {feature: pooled}}\n")
    model = load_config(_parse(f"--config {path} --model sincos")).model
    assert (model.name, model.target, model.kwargs) == ("sincos", None, {})


@pytest.mark.parametrize("name", ["timm/resnet50", "not-a-model"])
def test_run_reports_config_errors_with_argparse_status(
    capsys: pytest.CaptureFixture[str], name: str
) -> None:
    with pytest.raises(SystemExit) as error:
        run(_parse(f"--model {name} --dry-run"))
    assert error.value.code == 2
    assert re.search(r"image model|Unknown model config", capsys.readouterr().err)


def test_coord_rejects_abbreviated_flags(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        cli_main(["coord", "--model", "sincos", "--knn-k", "5", "--dry-run"])
    assert error.value.code == 2
    assert "--knn-k" in capsys.readouterr().err


def test_dry_run_roundtrip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run(_parse("--model sincos --methods linear --dry-run"))
    path = tmp_path / "coord.yaml"
    path.write_text(capsys.readouterr().out)
    config = load_coord_config(path)
    assert config.evaluation.methods == ["linear"]
    assert load_config(_parse(f"--config {path}")) == config


@pytest.mark.parametrize(
    "model",
    [{"name": "mind"}, {"name": "custom", "target": "uncached_optional_encoder.Custom", "kwargs": {"dim": 8}}],
)  # fmt: skip
def test_dry_run_does_not_import_runtime(tmp_path: Path, model: dict[str, Any]) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"model": model, "datasets": ["country"]}))
    script = """
import importlib.abc, sys
blocked = {"torch", "numpy", "pandas", "omegaconf", "sklearn", "huggingface_hub",
           "uncached_optional_encoder", "torchgeo_bench.coordbench.run",
           "torchgeo_bench.coordbench.datasets"}
class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked or fullname.split(".")[0] in blocked:
            raise AssertionError(f"dry-run imported {fullname}")
sys.meta_path.insert(0, BlockRuntime())
from torchgeo_bench.cli import main
main(["coord", "--config", sys.argv[1], "--dry-run"])
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        capture_output=True, text=True, check=False, env={**os.environ, "HF_HUB_OFFLINE": "1"},
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    assert yaml.safe_load(result.stdout)["model"] == {"target": None, "kwargs": {}, **model}


def test_command_dispatches_typed_config(monkeypatch: pytest.MonkeyPatch) -> None:
    configs = []
    monkeypatch.setattr("torchgeo_bench.coordbench.run.run_coordbench", configs.append)
    run(_parse("--model sincos --dataset country"))
    (config,) = configs
    assert isinstance(config, CoordConfig)
    assert config.datasets == ["country"]
