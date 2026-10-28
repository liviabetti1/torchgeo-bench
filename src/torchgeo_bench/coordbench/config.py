"""Strict, lightweight configuration for coordinate encoder evaluation."""

from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import (
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from torchgeo_bench.config.presets import ModelPreset, load_model_preset
from torchgeo_bench.config.schema import (
    Device,
    KnnDevice,
    Methods,
    ModelConfig,
    OutputPaths,
    SchemaVersion,
    StrictModel,
    default_methods,
    load_yaml,
)
from torchgeo_bench.coordbench.catalog import FAMILY_BENCHMARKS
from torchgeo_bench.coordbench.config_enums import TemporalResolution

RIDGE_ALPHAS = tuple(10.0 ** (step / 2) for step in range(-8, 13))


class CoordRuntimeConfig(StrictModel):
    """Coordinate encoder and linear-probe execution settings."""

    device: Device = "cpu"
    seed: StrictInt = Field(default=0, ge=0, le=2**64 - 1)


class CoordOutputConfig(OutputPaths):
    """Coordinate result CSV and resume settings."""

    resume: StrictBool = False


class CoordEvaluationConfig(StrictModel):
    """Random/spatial cross-validation and coordinate probe settings."""

    methods: Methods = Field(default_factory=default_methods, min_length=1)
    split: Literal["random", "spatial", "temporal", "all"] = "random"
    folds: StrictInt = Field(default=5, ge=2)
    cell_deg: StrictFloat = Field(default=10.0, gt=0)
    cell_degs: list[Annotated[StrictFloat, Field(gt=0)]] | None = None
    # A DISCRETIZATION_METHODS_TO_SECONDS name, e.g. "weekly".
    temporal_grid_size: StrictStr | None = None
    temporal_grid_sizes: list[StrictStr] | None = None
    knn_k: StrictInt = Field(default=5, gt=0)
    knn_device: KnnDevice = "cpu"
    ridge_alphas: list[Annotated[StrictFloat, Field(gt=0)]] = Field(
        default_factory=lambda: list(RIDGE_ALPHAS), min_length=1
    )
    skip_no_timestamp: StrictBool = False
    temporal_sampling_year: StrictInt | None = None
    temporal_sampling_methods: list[StrictStr] = Field(default_factory=list)
    temporal_discretization_methods: list[StrictStr] = Field(default_factory=list)
    temporal_embedding_aggregation_methods: list[StrictStr] = Field(
        default_factory=lambda: ["statistics"], min_length=1
    )

    exclude_antarctica: StrictBool = False

    from_polygon: StrictBool = False
    spatial_aggregation_methods: list[StrictStr] = Field(default_factory=list)
    subsample: StrictBool = False
    temporal_checkerboard: StrictBool = False

    @model_validator(mode="after")
    def validate_temporal_grid(self) -> Self:
        """Require a known temporal grid size whenever the temporal split is requested alone."""
        sizes = self.temporal_grid_sizes or (
            [self.temporal_grid_size] if self.temporal_grid_size else []
        )
        unknown = sorted(set(sizes) - {r.value for r in TemporalResolution})
        if unknown:
            raise ValueError(f"Unknown temporal grid sizes: {unknown}")
        if self.split == "temporal" and not sizes:
            raise ValueError("split='temporal' requires temporal_grid_size or temporal_grid_sizes")
        return self


class CoordConfig(StrictModel):
    """Complete coordinate benchmark configuration, independent of image settings."""

    schema_version: SchemaVersion = 1
    model: ModelConfig
    datasets: list[StrictStr] = Field(default_factory=lambda: ["all"], min_length=1)
    evaluation: CoordEvaluationConfig = Field(default_factory=CoordEvaluationConfig)
    runtime: CoordRuntimeConfig = Field(default_factory=CoordRuntimeConfig)
    output: CoordOutputConfig = Field(default_factory=CoordOutputConfig)

    @field_validator("datasets")
    @classmethod
    def validate_datasets(cls, value: list[str]) -> list[str]:
        """Require distinct, non-blank names and an exclusive all selection."""
        if any(not name.strip() for name in value) or len(set(value)) != len(value):
            raise ValueError("datasets must contain distinct non-empty names")
        if "all" in value and len(value) != 1:
            raise ValueError("'all' cannot be combined with other datasets")
        known = {"all", *FAMILY_BENCHMARKS}
        known.update(name for names in FAMILY_BENCHMARKS.values() for name in names)
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(f"Unknown coordinate datasets: {unknown}")
        return value

    def model_dump_yaml(self) -> dict[str, Any]:
        """Return a reusable YAML mapping with explicit effective defaults."""
        return self.model_dump(mode="json")


def load_coord_config(path: str | Path) -> CoordConfig:
    """Load one strictly validated coordinate YAML configuration."""
    return CoordConfig.model_validate(load_yaml(path))


def resolve_coord_preset(config: CoordConfig) -> ModelPreset:
    """Resolve a coordinate preset without importing encoder implementations."""
    preset = load_model_preset(config.model, seed=config.runtime.seed)
    if config.model.target is None and preset.track != "coord":
        raise ValueError(f"{config.model.name!r} is an image model; use 'run'")
    return preset
