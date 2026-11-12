"""Strict, lightweight configuration for coordinate encoder evaluation."""

from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import (
    Field,
    Strict,
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
    ModelConfig,
    OutputPaths,
    SchemaVersion,
    StrictModel,
    load_yaml,
)
from torchgeo_bench.coordbench.catalog import FAMILY_BENCHMARKS

RIDGE_ALPHAS = tuple(10.0 ** (step / 2) for step in range(-8, 13))

Split = Literal[
    "random",
    "spatial",
    "temporal_forecast",
    "temporal_forecast_block",
    "temporal_checkerboard",
    "temporal_contiguous_block",
]


class CoordRuntimeConfig(StrictModel):
    """Coordinate encoder and linear-probe execution settings."""

    device: Device = "cpu"
    seed: StrictInt = Field(default=0, ge=0, le=2**64 - 1)


class CoordOutputConfig(OutputPaths):
    """Coordinate result CSV and resume settings."""

    resume: StrictBool = False


class CoordEvaluationConfig(StrictModel):
    """Test-set protocol, splitter, and coordinate probe settings."""

    methods: list[Literal["knn", "linear", "idw"]] = Field(default=["knn", "linear"], min_length=1)
    split_test_type: Literal["official", "nested_cv"] = "nested_cv"
    # (outer, inner) splitter pairs; Strict(False) lets YAML lists validate as tuples.
    nested_splits: list[Annotated[tuple[Split, Split], Strict(False)]] = Field(
        default=[("random", "random")], min_length=1
    )
    folds: StrictInt = Field(default=5, ge=2)
    inner_folds: StrictInt = Field(default=5, ge=2)
    cell_deg: StrictFloat = Field(default=10.0, gt=0)
    # (time_unit, time_unit_size) pairs; each temporal nested split runs once per pair.
    time_units: list[
        Annotated[
            tuple[
                Literal["timestamp", "day", "week", "month", "quarter", "season", "year"],
                Annotated[StrictInt, Field(gt=0)],
            ],
            Strict(False),
        ]
    ] = Field(default=[("month", 1)], min_length=1)
    reset_year: StrictBool = False
    max_train_size: StrictInt | None = None  # time units in the temporal_forecast_block window
    test_size: StrictInt = Field(default=1, gt=0)  # time units forecast per temporal_forecast fold
    knn_ks: list[StrictInt] = Field(default=[5])
    knn_device: KnnDevice = "cpu"
    idw_ks: list[StrictInt] = Field(default=[5])
    idw_cs: list[StrictFloat] = Field(default=[0.0])
    ridge_alphas: list[Annotated[StrictFloat, Field(gt=0)]] = Field(
        default_factory=lambda: list(RIDGE_ALPHAS), min_length=1
    )
    skip_no_timestamp: StrictBool = False
    temporal_sampling_year: StrictInt | None = None
    temporal_sampling_methods: list[StrictStr] = Field(default_factory=list)
    temporal_discretization_methods: list[StrictStr] = Field(default_factory=list)
    temporal_embedding_aggregation_methods: list[StrictStr] = Field(
        default_factory=lambda: ["mean"], min_length=1
    )

    exclude_antarctica: StrictBool = True

    from_polygon: StrictBool = False
    spatial_aggregation_methods: list[StrictStr] = Field(default_factory=list)
    subsample: StrictBool = False
    save_predictions: StrictBool = False

    @model_validator(mode="after")
    def require_window_size(self) -> Self:
        """Reject duplicate methods or splits and windowed temporal splits without window sizes."""
        if len(set(self.methods)) != len(self.methods):
            raise ValueError("methods must not contain duplicates")
        if len(set(self.nested_splits)) != len(self.nested_splits):
            raise ValueError("nested_splits must not contain duplicates")
        if len(set(self.time_units)) != len(self.time_units):
            raise ValueError("time_units must not contain duplicates")
        if self.split_test_type == "official" and len({o for o, _ in self.nested_splits}) > 1:
            raise ValueError("split_test_type='official' ignores the outer split; give it one")
        splits = {s for pair in self.nested_splits for s in pair}
        if "temporal_forecast_block" in splits and self.max_train_size is None:
            raise ValueError("temporal_forecast_block requires max_train_size")
        if "temporal_forecast_block" in splits and self.test_size > self.max_train_size:
            raise ValueError("test_size must not exceed max_train_size")
        if self.test_size != 1 and not any(s.startswith("temporal_forecast") for s in splits):
            raise ValueError("test_size only applies to the temporal_forecast splits")
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
