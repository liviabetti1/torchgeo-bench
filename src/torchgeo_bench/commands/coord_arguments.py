"""Argument registration for the coordinate benchmark command."""

import argparse

from ._config import add_output_arguments


def add_coord_arguments(parser: argparse.ArgumentParser) -> None:
    """Register explicit coordinate flags without importing runtime dependencies."""
    parser.add_argument("--config", help="Coordinate YAML configuration")
    parser.add_argument("--model", default=argparse.SUPPRESS, help="Coordinate encoder preset")
    parser.add_argument(
        "--dataset",
        "--datasets",
        "--name",
        "--names",
        dest="datasets",
        nargs="+",
        default=argparse.SUPPRESS,
        help="Benchmark names or families, or all",
    )
    parser.add_argument(
        "--methods", choices=("knn", "linear", "idw"), nargs="+", default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--split-test-type", choices=("official", "nested_cv"), default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--nested-splits",
        type=lambda s: tuple(s.split(",")),
        nargs="+",
        default=argparse.SUPPRESS,
        help="outer,inner splitter pairs, e.g. temporal_checkerboard,random spatial,random",
    )
    parser.add_argument("--folds", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--inner-folds", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--cell-deg", type=float, default=argparse.SUPPRESS)
    parser.add_argument(
        "--time-units",
        type=lambda s: (s.split(",")[0], int(s.split(",")[1])),
        nargs="+",
        default=argparse.SUPPRESS,
        help="unit,size pairs for temporal splits, e.g. week,1 month,1 year,2",
    )
    parser.add_argument(
        "--reset-year", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS
    )
    parser.add_argument("--max-train-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--test-size", type=int, default=argparse.SUPPRESS,
        help="Time units per eval window for the temporal_forecast splits",
    )
    parser.add_argument(
        "--ridge-alphas", type=float, nargs="+", default=argparse.SUPPRESS,
        help="Positive ridge penalties to CV-select; one value disables alpha tuning",
    )
    parser.add_argument("--knn-ks", type=int, nargs="+", default=argparse.SUPPRESS)
    parser.add_argument("--knn-device", default=argparse.SUPPRESS)
    parser.add_argument("--idw-ks", type=int, nargs="+", default=argparse.SUPPRESS)
    parser.add_argument("--idw-cs", type=float, nargs="+", default=argparse.SUPPRESS)
    parser.add_argument(
        "--skip-no-timestamps",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--temporal-discretization-methods", nargs="+", default=argparse.SUPPRESS
    )
    parser.add_argument("--temporal-sampling-year", type=int, default=argparse.SUPPRESS)

    parser.add_argument("--temporal-sampling-methods", nargs="+", default=argparse.SUPPRESS)
    parser.add_argument(
        "--temporal-embedding-aggregation-methods", nargs="+", default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--exclude-antarctica",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--from-polygon",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--subsample",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--save-predictions",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--spatial-aggregation-methods", nargs="+", default=argparse.SUPPRESS
    )
    parser.add_argument("--device", default=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    add_output_arguments(parser)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print validated YAML without running"
    )
