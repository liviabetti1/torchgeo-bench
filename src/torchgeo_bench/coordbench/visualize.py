"""Show raw performance (not rank) from a CoordBench results CSV: a table + a bar chart per holdout.

Usage: ``python -m torchgeo_bench.coordbench.visualize <results.csv> [--method linear|knn] [--out DIR]``
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from torchgeo_bench.coordbench.taxonomy import R2_FLOOR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--method", default="linear")
    ap.add_argument("--out", default="results/plots")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["method"].astype(str).str.startswith(args.method.rstrip("0123456789"))]
    if df.empty:
        raise SystemExit(f"no rows for method={args.method!r} in {args.csv}")
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    for view in ("random", "spatial"):
        sub = df[df["split"].isin([view, "official"])]
        if sub.empty:
            continue
        d = sub.copy()
        is_r2 = d["metric_name"] == "r2"
        d.loc[is_r2, "metric_value"] = d.loc[is_r2, "metric_value"].clip(lower=R2_FLOOR)
        scores = d.groupby(["dataset", "model_name", "metric_name"], as_index=False)["metric_value"].mean()

        for metric in scores["metric_name"].unique():
            table = scores[scores["metric_name"] == metric].pivot_table(
                index="dataset", columns="model_name", values="metric_value"
            )

            print(f"\n{view.upper()} holdout — {args.method} probe — {metric} (higher=better)")
            print(table.round(3))

            ax = table.plot.bar(figsize=(max(7, len(table) * 0.8), 4), title=f"{view} holdout — {args.method} probe")
            ax.set_ylabel(metric)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.figure.tight_layout()
            ax.figure.savefig(out_path / f"{args.method}_{view}_{metric}.png", dpi=150)
            plt.close(ax.figure)

    print(f"\nsaved plots to {out_path}/")


if __name__ == "__main__":
    main()
