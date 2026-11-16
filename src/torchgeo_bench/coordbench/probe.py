"""Frozen-encoder probes for the CoordBench location-encoder track.

Freeze the encoder, embed a labeled dataset's ``(lon, lat)``, fit a head, report a
held-out score. Three heads: ``linear`` (closed-form ridge in torch, CV-selected L2;
R^2 for regression, one-hot-ridge accuracy for classification), ``knn`` (FAISS
k-NN on standardized features, classification only), and ``idw`` (spatiotemporal
inverse-distance weighting on coordinates).

Each ``*_predictor`` returns ``predict(train, test, grid)`` giving one prediction array per grid
point; :func:`evaluate` tunes the grid on each outer fold's inner pairs and scores the outer test
rows. For splits, see splits.py.
"""

import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from torchgeo_bench.coordbench.spatiotemporal_idw import find_knn, knn_predict
from torchgeo_bench.coordbench.splits import OuterFold
from torchgeo_bench.devices import resolve_device
from torchgeo_bench.knn import KNNClassifier


@dataclass(frozen=True)
class ProbeData:
    """Features, coordinates, and labels for one probe, aligned on the benchmark's row axis."""

    features: np.ndarray | None  # None for IDW, which reads only coordinates
    
    labels: np.ndarray  # (N, tasks), one column per task sharing these features
    task_type: str  # "regression" or "classification"
    class_indices: np.ndarray | None  # (N, tasks) class ids for classification

    # Required by IDW, but embedding probes don't need them
    lat: np.ndarray | None = None
    lon: np.ndarray | None = None
    timestamp: np.ndarray | None = None

    @property
    def y(self) -> np.ndarray:
        """Scoring targets: class ids for classification, float labels for regression."""
        return self.labels if self.class_indices is None else self.class_indices


@dataclass(frozen=True)
class ProbeResult:
    """Outer-fold scores of one probe, with the tuned grid point per fold."""

    test_score: float  # mean over outer folds
    fold_scores: list[float]
    best_params: list
    predictions: np.ndarray  # out-of-fold, NaN for rows never tested


def _valid_mask(features: np.ndarray | None, labels: np.ndarray, task_type: str) -> np.ndarray:
    """Rows with a finite label and no non-finite feature (drops nodata/NaN)."""
    if task_type == "regression":
        valid = np.isfinite(labels.astype(np.float64))
    else:
        valid = np.array([v is not None and (isinstance(v, str) or np.isfinite(v)) for v in labels])
    return valid if features is None else valid & np.isfinite(features).all(axis=1)


def _score(y_true: np.ndarray, pred: np.ndarray, task_type: str) -> np.ndarray:
    """R^2 for regression, accuracy for classification, per task column."""
    if task_type == "regression":
        ss_tot = np.maximum(((y_true - y_true.mean(0)) ** 2).sum(0), 1e-12)
        return 1.0 - ((y_true - pred) ** 2).sum(0) / ss_tot
    return (pred == y_true).mean(0)


def evaluate(
    predict: Callable[[np.ndarray, np.ndarray, list], list[np.ndarray]],
    grid: list,
    folds: list[OuterFold],
    data: ProbeData,
) -> list[ProbeResult]:
    """Per task, tune on each outer fold's inner pairs, refit the best point, score its test."""
    y, task_type = data.y, data.task_type
    fold_scores, best_params = [[] for _ in range(y.shape[1])], [[] for _ in range(y.shape[1])]
    predictions = np.full(y.shape, np.nan)
    for fold in folds:
        best = [grid[0]] * y.shape[1]
        if len(grid) > 1:
            inner = [
                [_score(y[val], pred, task_type) for pred in predict(train, val, grid)]
                for train, val in fold.inner
            ]
            # Ties keep the first grid point.
            best = [grid[i] for i in np.argmax(np.mean(inner, axis=0), axis=0)]
            for point in best:
                if not isinstance(point, tuple) and point in (grid[0], grid[-1]):
                    warnings.warn(f"{point} selected at grid edge; widen the grid", stacklevel=2)
        points = list(dict.fromkeys(best))
        refit = predict(fold.train, fold.test, points)
        pred = np.column_stack([refit[points.index(b)][:, i] for i, b in enumerate(best)])
        for i, score in enumerate(_score(y[fold.test], pred, task_type)):
            fold_scores[i].append(float(score))
            best_params[i].append(best[i])
        predictions[fold.test] = pred
    return [
        ProbeResult(float(np.mean(scores)), scores, params, predictions[:, i])
        for i, (scores, params) in enumerate(zip(fold_scores, best_params, strict=True))
    ]


def ridge_predictions(  # noqa: PLR0913 - data, fold indices, grid, and standardization.
    features: torch.Tensor,
    targets: torch.Tensor,
    train_idx: torch.Tensor,
    test_idx: torch.Tensor,
    alphas: list[float],
    *,
    standardize: bool,
) -> Iterator[torch.Tensor]:
    """Predict each alpha from shared fold statistics with an unpenalized intercept."""
    x_tr, x_te = features[train_idx], features[test_idx]
    if standardize:
        mean, std = x_tr.mean(0, keepdim=True), x_tr.std(0, keepdim=True).clamp_min(1e-6)
        x_tr.sub_(mean).div_(std)
        x_te.sub_(mean).div_(std)
    # float64 normal equations: for high-dim features a small alpha is otherwise lost to
    # float32 rounding and the Gram matrix goes singular.
    x_tr, x_te = x_tr.double(), x_te.double()
    y_tr = targets[train_idx].double()
    x_mean, y_mean = x_tr.mean(0, keepdim=True), y_tr.mean(0, keepdim=True)
    x_tr.sub_(x_mean)
    x_te.sub_(x_mean)
    y_tr.sub_(y_mean)
    gram = x_tr.T @ x_tr
    rhs = x_tr.T @ y_tr
    del x_tr, y_tr
    diagonal = gram.diagonal().clone()
    for alpha in alphas:
        gram.diagonal().copy_(diagonal + alpha)
        weight = torch.linalg.solve(gram, rhs)
        yield (x_te @ weight).add_(y_mean)


def ridge_predictor(
    data: ProbeData, device: str, *, standardize: bool = True
) -> Callable[[np.ndarray, np.ndarray, list[float]], list[np.ndarray]]:
    """Ridge on each task's target (regression) or one-hot class ids (classification); grid=alphas.

    Every task is an output column of one ridge, so each fold's normal equations are built once.
    """
    dev = resolve_device(device)
    x = torch.as_tensor(data.features, dtype=torch.float32, device=dev)
    if data.class_indices is None:
        targets = torch.as_tensor(data.labels, dtype=torch.float32, device=dev)
    else:
        widths = [int(w) for w in data.class_indices.max(0) + 1]
        classes = torch.as_tensor(data.class_indices.clip(min=0), device=dev)
        targets = torch.cat(
            [torch.nn.functional.one_hot(c, w) for c, w in zip(classes.T, widths, strict=True)], 1
        ).float()

    def predict(train: np.ndarray, test: np.ndarray, alphas: list[float]) -> list[np.ndarray]:
        tr, te = torch.as_tensor(train, device=dev), torch.as_tensor(test, device=dev)
        preds = ridge_predictions(x, targets, tr, te, alphas, standardize=standardize)
        if data.class_indices is None:
            return [p.cpu().numpy() for p in preds]
        return [
            torch.stack([q.argmax(1) for q in p.split(widths, 1)], 1).cpu().numpy() for p in preds
        ]

    return predict


def knn_predictor(
    data: ProbeData, device: str
) -> Callable[[np.ndarray, np.ndarray, list[int]], list[np.ndarray]]:
    """k-NN classifier on features standardized per train fold; grid = ks."""
    if data.class_indices is None:
        raise ValueError("knn probe is classification-only")

    def predict(train: np.ndarray, test: np.ndarray, ks: list[int]) -> list[np.ndarray]:
        scaler = StandardScaler().fit(data.features[train])
        x_tr = scaler.transform(data.features[train]).astype(np.float32)
        x_te = scaler.transform(data.features[test]).astype(np.float32)
        preds = []
        for k in ks:
            per_task = []
            for classes in data.class_indices[train].T:
                clf = KNNClassifier(n_neighbors=k, device=device, metric="l2")
                clf.fit(x_tr, classes)
                per_task.append(np.asarray(clf.predict(x_te)).ravel())
            preds.append(np.column_stack(per_task))
        return preds

    return predict


def idw_predictor(
    data: ProbeData, device: str
) -> Callable[[np.ndarray, np.ndarray, list[tuple[int, float]]], list[np.ndarray]]:
    """Spatiotemporal IDW on coordinates; grid = ``(k, c)`` pairs, one neighbor search per c."""
    if data.lat is None or data.lon is None:
        raise ValueError("idw probe needs lat and lon in its ProbeData")
    lat, lon, timestamp, y = data.lat, data.lon, data.timestamp, data.y
    n_classes = np.zeros(y.shape[1], int) if data.class_indices is None else y.max(0) + 1

    def predict(
        train: np.ndarray, test: np.ndarray, grid: list[tuple[int, float]]
    ) -> list[np.ndarray]:
        t_tr = None if timestamp is None else timestamp[train]
        t_te = None if timestamp is None else timestamp[test]
        preds = {}
        for c in {c for _, c in grid}:
            ks = [k for k, grid_c in grid if grid_c == c]
            distances, indices = find_knn(
                lat[train], lon[train], t_tr, lat[test], lon[test], t_te,
                kmax=max(ks), c_param=c, device=device,
            )  # fmt: skip
            per_task = [
                knn_predict(
                    distances, indices, ks, y_tr, lat[test], lon[test], data.task_type, int(n)
                )
                for y_tr, n in zip(y[train].T, n_classes, strict=True)
            ]
            preds.update({(k, c): np.column_stack([p[k] for p in per_task]) for k in ks})
        return [preds[point] for point in grid]

    return predict
