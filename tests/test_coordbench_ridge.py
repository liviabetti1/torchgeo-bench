"""Sklearn parity for CoordBench ridge fitting and nested alpha selection."""

import numpy as np
import pytest
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from torchgeo_bench.coordbench.probe import ProbeData, evaluate, ridge_predictions, ridge_predictor
from torchgeo_bench.coordbench.splits import OuterFold


def _inner(train: np.ndarray, folds: int = 3) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(np.setdiff1d(train, train[i::folds]), train[i::folds]) for i in range(folds)]


@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("task_type", ["regression", "classification"])
@pytest.mark.parametrize("official", [False, True])
def test_ridge_matches_sklearn(*, standardize: bool, task_type: str, official: bool) -> None:
    rng = np.random.default_rng(42)
    features = (rng.normal(size=(90, 8)) + 12).astype(np.float32)
    features[:, -1] = 7  # Constant and duplicate columns exercise rank-deficient features.
    features[:, -2] = features[:, 0]
    labels = (features[:, 0] * 3 + rng.normal(size=90) + 100).astype(np.float32)
    if task_type == "classification":
        labels = np.digitize(labels, np.quantile(labels, [0.15, 0.4])).astype(np.int64)
    targets = labels[:, None] if task_type == "regression" else np.eye(3)[labels]
    alphas = [10.0, 0.01, 10000.0, 0.01]
    rows = np.arange(90)
    outer = (
        [(rows[:60], rows[60:])]
        if official
        else [(np.setdiff1d(rows, rows[i::3]), rows[i::3]) for i in range(3)]
    )
    folds = [OuterFold(train, test, _inner(train)) for train, test in outer]

    def score(train: np.ndarray, test: np.ndarray, alpha: float) -> float:
        x_train, x_test = torch.from_numpy(features[train]), torch.from_numpy(features[test])
        if standardize:
            mean = x_train.mean(0, keepdim=True)
            std = x_train.std(0, keepdim=True).clamp_min(1e-6)
            x_train, x_test = (x_train - mean) / std, (x_test - mean) / std
        model = Ridge(alpha=alpha, fit_intercept=True, solver="cholesky")
        model.fit(x_train.double().numpy(), targets[train].astype(np.float64))
        pred = model.predict(x_test.double().numpy())
        if task_type == "classification":
            return float(np.mean(pred.argmax(1) == labels[test]))
        return float(r2_score(targets[test], pred))

    expected = []
    for fold in folds:
        inner = [[score(train, val, alpha) for train, val in fold.inner] for alpha in alphas]
        best = alphas[int(np.argmax(np.mean(inner, axis=1)))]
        expected.append(score(fold.train, fold.test, best))

    if task_type == "regression":
        data = ProbeData(features, labels.astype(np.float64), task_type, None)
    else:
        data = ProbeData(features, labels, task_type, labels)
    result = evaluate(ridge_predictor(data, "cpu", standardize=standardize), alphas, folds, data)
    np.testing.assert_allclose(result.fold_scores, expected, atol=2e-7, rtol=2e-7)
    assert result.test_score == pytest.approx(np.mean(expected), abs=2e-7)


def test_fixed_alpha_skips_inner_tuning() -> None:
    features = np.arange(40, dtype=np.float32).reshape(20, 2)
    data = ProbeData(features, features[:, 0] + 50.0, "regression", None)
    predict = ridge_predictor(data, "cpu")
    calls = []

    def counting_predict(train: np.ndarray, test: np.ndarray, grid: list) -> list[np.ndarray]:
        calls.append(len(test))
        return predict(train, test, grid)

    rows = np.arange(20)
    folds = [OuterFold(rows[:15], rows[15:], _inner(rows[:15]))]
    result = evaluate(counting_predict, [0.01], folds, data)
    assert calls == [5]  # Only the outer refit; a single alpha needs no inner CV.
    assert result.test_score > 0.99
    assert result.fold_scores == [result.test_score]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ridge_preserves_shared_tensors(*, dtype: torch.dtype) -> None:
    features = torch.arange(60, dtype=dtype).reshape(20, 3)
    targets = features[:, :1].clone() + 10
    before_features, before_targets = features.clone(), targets.clone()
    list(
        ridge_predictions(
            features, targets, torch.arange(15), torch.arange(15, 20), [1.0, 10.0], standardize=True
        )
    )
    torch.testing.assert_close(features, before_features, rtol=0, atol=0)
    torch.testing.assert_close(targets, before_targets, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(80, 6), (20, 40)])
@pytest.mark.parametrize("outputs", [1, 3])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
        ),
    ],
)
def test_ridge_predictions_match_sklearn(shape: tuple[int, int], outputs: int, device: str) -> None:
    rng = np.random.default_rng(17)
    n_train, n_features = shape
    features = rng.normal(size=(n_train + 12, n_features)) + 7
    features[:, -1] = 3
    features[:, -2] = features[:, 0]
    targets = features @ rng.normal(size=(n_features, outputs)) + 250
    targets += rng.normal(size=targets.shape)
    alphas = [100.0, 0.01, 1.0, 100.0]
    predictions = ridge_predictions(
        torch.as_tensor(features, device=device),
        torch.as_tensor(targets, device=device),
        torch.arange(n_train, device=device),
        torch.arange(n_train, len(features), device=device),
        alphas,
        standardize=False,
    )
    for alpha, actual in zip(alphas, predictions, strict=True):
        reference = Ridge(alpha=alpha).fit(features[:n_train], targets[:n_train])
        expected = reference.predict(features[n_train:]).reshape(-1, outputs)
        np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=1e-10, atol=1e-10)


def test_inner_cv_builds_normal_equations_once_per_fold(monkeypatch: pytest.MonkeyPatch) -> None:
    products = []
    matmul = torch.Tensor.__matmul__

    def record(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        products.append((tuple(left.shape), tuple(right.shape)))
        return matmul(left, right)

    generator = torch.Generator().manual_seed(17)
    features = torch.randn(60, 6, generator=generator).numpy()
    labels = torch.randn(60, generator=generator).numpy().astype(np.float64)
    data = ProbeData(features, labels, "regression", None)
    rows = np.arange(60)
    folds = [OuterFold(rows[:45], rows[45:], _inner(rows[:45]))]
    monkeypatch.setattr(torch.Tensor, "__matmul__", record)
    evaluate(ridge_predictor(data, "cpu"), [0.01, 100.0, 1.0], folds, data)
    assert products.count(((6, 30), (30, 6))) == 3  # one Gram per inner fold, shared by alphas
    assert products.count(((6, 45), (45, 6))) == 1  # the outer refit


def test_ties_keep_first_alpha() -> None:
    data = ProbeData(np.zeros((12, 2), dtype=np.float32), np.full(12, 7.0), "regression", None)
    rows = np.arange(12)
    folds = [OuterFold(rows[:9], rows[9:], _inner(rows[:9]))]
    with pytest.warns(UserWarning, match="grid edge"):
        result = evaluate(ridge_predictor(data, "cpu"), [10.0, 0.1, 1.0], folds, data)
    assert result.best_params == [10.0]
    assert result.fold_scores == [1.0]
