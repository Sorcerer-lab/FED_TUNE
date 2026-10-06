"""Plain-numpy metrics (no torch needed), so they can be tested anywhere."""
import numpy as np


def accuracy(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def macro_f1(y_true, y_pred) -> float:
    """Mean of per-class F1 over every class that appears in the truth or the predictions.
    F1 of a class = 2TP / (2TP + FP + FN). Rare classes count as much as common ones."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    scores = []
    for c in np.union1d(y_true, y_pred):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(scores))


def dispersion(values) -> dict:
    """Spread of a per-client metric: mean, standard deviation, worst and best client."""
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()), "max": float(v.max())}
