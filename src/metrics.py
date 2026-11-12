"""Evaluation metrics shared by every model (baselines, deep models, graph models).

All functions take arrays shaped like the dataset windows:
  y, pred   [S, H, N]   S forecast origins, H forecast days, N modules
and report one block of metrics per requested horizon h (forecast day t+h -> index h-1),
pooling all origins and modules.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def _slice(a: np.ndarray, h: int) -> np.ndarray:
    return np.asarray(a, dtype=float)[:, h - 1, :]


def poisson_deviance(y: np.ndarray, mu: np.ndarray, floor: float = 0.05) -> float:
    """Mean Poisson deviance 2 * [y log(y / mu) - (y - mu)]; mu floored at `floor`."""
    y, mu = np.asarray(y, float).ravel(), np.maximum(np.asarray(mu, float).ravel(), floor)
    term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return float(np.mean(2 * (term - (y - mu))))


def top_k_hits(event: np.ndarray, score: np.ndarray, k: int = 3) -> np.ndarray:
    """Per day: how many of the k highest-scored modules had the event. event, score: [S, N].

    Ties at the k-th score are resolved by expectation (as if broken at random), so constant
    or 0/1 scores are neither rewarded nor punished by the order of the modules.
    """
    event, score = np.asarray(event, float), np.asarray(score, float)
    hits = np.empty(len(event))
    for i, (e, s) in enumerate(zip(event, score, strict=True)):
        kth = np.sort(s)[::-1][k - 1]
        above, tied = s > kth, s == kth
        hits[i] = e[above].sum() + (k - above.sum()) * e[tied].mean()
    return hits


def precision_at_k(event: np.ndarray, score: np.ndarray, k: int = 3) -> float:
    """Of the k modules flagged riskiest each day, the share that had the event."""
    return float(top_k_hits(event, score, k).mean() / k)


def recall_at_k(event: np.ndarray, score: np.ndarray, k: int = 3) -> float:
    """Share of all event module-days that were among that day's k flagged modules
    (recall under a fixed alert budget of k modules per day; needs no threshold)."""
    return float(top_k_hits(event, score, k).sum() / max(np.asarray(event).sum(), 1.0))


def regression_metrics(
    y: np.ndarray, pred: np.ndarray, horizons: list[int], k: int = 3, floor: float = 0.05
) -> dict[str, dict[str, float]]:
    out = {}
    for h in horizons:
        yt, pt = _slice(y, h), _slice(pred, h)
        err = pt - yt
        out[f"h{h}"] = {
            "MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err**2))),
            "PoissonDev": poisson_deviance(yt, pt, floor),
            f"P@{k}": precision_at_k(yt > 0, pt, k),  # event = at least one incident
        }
    return out


def f1_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Score threshold with the best F1 (chosen on the validation split, then reused)."""
    y, score = np.asarray(y).ravel(), np.asarray(score).ravel()
    precision, recall, thresholds = precision_recall_curve(y, score)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[np.argmax(f1[:-1])])


def score(
    target: str, truth: dict, pred: dict, horizons: list[int], k: int = 3, floor: float = 0.05
) -> dict:
    """Metrics for {"val": ..., "test": ...} arrays. Classification thresholds (used only for
    recall/precision_at_threshold) are chosen on validation and reused on test."""
    if target == "regression":
        return {p: regression_metrics(truth[p], pred[p], horizons, k, floor) for p in pred}
    thr = {f"h{h}": f1_threshold(truth["val"][:, h - 1], pred["val"][:, h - 1]) for h in horizons}
    return {
        p: {
            **classification_metrics(truth[p], pred[p], horizons, thr, k),
            "threshold_from_val": thr,
        }
        for p in pred
    }


def classification_metrics(
    y: np.ndarray,
    score: np.ndarray,
    horizons: list[int],
    thresholds: dict[str, float],
    k: int = 3,
) -> dict[str, dict[str, float]]:
    """y: 1 if at least one P1/P2 incident. thresholds: {"h1": t, ...} from validation."""
    out = {}
    for h in horizons:
        yt, st = _slice(y, h), _slice(score, h)
        yf, sf = yt.ravel(), st.ravel()
        flagged = sf >= thresholds[f"h{h}"]
        tp = float((flagged & (yf == 1)).sum())
        out[f"h{h}"] = {
            "PR_AUC": float(average_precision_score(yf, sf)),
            "ROC_AUC": float(roc_auc_score(yf, sf)),
            f"P@{k}": precision_at_k(yt, st, k),
            f"R@{k}": recall_at_k(yt, st, k),
            # at the F1-optimal threshold from validation; can degenerate to "flag everything"
            # (recall 1, precision = positive rate) for weak scores such as the naive 0/1
            "recall_at_threshold": tp / max(float((yf == 1).sum()), 1.0),
            "precision_at_threshold": tp / max(float(flagged.sum()), 1.0),
            "positive_rate": float(yf.mean()),
        }
    return out
