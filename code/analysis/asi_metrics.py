"""
asi_metrics.py — Class-Wise Adaptation Sufficiency Index (ASI) 核心指标

ASI 衡量 frozen foundation model 特征空间中，某缺陷类相对其余类别的可分性。
值越高 → 均匀 PEFT 越容易达到高 IoU；值越低 → 需要更强的局部归纳偏置 (Conv-LoRA)。

组成 (默认等权):
  - Fisher ratio (one-vs-rest)
  - Linear probe AUC (one-vs-rest)
  - Silhouette coefficient (one-vs-rest 二值标签)
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DEFECT_CLASSES = ["patches", "inclusion", "scratches"]
CLASS_ID_TO_NAME = {1: "patches", 2: "inclusion", 3: "scratches"}


def fisher_ratio_one_vs_rest(features: np.ndarray, labels: np.ndarray, class_id: int) -> float:
    """
    Fisher discriminant ratio for class_id vs all other pixels.
    features: (N, D), labels: (N,) integer class ids
    """
    pos = features[labels == class_id]
    neg = features[labels != class_id]
    if len(pos) < 2 or len(neg) < 2:
        return 0.0

    mu_p = pos.mean(axis=0)
    mu_n = neg.mean(axis=0)
    var_p = pos.var(axis=0) + 1e-8
    var_n = neg.var(axis=0) + 1e-8

    between = np.sum((mu_p - mu_n) ** 2)
    within = np.sum(var_p + var_n)
    return float(between / (within + 1e-8))


def linear_probe_auc_one_vs_rest(
    features: np.ndarray,
    labels: np.ndarray,
    class_id: int,
    max_train: int = 8000,
    seed: int = 42,
) -> float:
    """Train logistic regression (one-vs-rest) and return ROC-AUC."""
    y = (labels == class_id).astype(np.int32)
    if y.sum() < 10 or (1 - y).sum() < 10:
        return 0.5

    rng = np.random.default_rng(seed)
    n = len(y)
    x = features
    if n > max_train:
        idx = rng.choice(n, max_train, replace=False)
        x, y = features[idx], y[idx]

    try:
        x_tr, x_te, y_tr, y_te = train_test_split(
            x, y, test_size=0.3, random_state=seed, stratify=y,
        )
    except ValueError:
        return 0.5

    scaler = StandardScaler()
    x_tr = scaler.fit_transform(x_tr)
    x_te = scaler.transform(x_te)

    try:
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=seed,
        )
        clf.fit(x_tr, y_tr)
        prob = clf.predict_proba(x_te)[:, 1]
        return float(roc_auc_score(y_te, prob))
    except Exception:
        return 0.5


def silhouette_one_vs_rest(features: np.ndarray, labels: np.ndarray, class_id: int) -> float:
    """Silhouette score for binary problem: class_id vs rest."""
    y = (labels == class_id).astype(np.int32)
    if y.sum() < 5 or (1 - y).sum() < 5:
        return 0.0
    if len(np.unique(y)) < 2:
        return 0.0
    try:
        return float(silhouette_score(features, y, sample_size=min(5000, len(y)), random_state=42))
    except Exception:
        return 0.0


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    """Min-max normalize to [0, 1] across defect classes."""
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-8:
        return {k: 0.5 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def compute_class_asi(
    features: np.ndarray,
    labels: np.ndarray,
    class_id: int,
    weights: dict[str, float] | None = None,
    class_id_to_name: dict[int, str] | None = None,
) -> dict:
    """
    Compute raw metrics and composite ASI for one defect class.

    Returns dict with keys: fisher, auc, silhouette, asi_raw, class_name
    """
    if weights is None:
        weights = {"fisher": 1 / 3, "auc": 1 / 3, "silhouette": 1 / 3}

    id_to_name = class_id_to_name or CLASS_ID_TO_NAME
    name = id_to_name[class_id]
    fisher = fisher_ratio_one_vs_rest(features, labels, class_id)
    auc = linear_probe_auc_one_vs_rest(features, labels, class_id)
    sil = silhouette_one_vs_rest(features, labels, class_id)

    return {
        "class_id": class_id,
        "class_name": name,
        "fisher": fisher,
        "auc": auc,
        "silhouette": sil,
        "n_pixels": int((labels == class_id).sum()),
    }


def compute_asi_table(
    features: np.ndarray,
    labels: np.ndarray,
    defect_class_ids: list[int] | None = None,
    class_id_to_name: dict[int, str] | None = None,
) -> dict:
    """
    Compute per-class metrics and composite ASI for all defect classes.

    ASI 流程:
      1. 对 fisher / auc / silhouette 分别做 min-max 归一化 (跨类)
      2. ASI_c = w_f * fisher_norm + w_a * auc_norm + w_s * sil_norm
    """
    if defect_class_ids is None:
        defect_class_ids = [1, 2, 3]

    per_class = []
    for cid in defect_class_ids:
        if (labels == cid).sum() == 0:
            continue
        per_class.append(
            compute_class_asi(features, labels, cid, class_id_to_name=class_id_to_name)
        )

    if not per_class:
        return {"per_class": [], "mean_asi": 0.0}

    if len(per_class) == 1:
        r = per_class[0]
        r["fisher_norm"] = r["auc_norm"] = r["silhouette_norm"] = 1.0
        r["asi"] = 1.0
        return {"per_class": per_class, "mean_asi": 1.0}

    fisher_norm = normalize_scores({r["class_name"]: r["fisher"] for r in per_class})
    auc_norm = normalize_scores({r["class_name"]: r["auc"] for r in per_class})
    sil_norm = normalize_scores({r["class_name"]: r["silhouette"] for r in per_class})

    for r in per_class:
        name = r["class_name"]
        r["fisher_norm"] = fisher_norm[name]
        r["auc_norm"] = auc_norm[name]
        r["silhouette_norm"] = sil_norm[name]
        r["asi"] = (r["fisher_norm"] + r["auc_norm"] + r["silhouette_norm"]) / 3.0

    mean_asi = float(np.mean([r["asi"] for r in per_class]))
    return {"per_class": per_class, "mean_asi": mean_asi}


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x ** 2).sum() * (y ** 2).sum())
    if denom < 1e-12:
        return float("nan")
    return float((x * y).sum() / denom)


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rx = x.argsort().argsort().astype(np.float64)
    ry = y.argsort().argsort().astype(np.float64)
    return pearson_correlation(rx, ry)
