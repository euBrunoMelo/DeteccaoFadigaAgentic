"""
Ferramentas para validar o uso do best_model.onnx offline a partir de parquets.

V2: ONNX-only, 19 features (TEV7 Agentic V3). Sem dependência de PyTorch.
Assume DataFrames com janelas agregadas (19 features).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import f1_score, recall_score, roc_auc_score
except ImportError as e:
    raise ImportError(
        "offline_eval requer scikit-learn. "
        "Instale com `pip install scikit-learn` (dev-only, não necessário no RPi)."
    ) from e

from .model_loader import (
    FEATURE_NAMES_19,
    InferenceConfig,
    ONNXModelWrapper,
    scale_features,
)


@dataclass
class OfflineEvalConfig:
    """
    Configuração simples para avaliação offline.

    - `threshold`: limiar de probabilidade para classe Danger.
    - `safe_label`: inteiro representando classe Safe no vetor de labels.
    - `danger_label`: inteiro representando classe Danger no vetor de labels.
    """

    threshold: float = 0.41
    safe_label: int = 0
    danger_label: int = 1


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax estável (numpy)."""
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def run_offline_eval(
    df_windows: pd.DataFrame,
    model: ONNXModelWrapper,
    config: InferenceConfig,
    *,
    eval_config: Optional[OfflineEvalConfig] = None,
    feature_names: Sequence[str] | None = None,
    label_column: str = "label",
) -> Tuple[pd.DataFrame, dict]:
    """
    Avalia o best_model.onnx em um conjunto de janelas agregadas (19 features).

    Parâmetros
    ----------
    df_windows:
        DataFrame com uma linha por janela. Deve conter as 19 features e label_column.
    model:
        ONNXModelWrapper carregado via load_best_model().
    config:
        InferenceConfig com scaler e feature_names.
    eval_config:
        Configuração de avaliação (threshold, labels). Se None, usa config.threshold.
    feature_names:
        Ordem das features. Se None, usa config.feature_names.
    label_column:
        Nome da coluna de rótulo binário (0=Safe, 1=Danger).

    Retornos
    --------
    df_result:
        DataFrame original com colunas extras `prob_danger` e `pred_label`.
    metrics:
        Dicionário com balanced_accuracy, f1_danger, safe_recall, danger_recall, auc_roc.
    """
    if eval_config is None:
        eval_config = OfflineEvalConfig(threshold=config.threshold)
    if feature_names is None:
        feature_names = config.feature_names

    if label_column not in df_windows.columns:
        raise KeyError(
            f"Coluna de label '{label_column}' não encontrada em df_windows."
        )

    y_true = df_windows[label_column].astype(int).to_numpy()

    features_mat = df_windows[list(feature_names)].astype("float32").to_numpy()
    features_scaled = scale_features(features_mat, config)

    logits = model(features_scaled)
    probs = _softmax(logits)[:, eval_config.danger_label]

    preds = (probs >= eval_config.threshold).astype(int)

    safe_recall = recall_score(
        y_true,
        preds,
        pos_label=eval_config.safe_label,
        zero_division=0,
    )
    danger_recall = recall_score(
        y_true,
        preds,
        pos_label=eval_config.danger_label,
        zero_division=0,
    )
    bacc = (safe_recall + danger_recall) / 2.0
    f1_danger = f1_score(
        y_true,
        preds,
        pos_label=eval_config.danger_label,
        zero_division=0,
    )
    try:
        auc = roc_auc_score(y_true, probs)
    except ValueError:
        auc = 0.5

    metrics = {
        "balanced_accuracy": float(bacc),
        "f1_danger": float(f1_danger),
        "safe_recall": float(safe_recall),
        "danger_recall": float(danger_recall),
        "auc_roc": float(auc),
    }

    df_out = df_windows.copy()
    df_out["prob_danger"] = probs
    df_out["pred_label"] = preds
    return df_out, metrics


def trivial_always_danger_baseline(
    y_true: np.ndarray,
    *,
    danger_label: int = 1,
) -> dict:
    """
    Baseline trivial: sempre prever Danger.
    """
    preds = np.full_like(y_true, fill_value=danger_label)

    safe_recall = recall_score(
        y_true, preds, pos_label=0, zero_division=0
    )
    danger_recall = recall_score(
        y_true, preds, pos_label=danger_label, zero_division=0
    )
    bacc = (safe_recall + danger_recall) / 2.0

    f1_danger = f1_score(
        y_true, preds, pos_label=danger_label, zero_division=0
    )

    return {
        "balanced_accuracy": float(bacc),
        "f1_danger": float(f1_danger),
        "safe_recall": float(safe_recall),
        "danger_recall": float(danger_recall),
        "auc_roc": 0.5,
    }


def compare_realtime_offline_features(
    df_offline: pd.DataFrame,
    df_realtime: pd.DataFrame,
    *,
    feature_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Compara distribuições das 19 features entre janelas off-line e em tempo real.
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES_19

    rows = []
    for name in feature_names:
        if name not in df_offline.columns or name not in df_realtime.columns:
            continue
        off_vals = df_offline[name].astype("float64")
        rt_vals = df_realtime[name].astype("float64")

        rows.append(
            {
                "feature": name,
                "offline_mean": float(off_vals.mean()),
                "offline_std": float(off_vals.std()),
                "offline_min": float(off_vals.min()),
                "offline_max": float(off_vals.max()),
                "realtime_mean": float(rt_vals.mean()),
                "realtime_std": float(rt_vals.std()),
                "realtime_min": float(rt_vals.min()),
                "realtime_max": float(rt_vals.max()),
            }
        )

    return pd.DataFrame(rows)
