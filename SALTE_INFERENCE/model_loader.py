"""
Carregamento do modelo MLP TEV7 Agentic V3 e artefatos de inferência.

V2: ONNX-only, sem PyTorch. Scaler via JSON (inference_config.json).
Suporta 19 features (Grupo 8 microsleep removido por ablation).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Tuple, Union

import numpy as np

try:
    import onnxruntime as ort

    HAS_ONNX = True
except ImportError:
    ort = None  # type: ignore[assignment]
    HAS_ONNX = False


# Ordem canônica das 19 features (TEV7 Agentic V3, sem microsleep)
FEATURE_NAMES_19: List[str] = [
    "ear_mean",
    "ear_std",
    "ear_min",
    "ear_vel_mean",
    "ear_vel_std",
    "mar_mean",
    "pitch_mean",
    "pitch_std",
    "yaw_std",
    "roll_std",
    "blink_count",
    "blink_rate_per_min",
    "blink_mean_dur_ms",
    "perclos_p80_mean",
    "perclos_p80_max",
    "blink_closing_vel_mean",
    "blink_opening_vel_mean",
    "long_blink_pct",
    "blink_regularity",
]


# ── Inference config (single source of truth) ────────────────────────────────


@dataclass
class InferenceConfig:
    """Configuração de inferência carregada de inference_config.json."""

    threshold: float
    feature_names: List[str]
    n_features: int
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    scaler_indices: List[int]


def load_inference_config(
    config_path: Union[Path, str] = "models/inference_config.json",
) -> InferenceConfig:
    """
    Carrega inference_config.json com scaler e metadados.

    Retorna InferenceConfig com threshold, feature_names, scaler_mean/scale/indices.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"inference_config não encontrado em {path}")

    with path.open("r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    return InferenceConfig(
        threshold=float(cfg["threshold"]),
        feature_names=list(cfg["feature_names"]),
        n_features=int(cfg["n_features"]),
        scaler_mean=np.array(cfg["scaler_mean"], dtype=np.float32),
        scaler_scale=np.array(cfg["scaler_scale"], dtype=np.float32),
        scaler_indices=list(cfg["scaler_indices"]),
    )


def scale_features(
    raw_19: np.ndarray, config: InferenceConfig
) -> np.ndarray:
    """
    C11: Selective scaling — features contínuas normalizadas, passthrough intactas.

    Aplica (x - mean) / scale apenas nos índices em config.scaler_indices.
    Aceita vetor (19,) ou batch (N, 19).
    """
    scaled = np.array(raw_19, dtype=np.float32, copy=True)
    idx = config.scaler_indices
    if scaled.ndim == 1:
        scaled[idx] = (scaled[idx] - config.scaler_mean) / config.scaler_scale
    else:
        scaled[:, idx] = (
            scaled[:, idx] - config.scaler_mean
        ) / config.scaler_scale
    return scaled


# ── ONNX Runtime wrapper ────────────────────────────────────────────────────


class ONNXModelWrapper:
    """Wrapper for onnxruntime.InferenceSession with a numpy-only interface."""

    def __init__(self, session: Any) -> None:
        self.session = session
        self.input_name = session.get_inputs()[0].name

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        return self.session.run(None, {self.input_name: x})[0]


# ── Model loading ─────────────────────────────────────────────────────────────


def load_best_model(
    checkpoint_path: Union[Path, str],
    config_path: Union[Path, str, None] = None,
) -> Tuple[ONNXModelWrapper, InferenceConfig]:
    """
    Carrega best_model.onnx e inference_config.json.

    Retorna (ONNXModelWrapper, InferenceConfig).
    config_path: se None, usa o mesmo diretório do checkpoint com nome inference_config.json
    """
    if not HAS_ONNX:
        raise RuntimeError(
            "onnxruntime não está instalado. "
            "Instale com `pip install onnxruntime`."
        )

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado em {path}")

    if path.suffix != ".onnx":
        raise ValueError(f"Formato esperado: .onnx. Recebido: {path.suffix}")

    # Config no mesmo dir do modelo se não especificado
    if config_path is None:
        config_path = path.parent / "inference_config.json"
    config = load_inference_config(config_path)

    data_file = path.parent / (path.name + ".data")
    if data_file.exists():
        print(
            f"[model] External data: {data_file.name} "
            f"({data_file.stat().st_size} bytes)"
        )

    session = ort.InferenceSession(str(path))
    print(f"[model] Loaded ONNX: {path} (19 features)")
    return ONNXModelWrapper(session), config


# ── Prediction ──────────────────────────────────────────────────────────────


def predict_fatigue(
    features: Union[Iterable[float], np.ndarray],
    model: ONNXModelWrapper,
    config: InferenceConfig,
    *,
    threshold_override: float | None = None,
) -> Tuple[float, str]:
    """
    Inferência de fadiga para UMA janela agregada (19 features).

    features: vetor de 19 valores na ordem config.feature_names.
    threshold_override: opcional; se None, usa config.threshold.
    """
    arr = np.asarray(list(features), dtype=np.float32)
    if arr.shape != (config.n_features,):
        raise ValueError(
            f"features deve ter shape ({config.n_features},), mas veio {arr.shape}. "
            f"Garanta que está usando a ordem de config.feature_names."
        )

    logits = model(arr.reshape(1, -1))
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    prob_danger = float(probs[0, 1])

    thresh = threshold_override if threshold_override is not None else config.threshold
    label = "Danger" if prob_danger >= thresh else "Safe"
    return prob_danger, label
