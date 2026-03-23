"""
C29 Boundary Engine — Layer 5 Guardrails baseada em head pose.

Regras de contorno que operam em paralelo ao MLP TEV7, usando features
C29 (head_activity, activity_drop, nodding_energy_hf, yaw_pitch_ratio)
computadas em tempo real a partir dos frames brutos (RTFrameFeatures).

Principio: Zero alteracao no modelo TEV7 — regras sao aditivas.
4 regras:
  R1 — Activity Drop sustentado (override -> Danger)
  R2 — Head Stillness extrema (boost +0.15)
  R3 — Pitch dominando Yaw + nodding energy (boost +0.20)
  R4 — Oscilacao involuntaria com baixa atividade (boost +0.25)

Boosts aditivos com cap em 0.40.  R1 e a unica com override.

Ref: SALTE_C29_Boundary_Rules_Plan.md, C29_Communication_Map.md.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional

import numpy as np


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class C29Config:
    """Thresholds e parametros das regras C29.  Carregavel via JSON."""

    fps: int = 30

    # R1 — Activity Drop Sustentado
    r1_ad_threshold: float = 0.5
    r1_duration_sec: float = 30.0
    r1_warmup_sec: float = 300.0   # 5 min para baseline estabilizar

    # R2 — Head Stillness Extrema
    r2_ha_threshold: float = 0.15
    r2_duration_sec: float = 20.0
    r2_boost: float = 0.15

    # R3 — Pitch Dominando Yaw (possivel nodding)
    r3_ypr_threshold: float = 1.0
    r3_ne_threshold: float = 50.0
    r3_ha_min: float = 0.3          # HA > 0.3  (excluir imobilidade total)
    r3_duration_sec: float = 10.0
    r3_boost: float = 0.20

    # R4 — Oscilacao Involuntaria (nodding + baixa atividade geral)
    r4_ne_threshold: float = 400.0
    r4_ha_threshold: float = 0.5    # HA < 0.5
    r4_yaw_threshold: float = 20.0  # |yaw| < 20 graus
    r4_boost: float = 0.25

    # Geral
    max_boost: float = 0.40
    override_only: bool = False     # --c29-override-only


# ── Alert dataclass ───────────────────────────────────────────────────────────


@dataclass
class C29Alert:
    """Resultado da avaliacao das regras C29."""

    override: bool = False
    boost: float = 0.0
    active_rules: List[str] = field(default_factory=list)
    values: Dict[str, float] = field(default_factory=dict)
    is_warm: bool = False

    @property
    def any_active(self) -> bool:
        return self.override or self.boost > 0


# ── Loader ────────────────────────────────────────────────────────────────────


def load_c29_config(config_path: str | Path) -> C29Config:
    """Carrega thresholds de c29_boundary_config.json.  Fallback: defaults."""
    path = Path(config_path)
    if not path.exists():
        return C29Config()
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    cfg = C29Config()
    for key, val in raw.items():
        if hasattr(cfg, key):
            setattr(cfg, key, type(getattr(cfg, key))(val))
    return cfg


# ── Engine ────────────────────────────────────────────────────────────────────


class C29BoundaryEngine:
    """
    Layer 5 Guardrails — regras de contorno baseadas em head pose C29.

    Consome RTFrameFeatures diretamente (Opcao A — pre-calibracao, ativo
    desde t=0 para R2/R3/R4, desde t=5min para R1).

    Silencioso durante warm-up do calibrador: push_frame() alimenta os
    ring buffers mas evaluate() so produz alertas quando chamado pelo
    pipeline (apos MLP calibrado).

    Ring buffers (RAM ~85 KB):
      motion_2s         60 frames   — head_activity
      pitch_2s          60 frames   — pitch_std p/ YPR
      pitch_5s         150 frames   — trend p/ detrending
      pitch_detrend_2s  60 frames   — NE_hf (var detrended)
      yaw_2s            60 frames   — yaw_std p/ YPR + |yaw| p/ R4
      activity_30s     900 frames   — activity current (AD)
      activity_5min   9000 frames   — activity baseline (AD)
    """

    def __init__(
        self,
        config: Optional[C29Config] = None,
        enabled: bool = True,
        log_path: Optional[str] = None,
    ) -> None:
        self.cfg = config or C29Config()
        self._enabled = enabled
        self._fps = self.cfg.fps

        # ── Ring buffers ──────────────────────────────────────────────
        buf_2s = max(int(2.0 * self._fps), 2)
        buf_5s = max(int(5.0 * self._fps), 2)
        buf_30s = max(int(30.0 * self._fps), 2)
        buf_5min = max(int(300.0 * self._fps), 2)

        self._motion_2s: Deque[float] = deque(maxlen=buf_2s)
        self._pitch_2s: Deque[float] = deque(maxlen=buf_2s)
        self._pitch_5s: Deque[float] = deque(maxlen=buf_5s)
        self._pitch_detrend_2s: Deque[float] = deque(maxlen=buf_2s)
        self._yaw_2s: Deque[float] = deque(maxlen=buf_2s)
        self._activity_30s: Deque[float] = deque(maxlen=buf_30s)
        self._activity_5min: Deque[float] = deque(maxlen=buf_5min)

        # ── Prev-frame state ─────────────────────────────────────────
        self._prev_pitch: float = 0.0
        self._prev_yaw: float = 0.0
        self._prev_roll: float = 0.0
        self._has_prev: bool = False

        # ── Rule duration counters (frames) ───────────────────────────
        self._r1_count: int = 0
        self._r2_count: int = 0
        self._r3_count: int = 0
        self._r4_count: int = 0

        self._total_frames: int = 0

        # ── Cached features ───────────────────────────────────────────
        self._head_activity: float = 0.0
        self._activity_drop: float = 0.0
        self._nodding_energy_hf: float = 0.0
        self._yaw_pitch_ratio: float = 2.0  # neutral default

        # ── Logging ───────────────────────────────────────────────────
        self._log_file = None
        if log_path is not None:
            self._log_file = open(log_path, "a", encoding="utf-8")

        if enabled:
            print(f"[c29] C29BoundaryEngine initialized (fps={self._fps})")
            if self.cfg.override_only:
                print("[c29] Mode: override-only (R1 only, no boosts)")

    # ── Public API ────────────────────────────────────────────────────────

    def push_frame(self, feats) -> None:
        """
        Alimenta um frame.  Atualiza ring buffers.

        Args:
            feats: objeto com head_pitch, head_yaw, head_roll, face_detected
                   (RTFrameFeatures ou qualquer duck-type compativel).
        """
        if not self._enabled:
            return

        self._total_frames += 1

        if not feats.face_detected:
            self._r2_count = 0  # R2 requer face_detected=True
            return

        pitch = feats.head_pitch
        yaw = feats.head_yaw
        roll = feats.head_roll

        # ── Frame-level motion deltas ─────────────────────────────────
        if self._has_prev:
            total_motion = (
                abs(pitch - self._prev_pitch)
                + abs(yaw - self._prev_yaw)
                + abs(roll - self._prev_roll)
            )
        else:
            total_motion = 0.0

        self._prev_pitch = pitch
        self._prev_yaw = yaw
        self._prev_roll = roll
        self._has_prev = True

        # ── Push to ring buffers ──────────────────────────────────────
        self._motion_2s.append(total_motion)
        self._pitch_2s.append(pitch)
        self._pitch_5s.append(pitch)
        self._yaw_2s.append(yaw)

        # Detrended pitch: pitch - trend_5s
        pitch_trend = float(np.mean(self._pitch_5s))
        self._pitch_detrend_2s.append(pitch - pitch_trend)

        # ── head_activity = log(1 + mean(motion_2s)) ─────────────────
        if len(self._motion_2s) >= 2:
            ha = float(np.log1p(np.mean(self._motion_2s)))
        else:
            ha = 0.0
        self._head_activity = ha
        self._activity_30s.append(ha)
        self._activity_5min.append(ha)

        # ── Compute derived features ──────────────────────────────────
        self._compute_features()

        # ── Update rule duration counters ─────────────────────────────
        self._update_rule_counters(yaw)

    def evaluate(self) -> C29Alert:
        """
        Avalia regras C29 e retorna C29Alert.

        Chamado apos MLP + G1-G3, quando window_feats e produzido.
        """
        if not self._enabled:
            return C29Alert()

        cfg = self.cfg
        warmup_frames = int(cfg.r1_warmup_sec * self._fps)

        alert = C29Alert()
        alert.is_warm = self._total_frames >= warmup_frames
        alert.values = {
            "head_activity": round(self._head_activity, 4),
            "activity_drop": round(self._activity_drop, 4),
            "nodding_energy_hf": round(self._nodding_energy_hf, 4),
            "yaw_pitch_ratio": round(self._yaw_pitch_ratio, 4),
        }

        # R1: Override -> Danger (requer warm-up de 5 min)
        r1_dur_frames = int(cfg.r1_duration_sec * self._fps)
        if self._r1_count >= r1_dur_frames:
            alert.override = True
            alert.active_rules.append("R1")

        if cfg.override_only:
            return alert

        boost = 0.0

        # R2: Boost +0.15
        r2_dur_frames = int(cfg.r2_duration_sec * self._fps)
        if self._r2_count >= r2_dur_frames:
            boost += cfg.r2_boost
            alert.active_rules.append("R2")

        # R3: Boost +0.20
        r3_dur_frames = int(cfg.r3_duration_sec * self._fps)
        if self._r3_count >= r3_dur_frames:
            boost += cfg.r3_boost
            alert.active_rules.append("R3")

        # R4: Boost +0.25 (instantaneo — dispara no 1o frame)
        if self._r4_count >= 1:
            boost += cfg.r4_boost
            alert.active_rules.append("R4")

        alert.boost = min(boost, cfg.max_boost)
        return alert

    def log_alert(
        self,
        alert: C29Alert,
        mlp_label: str,
        mlp_prob: float,
        final_label: str,
        final_prob: float,
    ) -> None:
        """Loga alerta C29 em JSONL (--c29-log)."""
        if self._log_file is None or not alert.any_active:
            return
        entry = {
            "timestamp": time.time(),
            "rules": alert.active_rules,
            "action": (
                "OVERRIDE_DANGER" if alert.override
                else f"BOOST_{alert.boost:.2f}"
            ),
            "values": alert.values,
            "is_warm": alert.is_warm,
            "mlp_label": mlp_label,
            "mlp_prob": round(mlp_prob, 4),
            "final_label": final_label,
            "final_prob": round(final_prob, 4),
        }
        self._log_file.write(json.dumps(entry) + "\n")
        self._log_file.flush()

    def reset_long_buffers(self) -> None:
        """
        Reseta buffers longos apos auto-recalibracao R3.
        Mantem janelas curtas (motion_2s, pitch_*, yaw_2s).
        Ref: C29_Communication_Map secao 6.3.
        """
        self._activity_5min.clear()
        self._activity_30s.clear()
        self._r1_count = 0
        print("[c29] Long buffers reset (auto-recalibration)")

    def close(self) -> None:
        """Fecha log file."""
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def features(self) -> Dict[str, float]:
        """Features C29 atuais (debug/logging)."""
        return {
            "head_activity": self._head_activity,
            "activity_drop": self._activity_drop,
            "nodding_energy_hf": self._nodding_energy_hf,
            "yaw_pitch_ratio": self._yaw_pitch_ratio,
        }

    @property
    def is_warm(self) -> bool:
        """True apos 5 min (R1 disponivel)."""
        return self._total_frames >= int(self.cfg.r1_warmup_sec * self._fps)

    @property
    def total_frames(self) -> int:
        return self._total_frames

    # ── Internal ──────────────────────────────────────────────────────────

    def _compute_features(self) -> None:
        """Computa NE_hf, AD e YPR a partir dos buffers."""
        min_samples = 10

        # nodding_energy_hf = var(pitch detrended over 2s)
        if len(self._pitch_detrend_2s) >= min_samples:
            self._nodding_energy_hf = float(np.var(self._pitch_detrend_2s))
        else:
            self._nodding_energy_hf = 0.0

        # activity_drop = 1 - (current_30s / baseline_5min)
        warmup_frames = int(self.cfg.r1_warmup_sec * self._fps)
        if (
            self._total_frames >= warmup_frames
            and len(self._activity_5min) >= warmup_frames
        ):
            act_cur = (
                float(np.mean(self._activity_30s))
                if self._activity_30s else 0.0
            )
            act_base = (
                float(np.mean(self._activity_5min))
                if self._activity_5min else 1e-8
            )
            self._activity_drop = 1.0 - (act_cur / (act_base + 1e-8))
        else:
            self._activity_drop = 0.0

        # yaw_pitch_ratio = yaw_std / pitch_std
        if (
            len(self._pitch_2s) >= min_samples
            and len(self._yaw_2s) >= min_samples
        ):
            pitch_std = float(np.std(self._pitch_2s))
            yaw_std = float(np.std(self._yaw_2s))
            self._yaw_pitch_ratio = yaw_std / (pitch_std + 1e-6)
        else:
            self._yaw_pitch_ratio = 2.0

    def _update_rule_counters(self, current_yaw: float) -> None:
        """Atualiza contadores de duracao de cada regra."""
        cfg = self.cfg
        warmup_frames = int(cfg.r1_warmup_sec * self._fps)

        # R1: AD > threshold sustentado (requer warm-up 5 min)
        if (
            self._total_frames >= warmup_frames
            and self._activity_drop > cfg.r1_ad_threshold
        ):
            self._r1_count += 1
        else:
            self._r1_count = 0

        # R2: HA < threshold sustentado (face_detected ja checado)
        if self._head_activity < cfg.r2_ha_threshold:
            self._r2_count += 1
        else:
            self._r2_count = 0

        # R3: YPR < thresh + NE_hf > thresh + HA > thresh
        r3_active = (
            self._yaw_pitch_ratio < cfg.r3_ypr_threshold
            and self._nodding_energy_hf > cfg.r3_ne_threshold
            and self._head_activity > cfg.r3_ha_min
        )
        self._r3_count = self._r3_count + 1 if r3_active else 0

        # R4: NE_hf > thresh + HA < thresh + |yaw| < thresh
        r4_active = (
            self._nodding_energy_hf > cfg.r4_ne_threshold
            and self._head_activity < cfg.r4_ha_threshold
            and abs(current_yaw) < cfg.r4_yaw_threshold
        )
        self._r4_count = self._r4_count + 1 if r4_active else 0
