"""
Calibração per-subject em tempo real (C5 + C6-V2).

Implementa a mesma lógica do SubjectCalibrator offline:
- Fase de warm-up: coleta frames por `search_sec` segundos (padrão 120s).
- Ao final do warm-up, encontra o segmento de `baseline_sec` segundos
  (padrão 30s) com maior EAR médio (constraint C6-V2).
- Calcula baseline (mean, std) para EAR, MAR e head pose.
- Após calibração, Z-Normaliza cada frame em tempo real.

Referências:
- C5: Z-Norm per subject (nunca global)
- C6-V2: Calibração pelo segmento de maior EAR nos primeiros 120s
- C13: PERCLOS sobre EAR raw (nunca Z-normalizado)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from .feature_extractor_rt import RTFrameFeatures
except ImportError:
    from feature_extractor_rt import RTFrameFeatures


@dataclass
class CalibrationConfig:
    """Configuração da calibração per-subject."""

    fps: int = 30
    search_sec: float = 120.0       # Janela de busca (C6-V2: primeiros 120s)
    baseline_sec: float = 30.0      # Tamanho do segmento de baseline
    stride_sec: float = 1.0         # Stride para busca do melhor segmento
    min_face_ratio: float = 0.90    # Mínimo de face detectada no segmento
    fallback_to_first: bool = True  # Se nenhum segmento válido, usa primeiros 30s

    # Per-feature min_std fisiológicos.
    # O offline (tev7.py) usa 1e-8, mas os vídeos de motoristas têm variação
    # natural (blinks, olhar espelhos, etc). No warm-up RT a pessoa fica parada,
    # produzindo sigmas minúsculos que explodem a Z-norm.
    # Valores baseados na variação mínima esperada em 30s de operação normal.
    ear_min_std: float = 0.015      # EAR varia ±0.015 só com ritmo de blinks
    mar_min_std: float = 0.01       # MAR variação mínima
    pose_min_std: float = 2.0       # Head pose (graus) — sway natural ≥ ±2°


@dataclass
class SubjectBaseline:
    """Resultado da calibração de um sujeito."""

    ear_mean: float
    ear_std: float
    mar_mean: float
    mar_std: float
    pitch_mean: float
    pitch_std: float
    yaw_mean: float
    yaw_std: float
    roll_mean: float
    roll_std: float
    ear_p90_raw: float = 0.0    # P90 do EAR raw no warm-up (para PERCLOS — C13)
    is_valid: bool = True
    segment_start: int = 0
    segment_end: int = 0


@dataclass
class CalibratedFrame:
    """
    Frame com sinais raw E Z-normalizados.

    - Campos `*_raw`: valores originais do MediaPipe (para PERCLOS, C13)
    - Campos `*_znorm`: Z-normalizados per-subject (para agregação de janela)
    """

    timestamp_ms: float
    frame_idx: int
    face_detected: bool

    # Raw (para PERCLOS e blink detection — C13)
    ear_avg_raw: float
    ear_l_raw: float
    ear_r_raw: float
    mar_raw: float
    head_pitch_raw: float
    head_yaw_raw: float
    head_roll_raw: float

    # Z-Normalized (para agregação de features do modelo)
    ear_avg_znorm: float
    mar_znorm: float
    head_pitch_znorm: float
    head_yaw_znorm: float
    head_roll_znorm: float


class RTSubjectCalibrator:
    """
    Calibrador per-subject em tempo real.

    Ciclo de vida:
    1. `push(frame_feats)` durante warm-up → retorna None
    2. Após `search_sec` segundos, `is_calibrated` vira True
    3. `calibrate(frame_feats)` transforma frames raw → CalibratedFrame

    Se `force_calibrate()` for chamado antes do warm-up terminar,
    usa os frames coletados até o momento.
    """

    def __init__(self, config: Optional[CalibrationConfig] = None) -> None:
        self.cfg = config or CalibrationConfig()
        self._warmup_buffer: List[RTFrameFeatures] = []
        self._baseline: Optional[SubjectBaseline] = None
        self._search_frames = int(self.cfg.search_sec * self.cfg.fps)

    @property
    def is_calibrated(self) -> bool:
        return self._baseline is not None

    @property
    def baseline(self) -> Optional[SubjectBaseline]:
        return self._baseline

    @property
    def warmup_progress(self) -> float:
        """Progresso do warm-up (0.0 a 1.0)."""
        if self._baseline is not None:
            return 1.0
        return min(len(self._warmup_buffer) / max(self._search_frames, 1), 1.0)

    def push(self, frame_feats: RTFrameFeatures) -> Optional[CalibratedFrame]:
        """
        Alimenta um frame durante o warm-up.

        Retorna:
        - None enquanto estiver em warm-up
        - CalibratedFrame após calibração (e para todos os frames subsequentes)
        """
        if self._baseline is not None:
            # Já calibrado → transforma direto
            return self._apply_znorm(frame_feats)

        self._warmup_buffer.append(frame_feats)

        if len(self._warmup_buffer) >= self._search_frames:
            self._compute_baseline()
            return self._apply_znorm(frame_feats)

        return None

    def force_calibrate(self) -> bool:
        """
        Força calibração com os frames coletados até agora.
        Útil se o operador precisa começar antes de 120s.

        Retorna True se conseguiu calibrar, False se dados insuficientes.
        """
        min_frames = int(self.cfg.baseline_sec * self.cfg.fps * 0.5)
        if len(self._warmup_buffer) < min_frames:
            return False
        self._compute_baseline()
        return self._baseline is not None

    def calibrate(self, frame_feats: RTFrameFeatures) -> CalibratedFrame:
        """
        Aplica Z-Norm a um frame. Requer calibração prévia.
        Levanta RuntimeError se não calibrado.
        """
        if self._baseline is None:
            raise RuntimeError(
                "Calibrador não está calibrado. "
                "Use push() durante warm-up ou force_calibrate()."
            )
        return self._apply_znorm(frame_feats)

    def _compute_baseline(self) -> None:
        """
        Implementa C6-V2: encontra o segmento de baseline_sec com maior
        EAR médio dentro dos frames coletados.
        """
        buf = self._warmup_buffer
        fps = self.cfg.fps
        baseline_frames = int(self.cfg.baseline_sec * fps)
        stride_frames = max(int(self.cfg.stride_sec * fps), 1)

        if len(buf) < baseline_frames:
            # Fallback: usa tudo que tem
            baseline_frames = len(buf)

        best_ear = -1.0
        best_start = 0

        for start in range(0, len(buf) - baseline_frames + 1, stride_frames):
            segment = buf[start : start + baseline_frames]

            # Checa face_ratio
            face_ratio = sum(1 for f in segment if f.face_detected) / len(segment)
            if face_ratio < self.cfg.min_face_ratio:
                continue

            # EAR médio dos frames com face
            ear_vals = [f.ear_avg for f in segment if f.face_detected]
            if not ear_vals:
                continue
            ear_mean = float(np.mean(ear_vals))

            if ear_mean > best_ear:
                best_ear = ear_mean
                best_start = start

        # Se nenhum segmento válido e fallback habilitado, usa primeiros N frames
        if best_ear < 0 and self.cfg.fallback_to_first:
            best_start = 0

        segment = buf[best_start : best_start + baseline_frames]
        valid_frames = [f for f in segment if f.face_detected]

        if not valid_frames:
            # Último recurso: pega tudo que tem face
            valid_frames = [f for f in buf if f.face_detected]
            if not valid_frames:
                # Sem nenhuma face → baseline inválido
                self._baseline = SubjectBaseline(
                    ear_mean=0.3, ear_std=0.05,
                    mar_mean=0.1, mar_std=0.05,
                    pitch_mean=0.0, pitch_std=5.0,
                    yaw_mean=0.0, yaw_std=5.0,
                    roll_mean=0.0, roll_std=5.0,
                    is_valid=False,
                    segment_start=0,
                    segment_end=0,
                )
                return

        ears = np.array([f.ear_avg for f in valid_frames])
        mars = np.array([f.mar for f in valid_frames])
        pitches = np.array([f.head_pitch for f in valid_frames])
        yaws = np.array([f.head_yaw for f in valid_frames])
        rolls = np.array([f.head_roll for f in valid_frames])

        # P90 do EAR raw excluindo blinks (para debug/referência).
        # Nota: PERCLOS baseline agora usa ear_mean (alinhado com FFV5 offline).
        all_valid = [f for f in buf if f.face_detected]
        all_ears = np.array([f.ear_avg for f in all_valid])
        if len(all_ears) > 0:
            # Excluir blink frames: EAR < 70% da mediana são troughs de blink
            median_ear = float(np.median(all_ears))
            blink_thresh = median_ear * 0.7
            open_ears = all_ears[all_ears >= blink_thresh]
            if len(open_ears) > 10:
                ear_p90 = float(np.percentile(open_ears, 90))
            else:
                ear_p90 = float(np.percentile(all_ears, 90))
        else:
            ear_p90 = float(ears.mean())

        self._baseline = SubjectBaseline(
            ear_mean=float(ears.mean()),
            ear_std=max(float(ears.std()), self.cfg.ear_min_std),
            mar_mean=float(mars.mean()),
            mar_std=max(float(mars.std()), self.cfg.mar_min_std),
            pitch_mean=float(pitches.mean()),
            pitch_std=max(float(pitches.std()), self.cfg.pose_min_std),
            yaw_mean=float(yaws.mean()),
            yaw_std=max(float(yaws.std()), self.cfg.pose_min_std),
            roll_mean=float(rolls.mean()),
            roll_std=max(float(rolls.std()), self.cfg.pose_min_std),
            ear_p90_raw=ear_p90,
            is_valid=True,
            segment_start=best_start,
            segment_end=best_start + baseline_frames,
        )

    def _apply_znorm(self, f: RTFrameFeatures) -> CalibratedFrame:
        """Aplica Z-Norm usando o baseline calculado."""
        b = self._baseline
        assert b is not None

        if not f.face_detected:
            return CalibratedFrame(
                timestamp_ms=f.timestamp_ms,
                frame_idx=f.frame_idx,
                face_detected=False,
                ear_avg_raw=0.0, ear_l_raw=0.0, ear_r_raw=0.0,
                mar_raw=0.0,
                head_pitch_raw=0.0, head_yaw_raw=0.0, head_roll_raw=0.0,
                ear_avg_znorm=0.0, mar_znorm=0.0,
                head_pitch_znorm=0.0, head_yaw_znorm=0.0, head_roll_znorm=0.0,
            )

        return CalibratedFrame(
            timestamp_ms=f.timestamp_ms,
            frame_idx=f.frame_idx,
            face_detected=True,
            # Raw preservados (para PERCLOS — C13)
            ear_avg_raw=f.ear_avg,
            ear_l_raw=f.ear_l,
            ear_r_raw=f.ear_r,
            mar_raw=f.mar,
            head_pitch_raw=f.head_pitch,
            head_yaw_raw=f.head_yaw,
            head_roll_raw=f.head_roll,
            # Z-Normalized (para modelo — C5)
            ear_avg_znorm=(f.ear_avg - b.ear_mean) / b.ear_std,
            mar_znorm=(f.mar - b.mar_mean) / b.mar_std,
            head_pitch_znorm=(f.head_pitch - b.pitch_mean) / b.pitch_std,
            head_yaw_znorm=(f.head_yaw - b.yaw_mean) / b.yaw_std,
            head_roll_znorm=(f.head_roll - b.roll_mean) / b.roll_std,
        )
