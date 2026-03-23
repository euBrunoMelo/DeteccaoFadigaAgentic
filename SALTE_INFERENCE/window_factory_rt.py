"""
Versao online do WindowFactory da TEV7 para uso em tempo real.

Atualizado para usar CalibratedFrame (Z-Norm per-subject).

Correcoes aplicadas:
- FIX 1: EAR velocity computada sobre EAR RAW suavizado (median filter kernel=5)
- FIX 2: PERCLOS baseline = mean EAR do best-segment (alinha com FFV5 offline)
- FIX 3: Blink velocity onset = local max PRE-blink no sinal suavizado
- FIX 4: Microsleep filters (median, blink overlap, purity)
- FIX 5: C22 clamp rebaixado de 50->5 EAR/s (limite fisiologico humano)
- FIX 6: perclos_p80_max = pico de rolling 5s (nao max binario -- era sempre 1.0!)
- FIX-RT-2: Z-Score Clamp [-3, +3] em features Z-normed (C32)

Respeita:
- C5:  Z-Norm per subject para EAR/MAR/pose stats
- C13: PERCLOS sobre EAR raw (nunca Z-normalizado)
- C21: Velocidades em EAR/s
- C22: Bounded derived features (blink_vel in [0.01, 5])
- C32: Z-Score Clamp RT -- features Z-normed bounded [-3, +3]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .model_loader import FEATURE_NAMES_19
from .subject_calibrator_rt import CalibratedFrame


@dataclass
class RTWindowConfig:
    fps: int = 30
    window_sec: float = 15.0
    stride_infer: int = 60
    min_valid_ratio: float = 0.80
    # Fator PERCLOS: offline FFV5 usa 0.80 (FHWA P80), mas assume calibracao e
    # operacao no mesmo dominio (motorista dirigindo). No RT com webcam, a
    # calibracao captura pico de alerta (olhos bem abertos) e a operacao e mais
    # relaxada -- gap de ~20% no EAR. Fator 0.65 compensa esse domain shift.
    perclos_factor: float = 0.65
    # FIX-RT-2 (C32): Clamp Z-scores em [-zscore_clamp, +zscore_clamp].
    # Protege contra drift de camera/iluminacao que empurra Z-scores para
    # valores fora do range de treino. +/-3 sigma cobre 99.7% da distribuicao
    # normal; ear_z=-3 ainda e sinal forte de Danger (olho muito fechado vs
    # baseline).
    zscore_clamp: float = 3.0


class OnlineWindowFactory:
    """
    OnlineWindowFactory mantem um buffer de CalibratedFrames e,
    periodicamente, produz uma janela agregada com 19 features (TEV7 Agentic V3).

    Usa:
    - Campos *_znorm para: EAR stats, MAR, head pose
    - Campos *_raw para: EAR velocity, PERCLOS, blinks, microsleeps
    """

    def __init__(self, config: Optional[RTWindowConfig] = None) -> None:
        self.cfg = config or RTWindowConfig()
        self.window_frames = int(self.cfg.window_sec * self.cfg.fps)
        self._buffer: List[CalibratedFrame] = []
        self._since_last_emit = 0

        # FIX 2: baseline PERCLOS externo (P90 do warm-up de 120s)
        self._perclos_baseline_ear: Optional[float] = None

    def set_perclos_baseline(self, ear_baseline: float) -> None:
        """
        Define o baseline EAR para PERCLOS a partir do calibrador.

        FIX 2 alinhado com FFV5 offline: deve receber ear_mean do best-segment
        (nao P90). Threshold final = ear_baseline * 0.8 (FHWA P80).
        """
        self._perclos_baseline_ear = ear_baseline

    @staticmethod
    def _median_filter(arr: np.ndarray, kernel: int = 5) -> np.ndarray:
        """Median filter puro numpy (sem scipy). Remove spikes de 1-2 frames."""
        if len(arr) < kernel:
            return arr.copy()
        half_k = kernel // 2
        padded = np.pad(arr, half_k, mode="edge")
        result = np.empty_like(arr)
        for i in range(len(arr)):
            result[i] = np.median(padded[i : i + kernel])
        return result

    def push(self, frame: CalibratedFrame) -> Optional[Dict[str, float]]:
        """
        Adiciona um CalibratedFrame ao buffer.

        Retorna:
        - Um dicionario {feature_name: valor} quando uma nova janela
          agregada estiver disponivel.
        - None caso contrario.
        """
        self._buffer.append(frame)
        if len(self._buffer) > self.window_frames:
            self._buffer = self._buffer[-self.window_frames:]

        self._since_last_emit += 1
        if (
            len(self._buffer) < self.window_frames
            or self._since_last_emit < self.cfg.stride_infer
        ):
            return None

        self._since_last_emit = 0
        return self._aggregate_current_window()

    def _aggregate_current_window(self) -> Optional[Dict[str, float]]:
        if len(self._buffer) < self.window_frames:
            return None

        window = self._buffer[-self.window_frames:]
        face_mask = np.array([f.face_detected for f in window], dtype=bool)
        valid_ratio = float(face_mask.mean())
        if valid_ratio < self.cfg.min_valid_ratio:
            return None

        # -- Z-Normed arrays (para EAR stats, MAR stats, head pose) --
        ear_z = np.array([f.ear_avg_znorm for f in window], dtype=np.float32)
        mar_z = np.array([f.mar_znorm for f in window], dtype=np.float32)
        pitch_z = np.array([f.head_pitch_znorm for f in window], dtype=np.float32)
        yaw_z = np.array([f.head_yaw_znorm for f in window], dtype=np.float32)
        roll_z = np.array([f.head_roll_znorm for f in window], dtype=np.float32)

        # -- FIX-RT-2: Z-Score Clamp (C32) --------------------------------
        # Protege contra drift de camera/iluminacao que empurra Z-scores
        # para valores fora do range de treino do modelo. ear_z=-3 ainda
        # e forte sinal de Danger; pitch_z=+3 ainda captura head drop.
        # Features raw (PERCLOS, blinks, microsleeps) NAO sao afetadas --
        # operam sobre ear_raw, nunca sobre Z-norm (C13 preservado).
        ZSCORE_CLAMP = self.cfg.zscore_clamp
        ear_z = np.clip(ear_z, -ZSCORE_CLAMP, ZSCORE_CLAMP)
        mar_z = np.clip(mar_z, -ZSCORE_CLAMP, ZSCORE_CLAMP)
        pitch_z = np.clip(pitch_z, -ZSCORE_CLAMP, ZSCORE_CLAMP)
        yaw_z = np.clip(yaw_z, -ZSCORE_CLAMP, ZSCORE_CLAMP)
        roll_z = np.clip(roll_z, -ZSCORE_CLAMP, ZSCORE_CLAMP)
        # ------------------------------------------------------------------

        # -- Raw arrays (para EAR velocity, PERCLOS, blinks) --
        ear_raw = np.array([f.ear_avg_raw for f in window], dtype=np.float32)

        # FIX 1+5: Median filter remove spikes de landmark jitter (1-2 frames)
        # antes de computar velocidades. Sem isso, um salto de 0.18 EAR num
        # frame gera velocity de 5.4 EAR/s e contamina ear_vel_std e blink_vel.
        ear_smooth = self._median_filter(ear_raw)

        feats: Dict[str, float] = {}

        # === Grupo 1: EAR stats (Z-Normed, clampado C32) ===
        feats["ear_mean"] = float(ear_z.mean())
        feats["ear_std"] = float(ear_z.std())
        feats["ear_min"] = float(ear_z.min())

        # === Grupo 2: EAR velocity (RAW suavizado -- FIX 1) ===
        ear_vel_raw = np.diff(ear_smooth, prepend=ear_smooth[0]) * self.cfg.fps
        feats["ear_vel_mean"] = float(ear_vel_raw.mean())
        feats["ear_vel_std"] = float(ear_vel_raw.std())

        # === Grupo 3: MAR (Z-Normed, clampado C32) ===
        feats["mar_mean"] = float(mar_z.mean())

        # === Grupo 4: Head pose (Z-Normed, clampado C32) ===
        feats["pitch_mean"] = float(pitch_z.mean())
        feats["pitch_std"] = float(pitch_z.std())
        feats["yaw_std"] = float(yaw_z.std())
        feats["roll_std"] = float(roll_z.std())

        # === Grupo 5-7: Blink stats (RAW + smooth -- C13, FIX 3+5) ===
        blink_stats = self._compute_blink_stats(ear_raw, ear_smooth)
        feats.update(blink_stats)

        # === Grupo 6: PERCLOS (RAW -- C13, FIX 2) ===
        perclos_stats, micro_stats = self._compute_perclos_and_microsleeps(
            ear_raw, ear_smooth, face_mask
        )
        feats.update(perclos_stats)
        feats.update(micro_stats)  # para overlay/logging; NAO vai no vetor do modelo

        # Validacao: todas as 19 features de modelo devem existir
        missing = [f for f in FEATURE_NAMES_19 if f not in feats]
        if missing:
            raise RuntimeError(
                f"WindowFactoryRT nao preencheu todas as features: "
                f"faltando {missing}"
            )

        return feats

    def _compute_blink_stats(
        self, ear_raw: np.ndarray, ear_smooth: np.ndarray
    ) -> Dict[str, float]:
        """
        Deteccao de blinks usando EAR RAW (C13).

        FIX 3: Blink velocity usa o local max PRE-blink como onset real,
        computado sobre ear_smooth para evitar que jitter de landmark
        gere velocidades espurias.

        FIX 5: Velocities clampadas em [0.01, 5] EAR/s (C22).
        Limite fisiologico: blink humano ~ 2-4 EAR/s.
        """
        fps = self.cfg.fps
        window_sec = len(ear_raw) / max(fps, 1)

        zeros = {
            "blink_count": 0.0,
            "blink_rate_per_min": 0.0,
            "blink_mean_dur_ms": 0.0,
            "blink_closing_vel_mean": 0.0,
            "blink_opening_vel_mean": 0.0,
            "long_blink_pct": 0.0,
            "blink_regularity": 0.0,
        }

        valid_ear = ear_raw[~np.isnan(ear_raw)]
        if len(valid_ear) == 0:
            return zeros

        p10 = float(np.percentile(valid_ear, 10))
        thresh = max(0.7 * float(valid_ear.mean()), p10)

        below = ear_raw < thresh
        segments: List[Tuple[int, int]] = []
        if below.any():
            padded = np.concatenate([[False], below.astype(bool), [False]])
            diff = np.diff(padded.astype(np.int8))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            segments = list(zip(starts.tolist(), ends.tolist()))

        min_blink_frames = 2
        blinks = [(s, e) for (s, e) in segments if (e - s) >= min_blink_frames]

        n_blinks = len(blinks)
        blink_count = float(n_blinks)
        blink_rate = (blink_count / window_sec) * 60.0 if window_sec > 0 else 0.0

        blink_durs_ms: List[float] = []
        closing_vels: List[float] = []
        opening_vels: List[float] = []
        blink_starts: List[int] = []
        long_blinks = 0

        pre_blink_lookback = max(int(fps * 0.2), 3)

        for s, e in blinks:
            seg = ear_raw[s:e]
            if len(seg) == 0:
                continue
            peak_offset = int(np.nanargmin(seg))
            peak_frame = s + peak_offset
            ear_peak = float(ear_smooth[peak_frame])

            # FIX 3+5: onset/offset via ear_smooth (remove jitter de landmark)
            lookback_start = max(0, s - pre_blink_lookback)
            pre_blink_region = ear_smooth[lookback_start : s + 1]
            if len(pre_blink_region) > 0:
                onset_local_idx = int(np.nanargmax(pre_blink_region))
                onset_frame = lookback_start + onset_local_idx
                ear_onset = float(ear_smooth[onset_frame])
            else:
                onset_frame = s
                ear_onset = float(ear_smooth[s])

            post_blink_end = min(len(ear_smooth), e + pre_blink_lookback)
            post_blink_region = ear_smooth[e - 1 : post_blink_end]
            if len(post_blink_region) > 0:
                offset_local_idx = int(np.nanargmax(post_blink_region))
                offset_frame = (e - 1) + offset_local_idx
                ear_offset = float(ear_smooth[offset_frame])
            else:
                offset_frame = e - 1
                ear_offset = float(ear_smooth[e - 1])

            duration_frames = e - s
            duration_ms = duration_frames / max(fps, 1) * 1000.0
            blink_durs_ms.append(duration_ms)

            closing_frames = max(peak_frame - onset_frame, 1)
            closing_vel = abs(ear_onset - ear_peak) / (closing_frames / max(fps, 1))

            opening_frames = max(offset_frame - peak_frame, 1)
            opening_vel = abs(ear_offset - ear_peak) / (opening_frames / max(fps, 1))

            # FIX 5: clamp fisiologico [0.01, 5] EAR/s
            closing_vel = float(np.clip(closing_vel, 0.01, 5.0))
            opening_vel = float(np.clip(opening_vel, 0.01, 5.0))

            closing_vels.append(closing_vel)
            opening_vels.append(opening_vel)

            if duration_ms > 300.0:
                long_blinks += 1

            blink_starts.append(s)

        blink_mean_dur_ms = float(np.mean(blink_durs_ms)) if blink_durs_ms else 0.0
        closing_mean = float(np.mean(closing_vels)) if closing_vels else 0.0
        opening_mean = float(np.mean(opening_vels)) if opening_vels else 0.0
        long_blink_pct = float(long_blinks) / float(n_blinks) if n_blinks > 0 else 0.0

        if len(blink_starts) >= 3:
            ibis = np.diff(sorted(blink_starts))
            blink_reg = float(np.std(ibis) / (np.mean(ibis) + 1e-6))
        else:
            blink_reg = 0.0

        return {
            "blink_count": blink_count,
            "blink_rate_per_min": blink_rate,
            "blink_mean_dur_ms": blink_mean_dur_ms,
            "blink_closing_vel_mean": closing_mean,
            "blink_opening_vel_mean": opening_mean,
            "long_blink_pct": long_blink_pct,
            "blink_regularity": blink_reg,
        }

    def _compute_perclos_and_microsleeps(
        self,
        ear_raw: np.ndarray,
        ear_smooth: np.ndarray,
        face_mask: np.ndarray,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        PERCLOS P80 e microsleeps usando EAR RAW (C13 + C25).

        FIX 2 (alinhado com FFV5 offline): baseline = mean EAR do best-segment
        de calibracao. Threshold = baseline * 0.8 (FHWA P80).
        Fallback se nao calibrado: P90 intra-janela.

        FIX 4: Microsleeps filtrados com median filter, blink overlap, purity.
        """
        valid = face_mask & ~np.isnan(ear_raw)
        valid_ear = ear_raw[valid]
        if len(valid_ear) == 0:
            perclos_stats = {"perclos_p80_mean": 0.0, "perclos_p80_max": 0.0}
            micro_stats = {"microsleep_count": 0.0, "microsleep_total_ms": 0.0}
            return perclos_stats, micro_stats

        # FIX 2: baseline = mean EAR do best-segment (alinha com FFV5 offline)
        if self._perclos_baseline_ear is not None:
            baseline_ear = self._perclos_baseline_ear
        else:
            baseline_ear = float(np.percentile(valid_ear, 90))

        threshold = baseline_ear * self.cfg.perclos_factor

        closed = (ear_raw < threshold) & valid
        closed_f = closed.astype(np.float64)

        fps = self.cfg.fps

        # perclos_p80_mean: fracao de frames fechados na janela inteira
        perclos_mean = float(np.nanmean(closed_f))

        # FIX 6: perclos_p80_max = pico da rolling average (NAO max binario!)
        sub_sec = 5.0
        sub_frames = max(int(sub_sec * fps), 1)
        if len(closed_f) >= sub_frames:
            cs = np.cumsum(closed_f)
            cs = np.insert(cs, 0, 0.0)
            rolling = (cs[sub_frames:] - cs[:-sub_frames]) / sub_frames
            perclos_max = float(np.max(rolling)) if len(rolling) > 0 else perclos_mean
        else:
            perclos_max = perclos_mean
        min_ms = 500.0
        min_frames = int(min_ms / 1000.0 * fps)

        microsleeps = 0
        total_ms = 0.0

        # ear_smooth ja vem pre-computado (FIX 1)
        closed_smooth = (ear_smooth < threshold) & valid

        if closed_smooth.any():
            padded = np.concatenate([[False], closed_smooth.astype(bool), [False]])
            diff = np.diff(padded.astype(np.int8))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]

            # (b) Coletar intervalos de blinks detectados para exclusao
            # Blinks: segmentos curtos (< 500ms) de olho fechado no sinal raw
            blink_intervals = []
            closed_raw = (ear_raw < threshold) & valid
            if closed_raw.any():
                bp = np.concatenate([[False], closed_raw.astype(bool), [False]])
                bd = np.diff(bp.astype(np.int8))
                b_starts = np.where(bd == 1)[0]
                b_ends = np.where(bd == -1)[0]
                for bs, be in zip(b_starts.tolist(), b_ends.tolist()):
                    dur = be - bs
                    if 2 <= dur < min_frames:  # blink: 2 frames ate < 500ms
                        blink_intervals.append((bs, be))

            for s, e in zip(starts.tolist(), ends.tolist()):
                length = e - s
                if length < min_frames:
                    continue

                # (b) Verificar se o segmento e apenas blinks concatenados
                # Se > 50% do segmento se sobrepe com blinks, descartar
                blink_overlap = 0
                for bs, be in blink_intervals:
                    overlap_start = max(s, bs)
                    overlap_end = min(e, be)
                    if overlap_end > overlap_start:
                        blink_overlap += overlap_end - overlap_start

                if length > 0 and blink_overlap / length > 0.5:
                    continue

                # (c) Purity check: no sinal suavizado, pelo menos 80% dos
                # frames devem estar abaixo do threshold
                seg_closed = closed_smooth[s:e]
                purity = float(seg_closed.mean()) if len(seg_closed) > 0 else 0.0
                if purity < 0.80:
                    continue

                microsleeps += 1
                total_ms += length / max(fps, 1) * 1000.0

        perclos_stats = {
            "perclos_p80_mean": perclos_mean,
            "perclos_p80_max": perclos_max,
        }
        micro_stats = {
            "microsleep_count": float(microsleeps),
            "microsleep_total_ms": float(total_ms),
        }
        return perclos_stats, micro_stats