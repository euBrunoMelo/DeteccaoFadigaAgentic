"""
Extrator de features em tempo real inspirado na FeatureFactoryV5.

V2: Suporta ONNXFaceMeshBackend (BlazeFace + FaceMesh ONNX) para RPi 5
sem dependência do pacote mediapipe.

V2.1 (FIX-RT-1): HeadPoseSanitizer — porta causal do sanitize_head_pose()
da FFV5 (linhas 684-720) para operação frame-a-frame com ring buffers.
Corrige PnP flip (pitch_std=131° → <15°).
Ref: FeatureFactoryV4.md §4.2, SALTE_COMPLETE_HISTORY Bug #2, FFV5 C18.

Dependências: OpenCV, numpy, onnxruntime
"""

from __future__ import annotations

from collections import deque  # NOVO — FIX-RT-1
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Protocol, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort

    HAS_ONNX = True
except ImportError:
    ort = None  # type: ignore[assignment]
    HAS_ONNX = False


class LandmarkBackend(Protocol):
    """
    Interface mínima para um backend de landmarks.

    A implementação deve, para cada frame, devolver:
      - `landmarks`: array [N, 2] em coordenadas normalizadas (0–1).
      - `has_face`: bool indicando se uma face foi detectada.
    """

    def process(
        self, frame_bgr: np.ndarray
    ) -> Tuple[Optional[np.ndarray], bool]:
        ...


@dataclass
class RTExtractorConfig:
    fps: int = 30
    max_interp_gap_ms: float = 300.0


@dataclass
class RTFrameFeatures:
    timestamp_ms: float
    frame_idx: int
    ear_l: float
    ear_r: float
    ear_avg: float
    ear_velocity: float
    mar: float
    head_pitch: float
    head_yaw: float
    head_roll: float
    face_detected: bool


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def compute_ear_from_landmarks(
    landmarks: np.ndarray, eye_indices: Iterable[int]
) -> float:
    """EAR clássico (Soukupová & Čech). `landmarks` em coords normalizadas."""
    pts = np.asarray([landmarks[i] for i in eye_indices], dtype=np.float64)
    v1 = _dist(pts[1], pts[5])
    v2 = _dist(pts[2], pts[4])
    hz = _dist(pts[0], pts[3])
    return (v1 + v2) / (2.0 * hz) if hz > 1e-6 else 0.0


# índices de grupos de landmarks compatíveis com FFV5 (468-landmark mesh)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

MOUTH_UPPER = [82, 13, 312]
MOUTH_LOWER = [87, 14, 317]
MOUTH_CORNERS = [78, 308]

POSE_LANDMARKS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}

FACE_3D_MODEL = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, -63.6, -12.5],
        [-43.3, 32.7, -26.0],
        [43.3, 32.7, -26.0],
        [-28.9, -28.9, -24.1],
        [28.9, -28.9, -24.1],
    ],
    dtype=np.float64,
)


def compute_mar_from_landmarks(landmarks: np.ndarray) -> float:
    """MAR aproximado em coordenadas normalizadas, espelhando a fórmula da FFV5."""

    def _pt(idx: int) -> np.ndarray:
        return np.asarray(landmarks[idx], dtype=np.float64)

    v1 = _dist(_pt(MOUTH_UPPER[0]), _pt(MOUTH_LOWER[0]))
    v2 = _dist(_pt(MOUTH_UPPER[1]), _pt(MOUTH_LOWER[1]))
    v3 = _dist(_pt(MOUTH_UPPER[2]), _pt(MOUTH_LOWER[2]))
    hz = _dist(_pt(MOUTH_CORNERS[0]), _pt(MOUTH_CORNERS[1]))
    return (v1 + v2 + v3) / (3.0 * hz) if hz > 1e-6 else 0.0


def compute_head_pose_from_landmarks(
    landmarks: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> Tuple[float, float, float]:
    """Head pose aproximado via solvePnP, inspirado na implementação da FFV5."""
    img_pts = np.array(
        [
            [landmarks[idx][0] * frame_width, landmarks[idx][1] * frame_height]
            for idx in POSE_LANDMARKS.values()
        ],
        dtype=np.float64,
    )

    fl = float(frame_width)
    cam_matrix = np.array(
        [[fl, 0, frame_width / 2.0], [0, fl, frame_height / 2.0], [0, 0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, _ = cv2.solvePnP(
        FACE_3D_MODEL,
        img_pts,
        cam_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return 0.0, 0.0, 0.0

    R, _ = cv2.Rodrigues(rvec)
    sy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))

    if sy > 1e-6:
        pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        yaw = np.degrees(np.arctan2(-R[2, 0], sy))
        roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    else:
        pitch = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        yaw = np.degrees(np.arctan2(-R[2, 0], sy))
        roll = 0.0

    return float(pitch), float(yaw), float(roll)


# ══════════════════════════════════════════════════════════════════════════════
# FIX-RT-1: HeadPoseSanitizer
# ══════════════════════════════════════════════════════════════════════════════
#
# Porta causal do sanitize_head_pose() da FFV5 (ffv5.py, C18) para operação
# frame-a-frame com ring buffers.
#
# Três passos idênticos ao offline:
#   1. Flip unwrap    — |pitch| > 90° → correção de ambiguidade PnP
#   2. Median filter  — ring buffer k=5, causal (sem lookahead)
#   3. Spike removal  — |Δ| > 30°/frame → hold last valid (max 3 frames)
#
# Diferenças RT vs Offline (FFV5):
#   - Median: rolling(k=5, center=True) bidirecional → deque(maxlen=5) causal
#   - Spike:  np.diff + interpolação linear → comparação frame anterior, hold
#   - NaN:    pd.interpolate(limit=3) → hold last valid (max 3 frames)
#
# Parâmetros idênticos à FFV5:
#   - head_flip_threshold_deg  = 90.0  (cfg.head_flip_threshold_deg)
#   - head_median_kernel       = 5     (cfg.head_median_kernel)
#   - head_spike_threshold_deg = 30.0  (cfg.head_spike_threshold_deg)
#
# Ref: FeatureFactoryV4.md §4.2, SALTE_COMPLETE_HISTORY Bug #2.
# ══════════════════════════════════════════════════════════════════════════════


class HeadPoseSanitizer:
    """
    FIX-RT-1: Porta sanitize_head_pose() da FFV5 para operação causal.

    Três passos idênticos ao offline:
      1. Flip unwrap    — |pitch| > flip_thresh → correção de ambiguidade PnP
      2. Median filter  — ring buffer k=median_k, causal (sem lookahead)
      3. Spike removal  — |Δ| > spike_thresh/frame → hold last valid

    Parâmetros idênticos à FFV5 (ffv5.py linhas 106-108).
    Ref: FeatureFactoryV4.md §4.2, SALTE_COMPLETE_HISTORY Bug #2.
    """

    def __init__(
        self,
        flip_thresh: float = 90.0,
        median_k: int = 5,
        spike_thresh: float = 30.0,
    ) -> None:
        self._flip_thresh = flip_thresh
        self._median_k = median_k
        self._spike_thresh = spike_thresh

        # Ring buffers para median causal (sem lookahead)
        self._pitch_buf: deque = deque(maxlen=median_k)
        self._yaw_buf: deque = deque(maxlen=median_k)
        self._roll_buf: deque = deque(maxlen=median_k)

        # Last valid para spike hold
        self._prev: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._hold_count: int = 0

    def sanitize(
        self, pitch: float, yaw: float, roll: float
    ) -> Tuple[float, float, float]:
        """
        Aplica os 3 passos de sanitização causal.

        Deve ser chamado uma vez por frame, imediatamente após solvePnP
        e antes de emitir RTFrameFeatures.

        Returns:
            (pitch_clean, yaw_clean, roll_clean) em graus.
        """
        # ── Step 1: Flip unwrap (idêntico à FFV5, frame-level) ──
        p, y, r = self._flip_unwrap(pitch, yaw, roll)

        # ── Step 2: Median filter (causal, sem lookahead) ────────
        self._pitch_buf.append(p)
        self._yaw_buf.append(y)
        self._roll_buf.append(r)
        p = float(np.median(self._pitch_buf))
        y = float(np.median(self._yaw_buf))
        r = float(np.median(self._roll_buf))

        # ── Step 3: Spike removal (hold last valid, max 3) ──────
        p, y, r = self._spike_check(p, y, r)

        return p, y, r

    def _flip_unwrap(
        self, p: float, y: float, r: float
    ) -> Tuple[float, float, float]:
        """
        Step 1: Corrige ambiguidade PnP quando |pitch| > flip_thresh.

        Lógica idêntica à FFV5:
          - pitch flipped:  sign(p)*180 - p
          - roll flipped:   sign(r)*180 - r
          - yaw offset:     y + (-180 se y>0 else +180)
        Seguido de unwrap de yaw/roll para [-90, +90].
        """
        if abs(p) > self._flip_thresh:
            p = np.sign(p) * 180.0 - p
            r = np.sign(r) * 180.0 - r
            y = y + (-180.0 if y > 0 else 180.0)

        # Unwrap yaw/roll para [-90, +90] (FFV5 Step 1b)
        if y > 90:
            y -= 180
        elif y < -90:
            y += 180
        if r > 90:
            r -= 180
        elif r < -90:
            r += 180

        return float(p), float(y), float(r)

    def _spike_check(
        self, p: float, y: float, r: float
    ) -> Tuple[float, float, float]:
        """
        Step 3: Spike removal causal.

        Se qualquer eixo muda > spike_thresh num frame, mantém o valor
        anterior por até 3 frames consecutivos. Após 3 holds, aceita
        o novo valor (equivalente ao limit=3 da interpolação offline).
        """
        pp, py, pr = self._prev
        dp = abs(p - pp)
        dy = abs(y - py)
        dr = abs(r - pr)
        is_spike = (
            dp > self._spike_thresh
            or dy > self._spike_thresh
            or dr > self._spike_thresh
        )

        if is_spike and self._hold_count < 3:
            self._hold_count += 1
            return pp, py, pr  # hold last valid

        # Aceita novo valor (ou hold expirou após 3 frames)
        self._hold_count = 0
        self._prev = (p, y, r)
        return p, y, r

    def reset(self) -> None:
        """Reseta estado interno. Chamar ao trocar de vídeo/sessão."""
        self._pitch_buf.clear()
        self._yaw_buf.clear()
        self._roll_buf.clear()
        self._prev = (0.0, 0.0, 0.0)
        self._hold_count = 0


# ══════════════════════════════════════════════════════════════════════════════


class RealTimeFeatureExtractor:
    """
    Extrator de features em tempo real aproximado da FFV5.

    V2.1: Integra HeadPoseSanitizer (FIX-RT-1) no pipeline,
    imediatamente após solvePnP e antes de emitir RTFrameFeatures.
    """

    def __init__(
        self,
        backend: LandmarkBackend,
        config: Optional[RTExtractorConfig] = None,
    ) -> None:
        self.backend = backend
        self.cfg = config or RTExtractorConfig()
        self._frame_idx = 0
        self._prev_ear_avg: float = 0.0
        # ── FIX-RT-1: HeadPoseSanitizer ──────────────────────────
        self._pose_sanitizer = HeadPoseSanitizer()

    def process_frame(self, frame_bgr: np.ndarray) -> RTFrameFeatures:
        h, w = frame_bgr.shape[:2]
        landmarks, has_face = self.backend.process(frame_bgr)

        if landmarks is None or not has_face:
            ear_velocity = (0.0 - self._prev_ear_avg) * self.cfg.fps
            feats = RTFrameFeatures(
                timestamp_ms=self._frame_idx * (1000.0 / self.cfg.fps),
                frame_idx=self._frame_idx,
                ear_l=0.0,
                ear_r=0.0,
                ear_avg=0.0,
                ear_velocity=ear_velocity,
                mar=0.0,
                head_pitch=0.0,
                head_yaw=0.0,
                head_roll=0.0,
                face_detected=False,
            )
            self._prev_ear_avg = 0.0
        else:
            ear_l = compute_ear_from_landmarks(landmarks, LEFT_EYE)
            ear_r = compute_ear_from_landmarks(landmarks, RIGHT_EYE)
            ear_avg = (ear_l + ear_r) / 2.0
            ear_velocity = (ear_avg - self._prev_ear_avg) * self.cfg.fps
            mar = compute_mar_from_landmarks(landmarks)

            # ── FIX-RT-1: solvePnP → HeadPoseSanitizer → features ──
            pitch_raw, yaw_raw, roll_raw = compute_head_pose_from_landmarks(
                landmarks, frame_width=w, frame_height=h
            )
            pitch, yaw, roll = self._pose_sanitizer.sanitize(
                pitch_raw, yaw_raw, roll_raw
            )
            # ─────────────────────────────────────────────────────────

            feats = RTFrameFeatures(
                timestamp_ms=self._frame_idx * (1000.0 / self.cfg.fps),
                frame_idx=self._frame_idx,
                ear_l=ear_l,
                ear_r=ear_r,
                ear_avg=ear_avg,
                ear_velocity=ear_velocity,
                mar=mar,
                head_pitch=pitch,
                head_yaw=yaw,
                head_roll=roll,
                face_detected=True,
            )
            self._prev_ear_avg = ear_avg

        self._frame_idx += 1
        return feats


class DummyBackend:
    """Backend de landmarks de exemplo. Não detecta nada – testes de integração."""

    def process(
        self, frame_bgr: np.ndarray
    ) -> Tuple[Optional[np.ndarray], bool]:
        _ = frame_bgr
        return None, False


# ── ONNXFaceMeshBackend (BlazeFace + FaceMesh ONNX) ──────────────────────────


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


class ONNXFaceMeshBackend:
    """
    Backend de landmarks via BlazeFace + FaceMesh ONNX (sem mediapipe).

    Pipeline:
    1. BlazeFace ONNX detecta face bbox (128x128 input)
    2. Crop + expand face region com 25% margin
    3. FaceMesh ONNX extrai 468 landmarks 3D (256x256 input)
    4. Landmarks convertidos para [0,1] relativos ao frame original

    Interface idêntica ao LandmarkBackend Protocol.
    """

    BLAZEFACE_INPUT_SIZE = 128
    FACEMESH_INPUT_SIZE = 256
    FACEMESH_NUM_LANDMARKS = 468
    CROP_MARGIN = 0.25

    def __init__(
        self,
        detector_path: str = "models/blazeface_detector.onnx",
        mesh_path: str = "models/face_mesh_landmark.onnx",
        *,
        min_face_score: float = 0.5,
    ) -> None:
        if not HAS_ONNX:
            raise RuntimeError(
                "onnxruntime não está instalado. "
                "Instale com `pip install onnxruntime` para ONNXFaceMeshBackend."
            )

        detector_path = str(Path(detector_path).resolve())
        mesh_path = str(Path(mesh_path).resolve())

        self._detector = ort.InferenceSession(detector_path)
        self._mesh = ort.InferenceSession(mesh_path)
        self._detector_input_name = self._detector.get_inputs()[0].name
        self._mesh_input_name = self._mesh.get_inputs()[0].name
        self._min_face_score = min_face_score

    def process(
        self, frame_bgr: np.ndarray
    ) -> Tuple[Optional[np.ndarray], bool]:
        h, w = frame_bgr.shape[:2]
        if h == 0 or w == 0:
            return None, False

        bbox = self._detect_face(frame_bgr)
        if bbox is None:
            return None, False

        x1, y1, x2, y2 = bbox

        # Expand bbox com margin (25%)
        bw = x2 - x1
        bh = y2 - y1
        pad_w = bw * self.CROP_MARGIN
        pad_h = bh * self.CROP_MARGIN
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)

        crop = frame_bgr[int(y1) : int(y2), int(x1) : int(x2)]
        if crop.size == 0:
            return None, False

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_resized = cv2.resize(
            crop_rgb,
            (self.FACEMESH_INPUT_SIZE, self.FACEMESH_INPUT_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )
        mesh_in = (crop_resized.astype(np.float32) / 255.0).reshape(
            1, self.FACEMESH_INPUT_SIZE, self.FACEMESH_INPUT_SIZE, 3
        )

        mesh_out = self._mesh.run(None, {self._mesh_input_name: mesh_in})[0]
        flat = mesh_out[0, 0, 0, : self.FACEMESH_NUM_LANDMARKS * 3]
        lms_crop = flat.reshape(self.FACEMESH_NUM_LANDMARKS, 3)

        x_crop = lms_crop[:, 0] / self.FACEMESH_INPUT_SIZE
        y_crop = lms_crop[:, 1] / self.FACEMESH_INPUT_SIZE

        crop_w_orig = float(x2 - x1)
        crop_h_orig = float(y2 - y1)
        x_frame = (x1 + x_crop * crop_w_orig) / w
        y_frame = (y1 + y_crop * crop_h_orig) / h

        landmarks = np.stack([x_frame, y_frame], axis=1).astype(np.float32)
        return landmarks, True

    def _detect_face(
        self, frame_bgr: np.ndarray
    ) -> Optional[Tuple[float, float, float, float]]:
        """Roda BlazeFace e retorna bbox (x1,y1,x2,y2) em coords de frame, ou None."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inp = cv2.resize(
            rgb,
            (self.BLAZEFACE_INPUT_SIZE, self.BLAZEFACE_INPUT_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )
        inp = (inp.astype(np.float32) / 255.0).reshape(
            1, self.BLAZEFACE_INPUT_SIZE, self.BLAZEFACE_INPUT_SIZE, 3
        )

        reg, cls = self._detector.run(None, {self._detector_input_name: inp})
        scores = _sigmoid(cls[0, :, 0])
        best_idx = int(np.argmax(scores))
        if scores[best_idx] < self._min_face_score:
            return None

        r = reg[0, best_idx, :]
        xc = float(r[0])
        yc = float(r[1])
        bw = float(r[2])
        bh = float(r[3])
        bw = max(bw, 8.0)
        bh = max(bh, 8.0)
        x1_128 = max(0, xc - bw / 2)
        y1_128 = max(0, yc - bh / 2)
        x2_128 = min(self.BLAZEFACE_INPUT_SIZE, xc + bw / 2)
        y2_128 = min(self.BLAZEFACE_INPUT_SIZE, yc + bh / 2)

        scale_x = w / self.BLAZEFACE_INPUT_SIZE
        scale_y = h / self.BLAZEFACE_INPUT_SIZE
        x1 = x1_128 * scale_x
        y1 = y1_128 * scale_y
        x2 = x2_128 * scale_x
        y2 = y2_128 * scale_y

        if x2 - x1 < 10 or y2 - y1 < 10:
            return None

        return (x1, y1, x2, y2)