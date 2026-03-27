"""
Guardrails para o pipeline de inferência DeteccaoFadiga2.

Três camadas de proteção:
  G1. Validação estruturada de saída (Pydantic-like, sem dependência)
  G2. Validação de entrada (features dentro de ranges fisiológicos)
  G3. Restrições comportamentais (rate limit de alertas, watchdog)
  G4. Validação de calibração (critérios fisiológicos e estatísticos)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import time
import numpy as np


# ── G1: Validação Estruturada de Saída ────────────────────────────────────


class AlertLevel(Enum):
    """Níveis de alerta graduados (ao invés de binário Safe/Danger)."""
    SAFE = 0
    WATCH = 1       # prob entre low_thresh e threshold
    DANGER = 2      # prob >= threshold
    CRITICAL = 3    # Danger consecutivo ou microsleep


@dataclass
class FatigueOutput:
    """Saída validada de uma inferência. Substitui a tupla (float, str)."""
    label: str                          # "Safe" | "Danger"
    prob_danger: float                  # [0.0, 1.0]
    alert_level: AlertLevel
    features_valid: bool                # todas 19 features dentro do range?
    confidence: str                     # "high" | "medium" | "low"
    window_quality: float               # ratio de frames válidos na janela
    timestamp_ms: float

    # Métricas-chave para overlay e logging
    perclos: float = 0.0
    blink_count: float = 0.0
    microsleep_count: float = 0.0

    def __post_init__(self):
        """Validação automática no momento da criação."""
        errors = []
        if not (0.0 <= self.prob_danger <= 1.0):
            errors.append(
                f"prob_danger={self.prob_danger} fora de [0,1]"
            )
        if self.label not in ("Safe", "Danger"):
            errors.append(f"label='{self.label}' inválido")
        if not (0.0 <= self.window_quality <= 1.0):
            errors.append(
                f"window_quality={self.window_quality} fora de [0,1]"
            )
        if errors:
            raise ValueError(
                f"FatigueOutput inválido: {'; '.join(errors)}"
            )


# ── G2: Validação de Entrada (Features) ──────────────────────────────────

# Ranges baseados em training_stats do inference_config.json
# Usa min/max do treino com margem de 20%
FEATURE_RANGES: Dict[str, Tuple[float, float]] = {
    "ear_mean":      (-18.0, 6.0),      # treino: [-14.9, 4.4]
    "ear_std":       (-2.5, 16.0),       # treino: [-1.7, 13.0]
    "ear_min":       (-10.0, 3.0),       # treino: [-8.0, 1.9]
    "ear_vel_mean":  (-33.0, 30.0),      # treino: [-27.5, 24.2]
    "ear_vel_std":   (-2.0, 16.0),       # treino: [-1.1, 13.2]
    "mar_mean":      (-2.0, 28.0),       # treino: [-1.3, 23.5]
    "pitch_mean":    (-11.0, 10.0),      # treino: [-8.6, 8.2]
    "pitch_std":     (-1.0, 31.0),       # treino: [-0.7, 25.3]
    "yaw_std":       (-1.0, 20.0),       # treino: [-0.6, 16.3]
    "roll_std":      (-1.0, 27.0),       # treino: [-0.6, 22.3]
    "blink_count":   (0.0, 35.0),        # treino: [0, 26]
    "blink_rate_per_min": (0.0, 130.0),  # treino: [0, 104]
    "blink_mean_dur_ms":  (0.0, 24000.0),# treino: [0, 19300]
    "perclos_p80_mean":   (0.0, 1.0),
    "perclos_p80_max":    (0.0, 1.0),
    "blink_closing_vel_mean": (-2.0, 7.0),  # treino: [-1.3, 5.0]
    "blink_opening_vel_mean": (-2.0, 11.0), # treino: [-1.4, 8.5]
    "long_blink_pct":     (0.0, 1.0),
    "blink_regularity":   (0.0, 3.0),       # treino: [0, 2.2]
}


def _check_feature_ranges(
    window_feats: Dict[str, float],
    feature_names: List[str],
) -> bool:
    """
    Retorna True se TODAS as features estão dentro dos ranges esperados.
    Loga warnings para features fora do range.
    """
    all_valid = True
    for name in feature_names:
        val = window_feats.get(name)
        if val is None:
            all_valid = False
            continue
        lo, hi = FEATURE_RANGES.get(name, (-np.inf, np.inf))
        if not (lo <= val <= hi):
            print(
                f"[guardrail] WARN: {name}={val:.4f} "
                f"fora do range [{lo}, {hi}]"
            )
            all_valid = False
    return all_valid


def _compute_confidence(
    window_feats: Dict[str, float],
    features_valid: bool,
) -> str:
    """
    Determina confiança da predição baseado na qualidade dos dados.

    high:   features válidas + blink_count > 0 (houve atividade ocular)
    medium: features válidas mas blink_count == 0 (pode ser estático)
    low:    alguma feature fora do range esperado
    """
    if not features_valid:
        return "low"
    blinks = window_feats.get("blink_count", 0.0)
    if blinks > 0:
        return "high"
    return "medium"


def validate_and_wrap(
    prob_danger: float,
    label: str,
    window_feats: Dict[str, float],
    feature_names: List[str],
    config,  # InferenceConfig
    timestamp_ms: float,
    threshold: float = 0.41,
    watch_threshold: float = 0.30,
) -> FatigueOutput:
    """Valida a predição e retorna FatigueOutput estruturado."""

    # Clamp de segurança (nunca deveria ser necessário, mas protege)
    prob_clamped = float(np.clip(prob_danger, 0.0, 1.0))

    # Determinar alert_level graduado
    if prob_clamped >= threshold:
        alert_level = AlertLevel.DANGER
    elif prob_clamped >= watch_threshold:
        alert_level = AlertLevel.WATCH
    else:
        alert_level = AlertLevel.SAFE

    # Checar se microsleep eleva para CRITICAL
    micros = window_feats.get("microsleep_count", 0.0)
    if micros > 0 and alert_level == AlertLevel.DANGER:
        alert_level = AlertLevel.CRITICAL

    # Validar ranges das features
    features_valid = _check_feature_ranges(window_feats, feature_names)

    # Determinar confiança
    confidence = _compute_confidence(window_feats, features_valid)

    return FatigueOutput(
        label=label,
        prob_danger=prob_clamped,
        alert_level=alert_level,
        features_valid=features_valid,
        confidence=confidence,
        window_quality=1.0,  # será calculado pelo window_factory
        timestamp_ms=timestamp_ms,
        perclos=window_feats.get("perclos_p80_mean", 0.0),
        blink_count=window_feats.get("blink_count", 0.0),
        microsleep_count=micros,
    )


# ── G3: Restrições Comportamentais ───────────────────────────────────────


@dataclass
class BehaviorGuardrailConfig:
    """Configuração das restrições comportamentais."""
    alert_cooldown_sec: float = 60.0     # Min 60s entre alertas sonoros
    post_calibration_grace_windows: int = 2  # Ignorar 2 janelas pós-calibração
    watchdog_timeout_sec: float = 30.0   # Alerta se sem inferência por 30s
    max_consecutive_danger: int = 20     # Safety cap — forçar ação após 20 Dangers
    min_face_ratio_for_inference: float = 0.50  # Abaixo disso, suspender


class BehaviorGuardRails:
    """
    Guardrails comportamentais — opera sobre o fluxo de saídas ao longo
    do tempo, não sobre uma única predição.
    """

    def __init__(
        self, config: Optional[BehaviorGuardrailConfig] = None
    ) -> None:
        self.cfg = config or BehaviorGuardrailConfig()
        self._last_alert_time: float = -(self.cfg.alert_cooldown_sec + 1.0)
        self._last_inference_time: float = time.monotonic()
        self._windows_since_calibration: int = 0
        self._consecutive_danger: int = 0
        self._calibration_just_completed: bool = False
        self._total_outputs: int = 0
        self._total_suppressed: int = 0

    def on_calibration_complete(self) -> None:
        """Chamado quando calibração termina. Inicia grace period."""
        self._calibration_just_completed = True
        self._windows_since_calibration = 0

    def process(self, output: FatigueOutput) -> FatigueOutput:
        """
        Aplica guardrails comportamentais sobre um FatigueOutput.

        Pode:
        - Rebaixar alert_level durante grace period pós-calibração
        - Elevar para CRITICAL se consecutive_danger > max
        - Suprimir flag de alerta sonoro se dentro do cooldown

        Retorna FatigueOutput (possivelmente modificado).
        """
        now = time.monotonic()
        self._last_inference_time = now
        self._total_outputs += 1

        # (1) Grace period pós-calibração
        if self._calibration_just_completed:
            self._windows_since_calibration += 1
            if self._windows_since_calibration <= self.cfg.post_calibration_grace_windows:
                # Rebaixar qualquer Danger para WATCH durante grace
                if output.alert_level in (
                    AlertLevel.DANGER, AlertLevel.CRITICAL
                ):
                    output.alert_level = AlertLevel.WATCH
                    output.confidence = "low"
                    self._total_suppressed += 1
                return output
            else:
                self._calibration_just_completed = False

        # (2) Tracking de consecutive danger
        if output.label == "Danger":
            self._consecutive_danger += 1
        else:
            self._consecutive_danger = 0

        # (3) Elevar para CRITICAL se muitos Dangers consecutivos
        if self._consecutive_danger >= self.cfg.max_consecutive_danger:
            output.alert_level = AlertLevel.CRITICAL

        return output

    def should_sound_alert(self, output: FatigueOutput) -> bool:
        """
        Rate limiter: retorna True somente se o alerta sonoro
        é permitido (respeitando cooldown).
        """
        if output.alert_level.value < AlertLevel.DANGER.value:
            return False

        now = time.monotonic()
        elapsed = now - self._last_alert_time
        if elapsed < self.cfg.alert_cooldown_sec:
            return False

        self._last_alert_time = now
        return True

    def check_watchdog(self) -> bool:
        """
        Retorna True se o pipeline está saudável.
        Retorna False se sem inferência por mais de watchdog_timeout_sec.
        Deve ser chamado periodicamente (ex: a cada 5s de um timer).
        """
        elapsed = time.monotonic() - self._last_inference_time
        if elapsed > self.cfg.watchdog_timeout_sec:
            print(
                f"[guardrail] WATCHDOG: sem inferência há "
                f"{elapsed:.1f}s (timeout={self.cfg.watchdog_timeout_sec}s)"
            )
            return False
        return True

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "total_outputs": self._total_outputs,
            "total_suppressed": self._total_suppressed,
            "consecutive_danger": self._consecutive_danger,
        }


# ── G4: Guardrail de Calibração ──────────────────────────────────────────


@dataclass
class CalibrationVerdict:
    """Resultado da validação da calibração."""
    is_acceptable: bool
    issues: List[str]
    recommendation: str  # "accept" | "retry" | "use_with_caution"


def validate_calibration(baseline) -> CalibrationVerdict:
    """
    Valida um SubjectBaseline contra critérios fisiológicos e estatísticos.

    Critérios:
    1. EAR mean entre [0.15, 0.40] — olhos humanos normais
    2. EAR std entre [0.015, 0.10] — se muito baixo, não houve piscadas
    3. Pitch std < 30° — se maior, dados corrompidos
    4. baseline.is_valid == True
    5. Yaw mean < 30° — operador frontal à câmera
    """
    issues = []

    if not baseline.is_valid:
        issues.append("Baseline marcado como inválido pelo calibrador")

    # EAR mean fisiológico
    if baseline.ear_mean < 0.15:
        issues.append(
            f"EAR mean={baseline.ear_mean:.4f} muito baixo (<0.15). "
            f"Possível: olhos parcialmente fechados durante warm-up ou "
            f"má detecção de landmarks"
        )
    elif baseline.ear_mean > 0.40:
        issues.append(
            f"EAR mean={baseline.ear_mean:.4f} muito alto (>0.40). "
            f"Possível: artefato de landmark (rosto parcial)"
        )

    # EAR std — deve ter variação (blinks)
    if baseline.ear_std < 0.015:
        issues.append(
            f"EAR std={baseline.ear_std:.4f} muito baixo. "
            f"Operador pode não ter piscado durante warm-up"
        )
    elif baseline.ear_std > 0.10:
        issues.append(
            f"EAR std={baseline.ear_std:.4f} muito alto. "
            f"Instabilidade nos landmarks ou iluminação variável"
        )

    # Pitch std — sanity check pós-sanitizer
    if baseline.pitch_std > 25.0:
        issues.append(
            f"Pitch std={baseline.pitch_std:.2f} excessivo. "
            f"HeadPoseSanitizer pode não estar corrigindo flip PnP"
        )

    # Yaw/Roll — operador deve estar relativamente frontal
    if abs(baseline.yaw_mean) > 30.0:
        issues.append(
            f"Yaw mean={baseline.yaw_mean:.2f} — operador não "
            f"frontal à câmera durante warm-up"
        )

    # Determinar recomendação
    if not issues:
        return CalibrationVerdict(
            is_acceptable=True,
            issues=[],
            recommendation="accept",
        )

    critical = any(
        "muito baixo (<0.15)" in i
        or "inválido" in i
        or "muito alto" in i
        or "excessivo" in i
        for i in issues
    )
    if critical:
        return CalibrationVerdict(
            is_acceptable=False,
            issues=issues,
            recommendation="retry",
        )

    return CalibrationVerdict(
        is_acceptable=True,
        issues=issues,
        recommendation="use_with_caution",
    )
