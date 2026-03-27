"""
Módulo de Reflection (Auto-Correção) para DeteccaoFadiga2.

Implementa o padrão Producer-Critic em três componentes:
  R1. DriftReflector   — detecta drift nas features ao longo do tempo
  R2. PredictionReflector — detecta padrões anômalos nas predições
  R3. AutoRecalibrationManager — fecha o loop de auto-correção

O "Producer" é o pipeline normal de inferência.
O "Critic" é este módulo, que opera a cada N janelas.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── R1: Detector de Drift ─────────────────────────────────────────────────


class DriftStatus(Enum):
    STABLE = "stable"
    WARNING = "warning"      # drift leve — logar
    DRIFTED = "drifted"      # drift severo — sugerir recalibração
    CRITICAL = "critical"    # drift extremo — suspender inferência


@dataclass
class DriftReport:
    """Relatório de drift para uma feature ou conjunto de features."""
    status: DriftStatus
    drifted_features: List[str]
    details: Dict[str, float]    # feature_name -> zscore_from_baseline
    recommendation: str          # "continue" | "recalibrate" | "suspend"
    windows_analyzed: int


@dataclass
class DriftReflectorConfig:
    """Configuração do detector de drift."""
    analysis_interval: int = 10     # Analisar a cada 10 janelas (~2.5 min)
    window_buffer_size: int = 20    # Manter últimas 20 janelas para análise
    warning_threshold: float = 1.5  # |z-score da média| > 1.5 -> warning
    drift_threshold: float = 2.5    # |z-score da média| > 2.5 -> drifted
    critical_threshold: float = 4.0 # |z-score da média| > 4.0 -> critical
    # Features monitoradas para drift (as mais sensíveis a mudanças externas)
    monitored_features: Tuple[str, ...] = (
        "ear_mean", "ear_std", "mar_mean",
        "pitch_mean", "pitch_std", "yaw_std",
    )


class DriftReflector:
    """
    R1: Critic que monitora drift entre as features recentes e o baseline.

    Lógica:
    - Mantém buffer circular das últimas N janelas de features
    - A cada analysis_interval janelas, compara a média recente
      contra os training_stats do inference_config.json
    - Se a média recente se desviou além do threshold, reporta drift

    O pipeline (Producer) é responsável por agir sobre o DriftReport.
    """

    def __init__(
        self,
        training_stats: Dict[str, Dict[str, float]],
        config: Optional[DriftReflectorConfig] = None,
    ) -> None:
        self.cfg = config or DriftReflectorConfig()
        self._training_stats = training_stats
        self._window_buffer: deque = deque(
            maxlen=self.cfg.window_buffer_size
        )
        self._windows_since_last_analysis: int = 0
        self._total_analyses: int = 0
        self._last_report: Optional[DriftReport] = None

    def push_window(
        self, window_feats: Dict[str, float]
    ) -> Optional[DriftReport]:
        """
        Alimenta uma janela de features. Retorna DriftReport quando é
        hora de analisar, ou None caso contrário.
        """
        self._window_buffer.append(window_feats)
        self._windows_since_last_analysis += 1

        if self._windows_since_last_analysis < self.cfg.analysis_interval:
            return None

        if len(self._window_buffer) < 5:
            return None  # dados insuficientes

        self._windows_since_last_analysis = 0
        self._total_analyses += 1

        report = self._analyze()
        self._last_report = report
        return report

    def _analyze(self) -> DriftReport:
        """
        Compara médias recentes vs training_stats usando z-score.

        Para cada feature monitorada:
          z = (mean_recente - mean_treino) / std_treino

        Se |z| > threshold para qualquer feature -> drift.
        """
        drifted = []
        details = {}
        worst_status = DriftStatus.STABLE

        for feat_name in self.cfg.monitored_features:
            stats = self._training_stats.get(feat_name)
            if stats is None:
                continue

            train_mean = stats["mean"]
            train_std = stats["std"]
            if train_std < 1e-8:
                continue

            # Média recente desta feature
            recent_vals = [
                w[feat_name] for w in self._window_buffer
                if feat_name in w
            ]
            if not recent_vals:
                continue

            recent_mean = float(np.mean(recent_vals))
            z = (recent_mean - train_mean) / train_std
            details[feat_name] = round(z, 3)

            abs_z = abs(z)
            if abs_z >= self.cfg.critical_threshold:
                drifted.append(feat_name)
                if worst_status != DriftStatus.CRITICAL:
                    worst_status = DriftStatus.CRITICAL
            elif abs_z >= self.cfg.drift_threshold:
                drifted.append(feat_name)
                if worst_status not in (
                    DriftStatus.CRITICAL,
                ):
                    worst_status = DriftStatus.DRIFTED
            elif abs_z >= self.cfg.warning_threshold:
                drifted.append(feat_name)
                if worst_status in (
                    DriftStatus.STABLE,
                ):
                    worst_status = DriftStatus.WARNING

        # Definir recomendação
        if worst_status == DriftStatus.CRITICAL:
            rec = "suspend"
        elif worst_status == DriftStatus.DRIFTED:
            rec = "recalibrate"
        else:
            rec = "continue"

        return DriftReport(
            status=worst_status,
            drifted_features=drifted,
            details=details,
            recommendation=rec,
            windows_analyzed=len(self._window_buffer),
        )

    @property
    def last_report(self) -> Optional[DriftReport]:
        return self._last_report


# ── R2: Crítico de Predição ───────────────────────────────────────────────


@dataclass
class PredictionReflection:
    """Resultado da reflexão sobre padrões de predição."""
    pattern: str           # "stable" | "oscillating" | "stuck_danger" | "sudden_transition"
    confidence_modifier: float  # Multiplicador de confiança: 1.0 = sem ajuste
    suggestion: str        # Texto explicando o padrão
    consecutive_danger: int
    consecutive_safe: int
    recent_prob_mean: float
    recent_prob_std: float


class PredictionReflector:
    """
    R2: Critic que analisa padrões temporais nas sequências de predições.

    Detecta:
    - Oscilação rápida Safe/Danger (indica threshold na zona de fronteira)
    - Stuck em Danger por muito tempo (possível falso positivo crônico)
    - Transição abrupta Safe->Danger sem features intermediárias
    """

    def __init__(self, buffer_size: int = 30) -> None:
        self._prob_buffer: deque = deque(maxlen=buffer_size)
        self._label_buffer: deque = deque(maxlen=buffer_size)
        self._consecutive_danger: int = 0
        self._consecutive_safe: int = 0
        self._c29_active: bool = False

    def push(
        self, prob_danger: float, label: str,
        *, c29_active: bool = False,
    ) -> PredictionReflection:
        """Alimenta uma predição e retorna reflexão."""
        self._c29_active = c29_active
        self._prob_buffer.append(prob_danger)
        self._label_buffer.append(label)

        if label == "Danger":
            self._consecutive_danger += 1
            self._consecutive_safe = 0
        else:
            self._consecutive_safe += 1
            self._consecutive_danger = 0

        return self._reflect()

    def _reflect(self) -> PredictionReflection:
        if len(self._prob_buffer) < 5:
            return PredictionReflection(
                pattern="stable",
                confidence_modifier=1.0,
                suggestion="Dados insuficientes para reflexão",
                consecutive_danger=self._consecutive_danger,
                consecutive_safe=self._consecutive_safe,
                recent_prob_mean=0.0,
                recent_prob_std=0.0,
            )

        probs = np.array(self._prob_buffer)
        labels = list(self._label_buffer)
        prob_mean = float(probs.mean())
        prob_std = float(probs.std())

        # Detectar oscilação: muitas transições Safe<->Danger
        transitions = sum(
            1 for i in range(1, len(labels))
            if labels[i] != labels[i - 1]
        )
        oscillation_rate = transitions / max(len(labels) - 1, 1)

        # Detectar stuck em Danger (suprimir se C29 esta causando os Dangers)
        if self._consecutive_danger >= 15 and not self._c29_active:
            return PredictionReflection(
                pattern="stuck_danger",
                confidence_modifier=0.7,
                suggestion=(
                    f"Danger contínuo por {self._consecutive_danger} janelas. "
                    f"Verificar se é fadiga real ou drift de calibração. "
                    f"Considerar recalibração."
                ),
                consecutive_danger=self._consecutive_danger,
                consecutive_safe=self._consecutive_safe,
                recent_prob_mean=prob_mean,
                recent_prob_std=prob_std,
            )

        # Detectar oscilação
        if oscillation_rate > 0.4 and len(labels) >= 10:
            return PredictionReflection(
                pattern="oscillating",
                confidence_modifier=0.5,
                suggestion=(
                    f"Oscilação Safe/Danger detectada "
                    f"(taxa={oscillation_rate:.2f}). "
                    f"Operador pode estar no limiar de fadiga. "
                    f"prob_mean={prob_mean:.3f} — próximo do threshold."
                ),
                consecutive_danger=self._consecutive_danger,
                consecutive_safe=self._consecutive_safe,
                recent_prob_mean=prob_mean,
                recent_prob_std=prob_std,
            )

        # Detectar transição abrupta
        if len(probs) >= 3:
            last_3 = probs[-3:]
            delta = float(last_3[-1] - last_3[0])
            if abs(delta) > 0.4:
                return PredictionReflection(
                    pattern="sudden_transition",
                    confidence_modifier=0.8,
                    suggestion=(
                        f"Transição abrupta de probabilidade: "
                        f"delta={delta:+.3f} em 3 janelas. "
                        f"Verificar mudança brusca de condições."
                    ),
                    consecutive_danger=self._consecutive_danger,
                    consecutive_safe=self._consecutive_safe,
                    recent_prob_mean=prob_mean,
                    recent_prob_std=prob_std,
                )

        return PredictionReflection(
            pattern="stable",
            confidence_modifier=1.0,
            suggestion="Padrão de predição estável",
            consecutive_danger=self._consecutive_danger,
            consecutive_safe=self._consecutive_safe,
            recent_prob_mean=prob_mean,
            recent_prob_std=prob_std,
        )


# ── R3: Auto-Recalibração Reflexiva ──────────────────────────────────────


@dataclass
class RecalibrationDecision:
    should_recalibrate: bool
    reason: str
    urgency: str  # "immediate" | "scheduled" | "none"


class AutoRecalibrationManager:
    """
    R3: Fecha o loop Reflection — decide quando auto-recalibrar.

    Combina sinais do DriftReflector e PredictionReflector para decidir
    se a recalibração é necessária.

    Regras:
    1. Drift CRITICAL -> recalibração imediata
    2. Drift DRIFTED + stuck_danger -> recalibração imediata
    3. Drift DRIFTED sozinho -> recalibração agendada (próximo checkpoint)
    4. stuck_danger sem drift -> sugerir (pode ser fadiga real)
    5. Cooldown: no mínimo 5 min entre recalibrações
    """

    def __init__(
        self,
        min_recal_interval_sec: float = 300.0,  # 5 min entre recalibrações
        max_recalibrations_per_hour: int = 4,
    ) -> None:
        self._min_interval = min_recal_interval_sec
        self._max_per_hour = max_recalibrations_per_hour
        self._recal_timestamps: List[float] = []
        self._last_recal_time: float = -(min_recal_interval_sec + 1.0)

    def evaluate(
        self,
        drift_report: Optional[DriftReport],
        pred_reflection: Optional[PredictionReflection],
    ) -> RecalibrationDecision:
        """Decide se deve recalibrar com base nos sinais combinados."""
        import time as _time
        now = _time.monotonic()

        # Cooldown check
        if now - self._last_recal_time < self._min_interval:
            return RecalibrationDecision(
                should_recalibrate=False,
                reason="Dentro do cooldown mínimo entre recalibrações",
                urgency="none",
            )

        # Rate limit check
        recent = [
            t for t in self._recal_timestamps
            if now - t < 3600
        ]
        if len(recent) >= self._max_per_hour:
            return RecalibrationDecision(
                should_recalibrate=False,
                reason=(
                    f"Limite de {self._max_per_hour} "
                    f"recalibrações/hora atingido"
                ),
                urgency="none",
            )

        # Avaliar sinais combinados
        drift_status = (
            drift_report.status if drift_report else DriftStatus.STABLE
        )
        pred_pattern = (
            pred_reflection.pattern if pred_reflection else "stable"
        )

        # Regra 1: drift critical -> imediato
        if drift_status == DriftStatus.CRITICAL:
            return self._approve_recal(
                now,
                "Drift CRITICAL detectado — features completamente "
                "fora da distribuição de treino",
                "immediate",
            )

        # Regra 2: drift + stuck -> imediato
        if (
            drift_status == DriftStatus.DRIFTED
            and pred_pattern == "stuck_danger"
        ):
            return self._approve_recal(
                now,
                "Drift DRIFTED + Danger contínuo — provável mudança "
                "de condições (não fadiga real)",
                "immediate",
            )

        # Regra 3: drift sozinho -> agendado
        if drift_status == DriftStatus.DRIFTED:
            return RecalibrationDecision(
                should_recalibrate=True,
                reason="Drift DRIFTED detectado — agendar recalibração",
                urgency="scheduled",
            )

        # Regra 4: stuck_danger sem drift
        if pred_pattern == "stuck_danger":
            # stuck_danger severo (30+ janelas) → forçar recalibração
            if (pred_reflection is not None
                    and pred_reflection.consecutive_danger >= 30):
                return self._approve_recal(
                    now,
                    f"Stuck Danger severo ({pred_reflection.consecutive_danger} "
                    f"janelas) sem drift de features. Forçando recalibração "
                    f"para limpar estado.",
                    "scheduled",
                )
            # stuck_danger curto → manter como sugestão (pode ser fadiga real)
            return RecalibrationDecision(
                should_recalibrate=False,
                reason=(
                    "Danger contínuo SEM drift — pode ser fadiga real. "
                    "Não recalibrar automaticamente."
                ),
                urgency="none",
            )

        return RecalibrationDecision(
            should_recalibrate=False,
            reason="Nenhum sinal de necessidade de recalibração",
            urgency="none",
        )

    def _approve_recal(
        self, now: float, reason: str, urgency: str
    ) -> RecalibrationDecision:
        self._last_recal_time = now
        self._recal_timestamps.append(now)
        return RecalibrationDecision(
            should_recalibrate=True,
            reason=reason,
            urgency=urgency,
        )
