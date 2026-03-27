"""
Multi-Agent Collaboration para DeteccaoFadiga.

Arquitetura Supervisor com 3 agentes especialistas:
  A1. AgentOpinion / AgentProtocol  — protocolo de comunicação
  A2. OcularAgent     — EAR + PERCLOS (microsleep iminente)
  A3. BlinkAgent      — padrões de piscada (degradação progressiva)
  A4. PosturalAgent   — head pose + MAR (fadiga postural)
  A5. SupervisorAgent — agrega opiniões, decisão final

Ref: Agentic Design Patterns Ch.7 — Multi-Agent Collaboration, Supervisor model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Protocol, Tuple

import numpy as np


# ── A1: Protocolo de Comunicação ─────────────────────────────────────────


class FatigueSignal(Enum):
    """Nível de fadiga reportado por um agente especialista."""
    CLEAR = 0       # Sem sinal de fadiga neste domínio
    MILD = 1        # Sinal leve — monitorar
    MODERATE = 2    # Sinal moderado — atenção
    SEVERE = 3      # Sinal forte — intervenção recomendada
    CRITICAL = 4    # Sinal extremo — ação imediata


@dataclass
class AgentOpinion:
    """
    Opinião estruturada emitida por um agente especialista.

    É o contrato de comunicação entre agentes e Supervisor.
    Todo agente DEVE preencher todos os campos.
    """
    agent_name: str              # "OcularAgent" | "BlinkAgent" | "PosturalAgent"
    signal: FatigueSignal        # Nível de fadiga detectado
    confidence: float            # [0.0, 1.0] — quão confiante o agente está
    score: float                 # [0.0, 1.0] — score contínuo de fadiga
    reasoning: str               # Explicação textual curta do diagnóstico
    key_indicators: Dict[str, float]  # Features mais relevantes para a decisão

    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence={self.confidence} fora de [0,1]")
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"score={self.score} fora de [0,1]")


class SpecialistAgent(Protocol):
    """Interface que todo agente especialista deve implementar."""

    @property
    def name(self) -> str: ...

    def analyze(self, window_feats: Dict[str, float]) -> AgentOpinion: ...


# ── A2: OcularAgent ─────────────────────────────────────────────────────


class OcularAgent:
    """
    A2: Especialista em sinais oculares (EAR + PERCLOS).

    Detecta:
    - Olhos progressivamente fechando (ear_mean baixo)
    - Variabilidade reduzida (ear_std baixo — olhar fixo/vidroso)
    - PERCLOS alto (proporção de tempo com olhos fechados)
    - Microsleep patterns (ear_min extremo + PERCLOS spike)

    Domínio: indicador mais IMEDIATO de fadiga — microsleep iminente.
    Tempo de reação: segundos.
    """

    def __init__(self) -> None:
        self._name = "OcularAgent"

    @property
    def name(self) -> str:
        return self._name

    def analyze(self, window_feats: Dict[str, float]) -> AgentOpinion:
        ear_mean = window_feats.get("ear_mean", 0.0)
        ear_std = window_feats.get("ear_std", 0.0)
        ear_min = window_feats.get("ear_min", 0.0)
        ear_vel_mean = window_feats.get("ear_vel_mean", 0.0)
        ear_vel_std = window_feats.get("ear_vel_std", 0.0)
        perclos_mean = window_feats.get("perclos_p80_mean", 0.0)
        perclos_max = window_feats.get("perclos_p80_max", 0.0)

        # Score composto: combinação ponderada dos indicadores
        # ear_mean negativo = olhos mais fechados que baseline
        ear_score = np.clip(-ear_mean / 3.0, 0.0, 1.0)  # -3→1.0, 0→0.0
        perclos_score = np.clip(perclos_mean / 0.4, 0.0, 1.0)  # 0.4→1.0
        perclos_peak = np.clip(perclos_max / 0.5, 0.0, 1.0)
        ear_min_score = np.clip(-ear_min / 4.0, 0.0, 1.0)  # -4→1.0

        # Peso: PERCLOS e ear_min são os mais diagnósticos
        score = float(
            0.25 * ear_score
            + 0.30 * perclos_score
            + 0.25 * perclos_peak
            + 0.20 * ear_min_score
        )
        score = float(np.clip(score, 0.0, 1.0))

        # Determinar sinal
        if score >= 0.75:
            signal = FatigueSignal.CRITICAL
        elif score >= 0.55:
            signal = FatigueSignal.SEVERE
        elif score >= 0.35:
            signal = FatigueSignal.MODERATE
        elif score >= 0.15:
            signal = FatigueSignal.MILD
        else:
            signal = FatigueSignal.CLEAR

        # Confiança: alta se EAR tem variação normal (dados bons)
        # Baixa se ear_std é muito baixo (pode ser artefato)
        confidence = 0.9
        if ear_std < 0.1:
            confidence = 0.6  # pouca variação — dados questionáveis
        if perclos_mean > 0.0 and ear_mean < -1.0:
            confidence = 0.95  # sinais convergentes — alta confiança

        # Reasoning
        reasons = []
        if ear_mean < -1.0:
            reasons.append(f"EAR abaixo do baseline (z={ear_mean:.2f})")
        if perclos_mean > 0.15:
            reasons.append(f"PERCLOS elevado ({perclos_mean:.2f})")
        if perclos_max > 0.3:
            reasons.append(f"Pico PERCLOS {perclos_max:.2f}")
        if ear_min < -3.0:
            reasons.append(f"EAR mínimo extremo (z={ear_min:.2f})")
        reasoning = "; ".join(reasons) if reasons else "Sinais oculares normais"

        return AgentOpinion(
            agent_name=self._name,
            signal=signal,
            confidence=confidence,
            score=score,
            reasoning=reasoning,
            key_indicators={
                "ear_mean": ear_mean,
                "perclos_p80_mean": perclos_mean,
                "perclos_p80_max": perclos_max,
                "ear_min": ear_min,
            },
        )


# ── A3: BlinkAgent ──────────────────────────────────────────────────────


class BlinkAgent:
    """
    A3: Especialista em padrões de piscada.

    Detecta:
    - Blinks longos (>300ms) indicando fadiga progressiva
    - Velocidade de fechamento reduzida (músculos lentos)
    - Taxa de blinks anormal (muito alta ou muito baixa)
    - Irregularidade de piscadas (perda de ritmo)

    Domínio: indicador de fadiga PROGRESSIVA — degradação ao longo
    de minutos/horas. Complementa o OcularAgent que é imediato.
    Tempo de reação: minutos.
    """

    def __init__(self) -> None:
        self._name = "BlinkAgent"

    @property
    def name(self) -> str:
        return self._name

    def analyze(self, window_feats: Dict[str, float]) -> AgentOpinion:
        blink_count = window_feats.get("blink_count", 0.0)
        blink_rate = window_feats.get("blink_rate_per_min", 0.0)
        blink_dur_ms = window_feats.get("blink_mean_dur_ms", 0.0)
        closing_vel = window_feats.get("blink_closing_vel_mean", 0.0)
        opening_vel = window_feats.get("blink_opening_vel_mean", 0.0)
        long_blink_pct = window_feats.get("long_blink_pct", 0.0)
        blink_reg = window_feats.get("blink_regularity", 0.0)

        # Caso especial: sem piscadas detectadas
        if blink_count == 0:
            return AgentOpinion(
                agent_name=self._name,
                signal=FatigueSignal.MILD,
                confidence=0.4,
                score=0.2,
                reasoning="Nenhuma piscada detectada — dados insuficientes ou olhar fixo",
                key_indicators={"blink_count": 0.0},
            )

        # Sub-scores individuais
        # Long blink percentage: > 40% é sinal forte
        long_blink_score = np.clip(long_blink_pct / 0.6, 0.0, 1.0)

        # Blink duration: > 500ms é slow blink
        dur_score = np.clip((blink_dur_ms - 200) / 800, 0.0, 1.0)

        # Closing velocity (Z-normed): negativo = mais lento que treino
        closing_score = np.clip(-closing_vel / 2.0, 0.0, 1.0)

        # Blink rate: muito alta (> 25/min = compensação) ou muito baixa (< 8/min = supressão)
        if blink_rate > 25:
            rate_score = np.clip((blink_rate - 25) / 30, 0.0, 0.6)
        elif blink_rate < 8:
            rate_score = np.clip((8 - blink_rate) / 8, 0.0, 0.6)
        else:
            rate_score = 0.0

        # Irregularidade: > 0.8 = ritmo desorganizado
        irreg_score = np.clip(blink_reg / 1.5, 0.0, 1.0)

        score = float(
            0.30 * long_blink_score
            + 0.25 * dur_score
            + 0.20 * closing_score
            + 0.10 * rate_score
            + 0.15 * irreg_score
        )
        score = float(np.clip(score, 0.0, 1.0))

        # Sinal
        if score >= 0.70:
            signal = FatigueSignal.SEVERE
        elif score >= 0.45:
            signal = FatigueSignal.MODERATE
        elif score >= 0.20:
            signal = FatigueSignal.MILD
        else:
            signal = FatigueSignal.CLEAR

        # Confiança: proporcional ao número de blinks (mais dados = mais confiável)
        confidence = float(np.clip(blink_count / 8.0, 0.3, 0.95))

        # Reasoning
        reasons = []
        if long_blink_pct > 0.3:
            reasons.append(f"Long blinks {long_blink_pct:.0%}")
        if blink_dur_ms > 400:
            reasons.append(f"Blinks lentos ({blink_dur_ms:.0f}ms)")
        if closing_vel < -0.5:
            reasons.append(f"Fechamento lento (z={closing_vel:.2f})")
        if blink_reg > 0.8:
            reasons.append(f"Ritmo irregular ({blink_reg:.2f})")
        reasoning = "; ".join(reasons) if reasons else "Padrão de piscadas normal"

        return AgentOpinion(
            agent_name=self._name,
            signal=signal,
            confidence=confidence,
            score=score,
            reasoning=reasoning,
            key_indicators={
                "long_blink_pct": long_blink_pct,
                "blink_mean_dur_ms": blink_dur_ms,
                "blink_closing_vel_mean": closing_vel,
                "blink_regularity": blink_reg,
            },
        )


# ── A4: PosturalAgent ───────────────────────────────────────────────────


class PosturalAgent:
    """
    A4: Especialista em sinais posturais (head pose + MAR).

    Detecta:
    - Head-nod (pitch negativo = cabeça caindo para frente)
    - Variabilidade postural alta (pitch_std, yaw_std = instabilidade)
    - Bocejo (mar_mean elevado)
    - Desvio lateral (roll_std alto)

    Domínio: indicador de fadiga POSTURAL — complementa sinais oculares.
    Nota: em produção, HeadPoseNeutralizer (C33) zera pitch/yaw/roll.
    Este agente é mais relevante em modo lab (--no-neutralize-pose).
    Em modo produção, baseia-se primariamente no MAR (bocejo).
    Tempo de reação: minutos.
    """

    def __init__(self, pose_neutralized: bool = True) -> None:
        self._name = "PosturalAgent"
        self._pose_neutralized = pose_neutralized

    @property
    def name(self) -> str:
        return self._name

    def analyze(self, window_feats: Dict[str, float]) -> AgentOpinion:
        pitch_mean = window_feats.get("pitch_mean", 0.0)
        pitch_std = window_feats.get("pitch_std", 0.0)
        yaw_std = window_feats.get("yaw_std", 0.0)
        roll_std = window_feats.get("roll_std", 0.0)
        mar_mean = window_feats.get("mar_mean", 0.0)

        # Se pose está neutralizada, os valores de pitch/yaw/roll são 0
        # após o scaler. O agente se baseia apenas no MAR.
        if self._pose_neutralized:
            return self._analyze_mar_only(mar_mean)

        return self._analyze_full(
            pitch_mean, pitch_std, yaw_std, roll_std, mar_mean
        )

    def _analyze_mar_only(self, mar_mean: float) -> AgentOpinion:
        """Análise quando pose está neutralizada (produção)."""
        # MAR Z-normed: > 1.0 = boca mais aberta que treino (bocejo?)
        mar_score = float(np.clip(mar_mean / 3.0, 0.0, 1.0))

        if mar_score >= 0.5:
            signal = FatigueSignal.MODERATE
        elif mar_score >= 0.2:
            signal = FatigueSignal.MILD
        else:
            signal = FatigueSignal.CLEAR

        reasons = []
        if mar_mean > 1.0:
            reasons.append(f"MAR elevado (z={mar_mean:.2f}) — possível bocejo")

        return AgentOpinion(
            agent_name=self._name,
            signal=signal,
            confidence=0.5,  # baixa: apenas MAR disponível
            score=mar_score,
            reasoning="; ".join(reasons) if reasons else "MAR normal, pose neutralizada",
            key_indicators={"mar_mean": mar_mean},
        )

    def _analyze_full(
        self,
        pitch_mean: float, pitch_std: float,
        yaw_std: float, roll_std: float,
        mar_mean: float,
    ) -> AgentOpinion:
        """Análise completa com pose (modo lab)."""
        # Pitch negativo = cabeça caindo (head-nod)
        nod_score = float(np.clip(-pitch_mean / 3.0, 0.0, 1.0))

        # Instabilidade postural
        instab_score = float(np.clip(
            (pitch_std + yaw_std + roll_std) / 6.0, 0.0, 1.0
        ))

        # MAR (bocejo)
        mar_score = float(np.clip(mar_mean / 3.0, 0.0, 1.0))

        score = float(
            0.40 * nod_score
            + 0.30 * instab_score
            + 0.30 * mar_score
        )
        score = float(np.clip(score, 0.0, 1.0))

        if score >= 0.65:
            signal = FatigueSignal.SEVERE
        elif score >= 0.40:
            signal = FatigueSignal.MODERATE
        elif score >= 0.18:
            signal = FatigueSignal.MILD
        else:
            signal = FatigueSignal.CLEAR

        reasons = []
        if pitch_mean < -1.0:
            reasons.append(f"Head-nod (pitch z={pitch_mean:.2f})")
        if pitch_std > 1.5:
            reasons.append(f"Instabilidade postural (pitch_std={pitch_std:.2f})")
        if mar_mean > 1.0:
            reasons.append(f"Bocejo (MAR z={mar_mean:.2f})")

        confidence = 0.85
        if abs(pitch_mean) < 0.3 and pitch_std < 0.5:
            confidence = 0.7  # pouco movimento — dados limitados

        return AgentOpinion(
            agent_name=self._name,
            signal=signal,
            confidence=confidence,
            score=score,
            reasoning="; ".join(reasons) if reasons else "Postura estável",
            key_indicators={
                "pitch_mean": pitch_mean,
                "pitch_std": pitch_std,
                "mar_mean": mar_mean,
            },
        )


# ── A5: SupervisorAgent ─────────────────────────────────────────────────


@dataclass
class SupervisorDecision:
    """Decisão final do Supervisor."""
    label: str                         # "Safe" | "Danger"
    combined_score: float              # [0.0, 1.0] — score agregado
    alert_level_suggestion: str        # "SAFE" | "WATCH" | "DANGER" | "CRITICAL"
    dominant_agent: str                # agente com maior contribuição
    agent_agreement: float             # [0.0, 1.0] — grau de concordância
    opinions: List[AgentOpinion]       # opiniões originais
    reasoning: str                     # explicação agregada
    fatigue_type: str                  # "ocular" | "behavioral" | "postural" | "mixed" | "none"


@dataclass
class SupervisorConfig:
    """Pesos dos agentes e thresholds do Supervisor."""
    # Pesos base dos agentes (somam 1.0)
    ocular_weight: float = 0.45       # Mais peso: indicador mais imediato
    blink_weight: float = 0.35        # Segundo: degradação progressiva
    postural_weight: float = 0.20     # Menor: neutralizado em produção

    # Thresholds de decisão
    danger_threshold: float = 0.45    # score >= threshold → Danger
    watch_threshold: float = 0.25     # score >= threshold → Watch
    critical_threshold: float = 0.70  # score >= threshold → Critical

    # Boost de convergência: se 2+ agentes concordam, amplificar
    convergence_boost: float = 0.10   # +10% ao score se convergência


class SupervisorAgent:
    """
    A5: Supervisor — agrega opiniões dos especialistas.

    Estratégia de agregação:
    1. Weighted average dos scores, ponderado por (weight × confidence)
    2. Convergence boost: se 2+ agentes reportam MODERATE+, soma +10%
    3. Dominant agent: identifica qual agente contribuiu mais
    4. Fatigue type: classifica o tipo dominante de fadiga

    O Supervisor NÃO substitui o modelo MLP — opera em paralelo.
    O resultado final combina a predição do MLP com a análise multi-agente.
    """

    def __init__(
        self,
        ocular: OcularAgent,
        blink: BlinkAgent,
        postural: PosturalAgent,
        config: Optional[SupervisorConfig] = None,
    ) -> None:
        self._ocular = ocular
        self._blink = blink
        self._postural = postural
        self.cfg = config or SupervisorConfig()

    def decide(
        self, window_feats: Dict[str, float]
    ) -> SupervisorDecision:
        """
        Consulta os 3 agentes e agrega suas opiniões.

        Pipeline:
        1. Cada agente analisa window_feats independentemente
        2. Weighted average dos scores × confidences
        3. Convergence boost se houver concordância
        4. Classificação final
        """
        # 1. Coletar opiniões
        op_ocular = self._ocular.analyze(window_feats)
        op_blink = self._blink.analyze(window_feats)
        op_postural = self._postural.analyze(window_feats)
        opinions = [op_ocular, op_blink, op_postural]

        # 2. Weighted average (score × confidence × weight)
        weights = {
            op_ocular.agent_name: self.cfg.ocular_weight,
            op_blink.agent_name: self.cfg.blink_weight,
            op_postural.agent_name: self.cfg.postural_weight,
        }

        total_weight = 0.0
        weighted_sum = 0.0
        contributions = {}

        for op in opinions:
            w = weights[op.agent_name]
            effective_w = w * op.confidence
            contribution = op.score * effective_w
            weighted_sum += contribution
            total_weight += effective_w
            contributions[op.agent_name] = contribution

        combined_score = weighted_sum / max(total_weight, 1e-8)

        # 3. Convergence boost
        agents_above_moderate = sum(
            1 for op in opinions
            if op.signal.value >= FatigueSignal.MODERATE.value
        )
        if agents_above_moderate >= 2:
            combined_score = min(
                combined_score + self.cfg.convergence_boost, 1.0
            )

        combined_score = float(np.clip(combined_score, 0.0, 1.0))

        # 4. Dominant agent
        dominant = max(contributions, key=contributions.get)

        # 5. Fatigue type
        fatigue_type = self._classify_fatigue_type(
            op_ocular, op_blink, op_postural
        )

        # 6. Agreement (1.0 se todos iguais, 0.0 se totalmente divergentes)
        signals = [op.signal.value for op in opinions]
        signal_range = max(signals) - min(signals)
        agreement = 1.0 - min(signal_range / 4.0, 1.0)

        # 7. Label e alert level
        if combined_score >= self.cfg.critical_threshold:
            label = "Danger"
            alert = "CRITICAL"
        elif combined_score >= self.cfg.danger_threshold:
            label = "Danger"
            alert = "DANGER"
        elif combined_score >= self.cfg.watch_threshold:
            label = "Safe"
            alert = "WATCH"
        else:
            label = "Safe"
            alert = "SAFE"

        # 8. Reasoning agregado
        agent_reasons = [
            f"[{op.agent_name}:{op.signal.name}] {op.reasoning}"
            for op in opinions
            if op.signal.value >= FatigueSignal.MILD.value
        ]
        if not agent_reasons:
            reasoning = "Todos os agentes reportam sinais normais"
        else:
            reasoning = " | ".join(agent_reasons)

        return SupervisorDecision(
            label=label,
            combined_score=combined_score,
            alert_level_suggestion=alert,
            dominant_agent=dominant,
            agent_agreement=agreement,
            opinions=opinions,
            reasoning=reasoning,
            fatigue_type=fatigue_type,
        )

    @staticmethod
    def _classify_fatigue_type(
        ocular: AgentOpinion,
        blink: AgentOpinion,
        postural: AgentOpinion,
    ) -> str:
        """Classifica o tipo dominante de fadiga."""
        above = {
            "ocular": ocular.signal.value >= FatigueSignal.MODERATE.value,
            "behavioral": blink.signal.value >= FatigueSignal.MODERATE.value,
            "postural": postural.signal.value >= FatigueSignal.MODERATE.value,
        }
        active = [k for k, v in above.items() if v]

        if len(active) == 0:
            return "none"
        if len(active) >= 2:
            return "mixed"
        return active[0]
