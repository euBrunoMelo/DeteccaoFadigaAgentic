"""Testes unitários para o módulo agents (A1-A5)."""

import pytest

from SALTE_INFERENCE.agents import (
    AgentOpinion,
    BlinkAgent,
    FatigueSignal,
    OcularAgent,
    PosturalAgent,
    SupervisorAgent,
    SupervisorConfig,
    SupervisorDecision,
)


# ── Helpers ───────────────────────────────────────────────────────────────


def _normal_feats() -> dict:
    """Features normais (todos Z-normed próximos de zero)."""
    return {
        "ear_mean": 0.0,
        "ear_std": 0.5,
        "ear_min": 0.0,
        "ear_vel_mean": 0.0,
        "ear_vel_std": 0.0,
        "perclos_p80_mean": 0.05,
        "perclos_p80_max": 0.1,
        "blink_count": 5.0,
        "blink_rate_per_min": 15.0,
        "blink_mean_dur_ms": 200.0,
        "blink_closing_vel_mean": 0.0,
        "blink_opening_vel_mean": 0.0,
        "long_blink_pct": 0.1,
        "blink_regularity": 0.3,
        "pitch_mean": 0.0,
        "pitch_std": 0.3,
        "yaw_std": 0.3,
        "roll_std": 0.2,
        "mar_mean": 0.0,
    }


def _fatigued_feats() -> dict:
    """Features de fadiga severa."""
    return {
        "ear_mean": -3.0,
        "ear_std": 0.5,
        "ear_min": -4.0,
        "ear_vel_mean": -1.0,
        "ear_vel_std": 0.5,
        "perclos_p80_mean": 0.5,
        "perclos_p80_max": 0.6,
        "blink_count": 10.0,
        "blink_rate_per_min": 8.0,
        "blink_mean_dur_ms": 800.0,
        "blink_closing_vel_mean": -1.5,
        "blink_opening_vel_mean": -1.0,
        "long_blink_pct": 0.7,
        "blink_regularity": 1.2,
        "pitch_mean": -2.5,
        "pitch_std": 2.0,
        "yaw_std": 1.5,
        "roll_std": 1.0,
        "mar_mean": 2.0,
    }


# ── A1: AgentOpinion ────────────────────────────────────────────────────


class TestAgentOpinionA1:

    def test_valid_opinion(self):
        op = AgentOpinion(
            agent_name="Test",
            signal=FatigueSignal.CLEAR,
            confidence=0.9,
            score=0.1,
            reasoning="OK",
            key_indicators={"ear_mean": 0.0},
        )
        assert op.agent_name == "Test"
        assert op.signal == FatigueSignal.CLEAR
        assert op.confidence == 0.9
        assert op.score == 0.1

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            AgentOpinion(
                agent_name="Test",
                signal=FatigueSignal.CLEAR,
                confidence=1.5,
                score=0.5,
                reasoning="bad",
                key_indicators={},
            )

    def test_invalid_score_raises(self):
        with pytest.raises(ValueError, match="score"):
            AgentOpinion(
                agent_name="Test",
                signal=FatigueSignal.CLEAR,
                confidence=0.5,
                score=-0.1,
                reasoning="bad",
                key_indicators={},
            )

    def test_fatigue_signal_ordering(self):
        assert FatigueSignal.CLEAR.value < FatigueSignal.MILD.value
        assert FatigueSignal.MILD.value < FatigueSignal.MODERATE.value
        assert FatigueSignal.MODERATE.value < FatigueSignal.SEVERE.value
        assert FatigueSignal.SEVERE.value < FatigueSignal.CRITICAL.value


# ── A2: OcularAgent ─────────────────────────────────────────────────────


class TestOcularAgentA2:

    def test_clear_normal_eyes(self):
        agent = OcularAgent()
        op = agent.analyze(_normal_feats())
        assert op.signal == FatigueSignal.CLEAR
        assert op.score < 0.15
        assert op.reasoning == "Sinais oculares normais"

    def test_critical_closed_eyes_high_perclos(self):
        agent = OcularAgent()
        feats = {
            "ear_mean": -3.0,
            "ear_std": 0.5,
            "ear_min": -4.0,
            "perclos_p80_mean": 0.5,
            "perclos_p80_max": 0.6,
        }
        op = agent.analyze(feats)
        assert op.signal == FatigueSignal.CRITICAL
        assert op.score > 0.75

    def test_moderate_perclos_only(self):
        agent = OcularAgent()
        feats = _normal_feats()
        feats["perclos_p80_mean"] = 0.25
        feats["perclos_p80_max"] = 0.35
        op = agent.analyze(feats)
        assert op.signal.value >= FatigueSignal.MILD.value

    def test_low_confidence_low_ear_std(self):
        agent = OcularAgent()
        feats = _normal_feats()
        feats["ear_std"] = 0.05  # very low variation
        op = agent.analyze(feats)
        assert op.confidence == 0.6

    def test_reasoning_lists_indicators(self):
        agent = OcularAgent()
        feats = _normal_feats()
        feats["ear_mean"] = -1.5
        feats["perclos_p80_mean"] = 0.2
        op = agent.analyze(feats)
        assert "EAR abaixo do baseline" in op.reasoning
        assert "PERCLOS elevado" in op.reasoning

    def test_score_bounds(self):
        agent = OcularAgent()
        # All zeros
        op1 = agent.analyze({})
        assert 0.0 <= op1.score <= 1.0
        # Extreme values
        op2 = agent.analyze(_fatigued_feats())
        assert 0.0 <= op2.score <= 1.0


# ── A3: BlinkAgent ──────────────────────────────────────────────────────


class TestBlinkAgentA3:

    def test_clear_normal_blinks(self):
        agent = BlinkAgent()
        feats = _normal_feats()
        feats["blink_count"] = 10.0
        feats["long_blink_pct"] = 0.1
        feats["blink_mean_dur_ms"] = 200.0
        op = agent.analyze(feats)
        assert op.signal == FatigueSignal.CLEAR
        assert op.score < 0.20

    def test_severe_long_slow_blinks(self):
        agent = BlinkAgent()
        feats = {
            "blink_count": 10.0,
            "blink_rate_per_min": 15.0,
            "blink_mean_dur_ms": 800.0,
            "blink_closing_vel_mean": -1.5,
            "blink_opening_vel_mean": -1.0,
            "long_blink_pct": 0.7,
            "blink_regularity": 1.2,
        }
        op = agent.analyze(feats)
        assert op.signal == FatigueSignal.SEVERE
        assert op.score >= 0.70

    def test_no_blinks_insufficient_data(self):
        agent = BlinkAgent()
        feats = _normal_feats()
        feats["blink_count"] = 0
        op = agent.analyze(feats)
        assert op.signal == FatigueSignal.MILD
        assert op.confidence == 0.4
        assert "insuficientes" in op.reasoning or "Nenhuma" in op.reasoning

    def test_confidence_scales_with_count(self):
        agent = BlinkAgent()
        feats_low = _normal_feats()
        feats_low["blink_count"] = 2.0
        feats_high = _normal_feats()
        feats_high["blink_count"] = 8.0
        op_low = agent.analyze(feats_low)
        op_high = agent.analyze(feats_high)
        assert op_low.confidence < op_high.confidence

    def test_high_rate_compensation(self):
        agent = BlinkAgent()
        feats = _normal_feats()
        feats["blink_count"] = 10.0
        feats["blink_rate_per_min"] = 40.0
        op = agent.analyze(feats)
        # High rate should contribute to score
        assert op.score > 0.0

    def test_irregular_rhythm(self):
        agent = BlinkAgent()
        feats = _normal_feats()
        feats["blink_count"] = 5.0
        feats["blink_regularity"] = 1.5
        op = agent.analyze(feats)
        assert "irregular" in op.reasoning.lower() or op.score > 0.0


# ── A4: PosturalAgent ───────────────────────────────────────────────────


class TestPosturalAgentA4:

    def test_clear_stable_posture(self):
        agent = PosturalAgent(pose_neutralized=False)
        feats = _normal_feats()
        op = agent.analyze(feats)
        assert op.signal == FatigueSignal.CLEAR
        assert op.reasoning == "Postura estável"

    def test_neutralized_mar_only(self):
        agent = PosturalAgent(pose_neutralized=True)
        feats = _normal_feats()
        feats["mar_mean"] = 2.0
        op = agent.analyze(feats)
        assert op.signal == FatigueSignal.MODERATE
        assert "MAR elevado" in op.reasoning or "bocejo" in op.reasoning

    def test_neutralized_low_confidence(self):
        agent = PosturalAgent(pose_neutralized=True)
        op = agent.analyze(_normal_feats())
        assert op.confidence == 0.5

    def test_full_head_nod(self):
        agent = PosturalAgent(pose_neutralized=False)
        feats = _normal_feats()
        feats["pitch_mean"] = -2.5
        feats["pitch_std"] = 2.0
        op = agent.analyze(feats)
        assert op.signal.value >= FatigueSignal.MODERATE.value
        assert "Head-nod" in op.reasoning

    def test_full_yawn(self):
        agent = PosturalAgent(pose_neutralized=False)
        feats = _normal_feats()
        feats["mar_mean"] = 2.0
        op = agent.analyze(feats)
        assert "Bocejo" in op.reasoning

    def test_full_instability(self):
        agent = PosturalAgent(pose_neutralized=False)
        feats = _normal_feats()
        feats["pitch_mean"] = -1.5
        feats["pitch_std"] = 3.0
        feats["yaw_std"] = 2.0
        feats["roll_std"] = 2.0
        feats["mar_mean"] = 1.5
        op = agent.analyze(feats)
        assert op.signal.value >= FatigueSignal.MODERATE.value


# ── A5: SupervisorAgent ─────────────────────────────────────────────────


class TestSupervisorAgentA5:

    def _make_supervisor(self, pose_neutralized=True):
        ocular = OcularAgent()
        blink = BlinkAgent()
        postural = PosturalAgent(pose_neutralized=pose_neutralized)
        return SupervisorAgent(ocular, blink, postural)

    def test_all_clear_is_safe(self):
        sup = self._make_supervisor()
        decision = sup.decide(_normal_feats())
        assert decision.label == "Safe"
        assert decision.fatigue_type == "none"
        assert decision.alert_level_suggestion == "SAFE"

    def test_ocular_severe_is_danger(self):
        sup = self._make_supervisor()
        feats = _fatigued_feats()
        decision = sup.decide(feats)
        assert decision.label == "Danger"
        assert decision.combined_score >= 0.45

    def test_convergence_boost(self):
        sup = self._make_supervisor(pose_neutralized=False)
        # Feats that make multiple agents report MODERATE+
        feats = _fatigued_feats()
        decision = sup.decide(feats)
        # With convergence, score should be higher
        assert decision.combined_score > 0.45

    def test_dominant_agent_correct(self):
        sup = self._make_supervisor()
        # Strong ocular signal, weak blink/postural
        feats = _normal_feats()
        feats["ear_mean"] = -3.0
        feats["perclos_p80_mean"] = 0.5
        feats["perclos_p80_max"] = 0.6
        feats["ear_min"] = -4.0
        decision = sup.decide(feats)
        assert decision.dominant_agent == "OcularAgent"

    def test_agreement_all_same(self):
        sup = self._make_supervisor()
        decision = sup.decide(_normal_feats())
        # All should be CLEAR or close → high agreement
        assert decision.agent_agreement >= 0.5

    def test_agreement_divergent(self):
        sup = self._make_supervisor()
        # Strong ocular signal but normal blink/postural
        feats = _normal_feats()
        feats["ear_mean"] = -3.0
        feats["perclos_p80_mean"] = 0.5
        feats["perclos_p80_max"] = 0.6
        feats["ear_min"] = -4.0
        decision = sup.decide(feats)
        # OcularAgent high, others low → lower agreement
        assert decision.agent_agreement < 1.0

    def test_fatigue_type_mixed(self):
        sup = self._make_supervisor(pose_neutralized=False)
        feats = _fatigued_feats()
        decision = sup.decide(feats)
        assert decision.fatigue_type == "mixed"

    def test_fatigue_type_single(self):
        sup = self._make_supervisor()
        # Only ocular signal, blink and postural normal
        feats = _normal_feats()
        feats["ear_mean"] = -3.0
        feats["perclos_p80_mean"] = 0.5
        feats["perclos_p80_max"] = 0.6
        feats["ear_min"] = -4.0
        feats["blink_count"] = 10.0
        decision = sup.decide(feats)
        assert decision.fatigue_type == "ocular"
