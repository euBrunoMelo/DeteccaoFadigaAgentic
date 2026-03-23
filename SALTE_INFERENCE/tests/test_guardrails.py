"""Testes unitários para o módulo guardrails (G1-G4)."""

import time
import pytest
import numpy as np

from SALTE_INFERENCE.guardrails import (
    AlertLevel,
    BehaviorGuardRails,
    BehaviorGuardrailConfig,
    CalibrationVerdict,
    FatigueOutput,
    validate_and_wrap,
    validate_calibration,
    _check_feature_ranges,
    _compute_confidence,
)


# ── Helpers ───────────────────────────────────────────────────────────────


def _make_window_feats(**overrides):
    """Cria um dict de 19 features com valores normais."""
    defaults = {
        "ear_mean": -0.5,
        "ear_std": 1.0,
        "ear_min": -2.0,
        "ear_vel_mean": 0.0,
        "ear_vel_std": 1.0,
        "mar_mean": 1.0,
        "pitch_mean": 0.0,
        "pitch_std": 1.0,
        "yaw_std": 1.0,
        "roll_std": 1.0,
        "blink_count": 5.0,
        "blink_rate_per_min": 20.0,
        "blink_mean_dur_ms": 300.0,
        "perclos_p80_mean": 0.1,
        "perclos_p80_max": 0.15,
        "blink_closing_vel_mean": 0.5,
        "blink_opening_vel_mean": 1.0,
        "long_blink_pct": 0.2,
        "blink_regularity": 0.5,
        "microsleep_count": 0.0,
    }
    defaults.update(overrides)
    return defaults


FEATURE_NAMES = [
    "ear_mean", "ear_std", "ear_min", "ear_vel_mean", "ear_vel_std",
    "mar_mean", "pitch_mean", "pitch_std", "yaw_std", "roll_std",
    "blink_count", "blink_rate_per_min", "blink_mean_dur_ms",
    "perclos_p80_mean", "perclos_p80_max",
    "blink_closing_vel_mean", "blink_opening_vel_mean",
    "long_blink_pct", "blink_regularity",
]


class _FakeConfig:
    feature_names = FEATURE_NAMES
    n_features = 19
    threshold = 0.41


class _FakeBaseline:
    def __init__(self, **kwargs):
        defaults = {
            "ear_mean": 0.28, "ear_std": 0.03,
            "mar_mean": 0.1, "mar_std": 0.02,
            "pitch_mean": 0.0, "pitch_std": 5.0,
            "yaw_mean": 0.0, "yaw_std": 5.0,
            "roll_mean": 0.0, "roll_std": 5.0,
            "is_valid": True,
        }
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


# ── G1: FatigueOutput validation ─────────────────────────────────────────


class TestFatigueOutputG1:

    def test_valid_output(self):
        out = FatigueOutput(
            label="Safe", prob_danger=0.3, alert_level=AlertLevel.SAFE,
            features_valid=True, confidence="high",
            window_quality=0.95, timestamp_ms=1000.0,
        )
        assert out.label == "Safe"
        assert out.prob_danger == 0.3

    def test_invalid_prob_raises(self):
        with pytest.raises(ValueError, match="prob_danger"):
            FatigueOutput(
                label="Safe", prob_danger=1.5, alert_level=AlertLevel.SAFE,
                features_valid=True, confidence="high",
                window_quality=1.0, timestamp_ms=1000.0,
            )

    def test_invalid_label_raises(self):
        with pytest.raises(ValueError, match="label"):
            FatigueOutput(
                label="Unknown", prob_danger=0.5, alert_level=AlertLevel.SAFE,
                features_valid=True, confidence="high",
                window_quality=1.0, timestamp_ms=1000.0,
            )

    def test_invalid_window_quality_raises(self):
        with pytest.raises(ValueError, match="window_quality"):
            FatigueOutput(
                label="Safe", prob_danger=0.5, alert_level=AlertLevel.SAFE,
                features_valid=True, confidence="high",
                window_quality=1.5, timestamp_ms=1000.0,
            )


# ── G2: Feature range checks ─────────────────────────────────────────────


class TestFeatureRangesG2:

    def test_all_valid(self):
        feats = _make_window_feats()
        assert _check_feature_ranges(feats, FEATURE_NAMES) is True

    def test_ear_mean_too_low(self):
        feats = _make_window_feats(ear_mean=-20.0)
        assert _check_feature_ranges(feats, FEATURE_NAMES) is False

    def test_blink_count_too_high(self):
        feats = _make_window_feats(blink_count=50.0)
        assert _check_feature_ranges(feats, FEATURE_NAMES) is False

    def test_confidence_high(self):
        feats = _make_window_feats(blink_count=5.0)
        assert _compute_confidence(feats, True) == "high"

    def test_confidence_medium_no_blinks(self):
        feats = _make_window_feats(blink_count=0.0)
        assert _compute_confidence(feats, True) == "medium"

    def test_confidence_low_invalid_features(self):
        feats = _make_window_feats()
        assert _compute_confidence(feats, False) == "low"


# ── G1+G2: validate_and_wrap ─────────────────────────────────────────────


class TestValidateAndWrap:

    def test_safe_output(self):
        feats = _make_window_feats()
        out = validate_and_wrap(
            0.2, "Safe", feats, FEATURE_NAMES, _FakeConfig(), 1000.0,
        )
        assert out.label == "Safe"
        assert out.alert_level == AlertLevel.SAFE
        assert out.confidence == "high"

    def test_danger_with_microsleep_is_critical(self):
        feats = _make_window_feats(microsleep_count=2.0)
        out = validate_and_wrap(
            0.8, "Danger", feats, FEATURE_NAMES, _FakeConfig(), 1000.0,
        )
        assert out.alert_level == AlertLevel.CRITICAL

    def test_watch_zone(self):
        feats = _make_window_feats()
        out = validate_and_wrap(
            0.35, "Safe", feats, FEATURE_NAMES, _FakeConfig(), 1000.0,
        )
        assert out.alert_level == AlertLevel.WATCH

    def test_prob_clamped(self):
        feats = _make_window_feats()
        out = validate_and_wrap(
            -0.1, "Safe", feats, FEATURE_NAMES, _FakeConfig(), 1000.0,
        )
        assert out.prob_danger == 0.0


# ── G3: BehaviorGuardRails ───────────────────────────────────────────────


class TestBehaviorGuardRailsG3:

    def _make_output(self, label="Safe", prob=0.2, alert=AlertLevel.SAFE):
        return FatigueOutput(
            label=label, prob_danger=prob, alert_level=alert,
            features_valid=True, confidence="high",
            window_quality=1.0, timestamp_ms=1000.0,
        )

    def test_grace_period_suppresses_danger(self):
        bg = BehaviorGuardRails()
        bg.on_calibration_complete()

        out1 = self._make_output("Danger", 0.8, AlertLevel.DANGER)
        result = bg.process(out1)
        assert result.alert_level == AlertLevel.WATCH
        assert result.confidence == "low"

        out2 = self._make_output("Danger", 0.8, AlertLevel.DANGER)
        result2 = bg.process(out2)
        assert result2.alert_level == AlertLevel.WATCH

        # 3rd window — grace period over
        out3 = self._make_output("Danger", 0.8, AlertLevel.DANGER)
        result3 = bg.process(out3)
        assert result3.alert_level == AlertLevel.DANGER

    def test_consecutive_danger_escalation(self):
        bg = BehaviorGuardRails(BehaviorGuardrailConfig(
            max_consecutive_danger=5
        ))
        for _ in range(5):
            out = self._make_output("Danger", 0.8, AlertLevel.DANGER)
            result = bg.process(out)
        assert result.alert_level == AlertLevel.CRITICAL

    def test_alert_cooldown(self):
        bg = BehaviorGuardRails(BehaviorGuardrailConfig(
            alert_cooldown_sec=60.0
        ))
        out = self._make_output("Danger", 0.8, AlertLevel.DANGER)
        assert bg.should_sound_alert(out) is True
        assert bg.should_sound_alert(out) is False  # within cooldown

    def test_watchdog_ok(self):
        bg = BehaviorGuardRails(BehaviorGuardrailConfig(
            watchdog_timeout_sec=30.0
        ))
        assert bg.check_watchdog() is True


# ── G4: CalibrationVerdict ───────────────────────────────────────────────


class TestCalibrationVerdictG4:

    def test_good_calibration_accepted(self):
        b = _FakeBaseline()
        v = validate_calibration(b)
        assert v.is_acceptable is True
        assert v.recommendation == "accept"
        assert v.issues == []

    def test_ear_mean_too_low_rejected(self):
        b = _FakeBaseline(ear_mean=0.08)
        v = validate_calibration(b)
        assert v.is_acceptable is False
        assert v.recommendation == "retry"

    def test_ear_std_too_low_cautioned(self):
        b = _FakeBaseline(ear_std=0.005)
        v = validate_calibration(b)
        assert v.is_acceptable is True
        assert v.recommendation == "use_with_caution"

    def test_pitch_std_excessive_cautioned(self):
        b = _FakeBaseline(pitch_std=131.0)
        v = validate_calibration(b)
        assert v.recommendation == "use_with_caution"

    def test_invalid_baseline_rejected(self):
        b = _FakeBaseline(is_valid=False)
        v = validate_calibration(b)
        assert v.is_acceptable is False
        assert v.recommendation == "retry"

    def test_yaw_mean_off_center_cautioned(self):
        b = _FakeBaseline(yaw_mean=35.0)
        v = validate_calibration(b)
        assert v.recommendation == "use_with_caution"
