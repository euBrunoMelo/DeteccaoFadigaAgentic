"""Testes unitários para o módulo reflection (R1-R3)."""

import time
import pytest
import numpy as np

from SALTE_INFERENCE.reflection import (
    AutoRecalibrationManager,
    DriftReflector,
    DriftReflectorConfig,
    DriftReport,
    DriftStatus,
    PredictionReflection,
    PredictionReflector,
    RecalibrationDecision,
)


# ── Helpers ───────────────────────────────────────────────────────────────


def _make_training_stats():
    """Cria training_stats simulando inference_config.json."""
    return {
        "ear_mean": {"mean": 0.0, "std": 1.0, "min": -15.0, "max": 4.5},
        "ear_std": {"mean": 0.0, "std": 1.0, "min": -1.7, "max": 13.0},
        "mar_mean": {"mean": 0.0, "std": 1.0, "min": -1.3, "max": 23.5},
        "pitch_mean": {"mean": 0.0, "std": 1.0, "min": -8.6, "max": 8.2},
        "pitch_std": {"mean": 0.0, "std": 1.0, "min": -0.7, "max": 25.3},
        "yaw_std": {"mean": 0.0, "std": 1.0, "min": -0.6, "max": 16.3},
    }


def _make_window(ear_mean=0.0, **overrides):
    """Cria uma janela de features simulada."""
    defaults = {
        "ear_mean": ear_mean,
        "ear_std": 0.5,
        "mar_mean": 0.3,
        "pitch_mean": 0.1,
        "pitch_std": 0.2,
        "yaw_std": 0.3,
    }
    defaults.update(overrides)
    return defaults


# ── R1: DriftReflector ───────────────────────────────────────────────────


class TestDriftReflectorR1:

    def test_stable_data_reports_stable(self):
        dr = DriftReflector(
            _make_training_stats(),
            DriftReflectorConfig(analysis_interval=5, window_buffer_size=10),
        )
        report = None
        for _ in range(10):
            report = dr.push_window(_make_window(ear_mean=0.0))
        assert report is not None
        assert report.status == DriftStatus.STABLE
        assert report.recommendation == "continue"

    def test_drifted_ear_mean(self):
        dr = DriftReflector(
            _make_training_stats(),
            DriftReflectorConfig(analysis_interval=5, window_buffer_size=10),
        )
        report = None
        for _ in range(10):
            report = dr.push_window(_make_window(ear_mean=3.0))
        assert report is not None
        assert report.status == DriftStatus.DRIFTED
        assert "ear_mean" in report.drifted_features
        assert report.recommendation == "recalibrate"

    def test_critical_drift(self):
        dr = DriftReflector(
            _make_training_stats(),
            DriftReflectorConfig(analysis_interval=5, window_buffer_size=10),
        )
        report = None
        for _ in range(10):
            report = dr.push_window(_make_window(ear_mean=5.0))
        assert report is not None
        assert report.status == DriftStatus.CRITICAL
        assert report.recommendation == "suspend"

    def test_warning_drift(self):
        dr = DriftReflector(
            _make_training_stats(),
            DriftReflectorConfig(analysis_interval=5, window_buffer_size=10),
        )
        report = None
        for _ in range(10):
            report = dr.push_window(_make_window(ear_mean=1.8))
        assert report is not None
        assert report.status == DriftStatus.WARNING
        assert report.recommendation == "continue"

    def test_no_report_before_interval(self):
        dr = DriftReflector(
            _make_training_stats(),
            DriftReflectorConfig(analysis_interval=10),
        )
        for _ in range(9):
            report = dr.push_window(_make_window())
        assert report is None

    def test_zscore_math_correct(self):
        stats = _make_training_stats()
        stats["ear_mean"] = {"mean": 2.0, "std": 0.5, "min": 0, "max": 5}
        dr = DriftReflector(
            stats,
            DriftReflectorConfig(analysis_interval=5, window_buffer_size=10),
        )
        for _ in range(10):
            report = dr.push_window(_make_window(ear_mean=3.5))
        # z = (3.5 - 2.0) / 0.5 = 3.0 -> DRIFTED
        assert report is not None
        assert abs(report.details["ear_mean"] - 3.0) < 0.01


# ── R2: PredictionReflector ──────────────────────────────────────────────


class TestPredictionReflectorR2:

    def test_stable_predictions(self):
        pr = PredictionReflector()
        for _ in range(10):
            r = pr.push(0.2, "Safe")
        assert r.pattern == "stable"
        assert r.confidence_modifier == 1.0

    def test_stuck_danger(self):
        pr = PredictionReflector()
        for _ in range(20):
            r = pr.push(0.8, "Danger")
        assert r.pattern == "stuck_danger"
        assert r.confidence_modifier == 0.7
        assert r.consecutive_danger == 20

    def test_oscillating_pattern(self):
        pr = PredictionReflector()
        for i in range(20):
            label = "Danger" if i % 2 == 0 else "Safe"
            prob = 0.45 if label == "Danger" else 0.35
            r = pr.push(prob, label)
        assert r.pattern == "oscillating"
        assert r.confidence_modifier == 0.5

    def test_sudden_transition(self):
        pr = PredictionReflector()
        # Build up buffer first
        for _ in range(10):
            pr.push(0.15, "Safe")
        # Then jump
        pr.push(0.15, "Safe")
        pr.push(0.35, "Safe")
        r = pr.push(0.75, "Danger")
        assert r.pattern == "sudden_transition"
        assert r.confidence_modifier == 0.8

    def test_insufficient_data(self):
        pr = PredictionReflector()
        r = pr.push(0.5, "Danger")
        assert r.pattern == "stable"
        assert r.suggestion == "Dados insuficientes para reflexão"


# ── R3: AutoRecalibrationManager ─────────────────────────────────────────


class TestAutoRecalibrationR3:

    def test_critical_drift_triggers_immediate(self):
        mgr = AutoRecalibrationManager(min_recal_interval_sec=0)
        report = DriftReport(
            status=DriftStatus.CRITICAL,
            drifted_features=["ear_mean"],
            details={"ear_mean": 5.0},
            recommendation="suspend",
            windows_analyzed=10,
        )
        dec = mgr.evaluate(drift_report=report, pred_reflection=None)
        assert dec.should_recalibrate is True
        assert dec.urgency == "immediate"

    def test_drifted_plus_stuck_triggers_immediate(self):
        mgr = AutoRecalibrationManager(min_recal_interval_sec=0)
        report = DriftReport(
            status=DriftStatus.DRIFTED,
            drifted_features=["ear_mean"],
            details={"ear_mean": 3.0},
            recommendation="recalibrate",
            windows_analyzed=10,
        )
        pred = PredictionReflection(
            pattern="stuck_danger",
            confidence_modifier=0.7,
            suggestion="...",
            consecutive_danger=20,
            consecutive_safe=0,
            recent_prob_mean=0.8,
            recent_prob_std=0.05,
        )
        dec = mgr.evaluate(drift_report=report, pred_reflection=pred)
        assert dec.should_recalibrate is True
        assert dec.urgency == "immediate"

    def test_drifted_alone_is_scheduled(self):
        mgr = AutoRecalibrationManager(min_recal_interval_sec=0)
        report = DriftReport(
            status=DriftStatus.DRIFTED,
            drifted_features=["ear_mean"],
            details={"ear_mean": 3.0},
            recommendation="recalibrate",
            windows_analyzed=10,
        )
        dec = mgr.evaluate(drift_report=report, pred_reflection=None)
        assert dec.should_recalibrate is True
        assert dec.urgency == "scheduled"

    def test_stuck_danger_without_drift_no_recal(self):
        mgr = AutoRecalibrationManager(min_recal_interval_sec=0)
        report = DriftReport(
            status=DriftStatus.STABLE,
            drifted_features=[],
            details={},
            recommendation="continue",
            windows_analyzed=10,
        )
        pred = PredictionReflection(
            pattern="stuck_danger",
            confidence_modifier=0.7,
            suggestion="...",
            consecutive_danger=20,
            consecutive_safe=0,
            recent_prob_mean=0.8,
            recent_prob_std=0.05,
        )
        dec = mgr.evaluate(drift_report=report, pred_reflection=pred)
        assert dec.should_recalibrate is False

    def test_cooldown_respected(self):
        mgr = AutoRecalibrationManager(min_recal_interval_sec=300)
        report = DriftReport(
            status=DriftStatus.CRITICAL,
            drifted_features=["ear_mean"],
            details={"ear_mean": 5.0},
            recommendation="suspend",
            windows_analyzed=10,
        )
        dec1 = mgr.evaluate(drift_report=report, pred_reflection=None)
        assert dec1.should_recalibrate is True

        # Second call within cooldown
        dec2 = mgr.evaluate(drift_report=report, pred_reflection=None)
        assert dec2.should_recalibrate is False

    def test_rate_limit_per_hour(self):
        mgr = AutoRecalibrationManager(
            min_recal_interval_sec=0,
            max_recalibrations_per_hour=2,
        )
        report = DriftReport(
            status=DriftStatus.CRITICAL,
            drifted_features=["ear_mean"],
            details={"ear_mean": 5.0},
            recommendation="suspend",
            windows_analyzed=10,
        )
        mgr.evaluate(drift_report=report, pred_reflection=None)
        mgr.evaluate(drift_report=report, pred_reflection=None)
        dec = mgr.evaluate(drift_report=report, pred_reflection=None)
        assert dec.should_recalibrate is False
        assert "Limite" in dec.reason

    def test_no_signals_no_recal(self):
        mgr = AutoRecalibrationManager(min_recal_interval_sec=0)
        dec = mgr.evaluate(drift_report=None, pred_reflection=None)
        assert dec.should_recalibrate is False
