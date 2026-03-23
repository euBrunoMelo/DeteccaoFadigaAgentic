"""Testes unitários para o módulo memory (M1-M3)."""

import json
import os
import tempfile
import time

import numpy as np
import pytest

from SALTE_INFERENCE.memory import (
    FeatureLogger,
    OperatorStore,
    SessionMemory,
)


# ── Helpers ───────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "ear_mean", "ear_std", "ear_min", "ear_vel_mean", "ear_vel_std",
    "mar_mean", "pitch_mean", "pitch_std", "yaw_std", "roll_std",
    "blink_count", "blink_rate_per_min", "blink_mean_dur_ms",
    "perclos_p80_mean", "perclos_p80_max",
    "blink_closing_vel_mean", "blink_opening_vel_mean",
    "long_blink_pct", "blink_regularity",
]


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


# ── M1: SessionMemory ───────────────────────────────────────────────────


class TestSessionMemoryM1:

    def test_initial_state(self):
        s = SessionMemory(operator_id="test-01")
        assert s.operator_id == "test-01"
        assert s.total_windows == 0
        assert s.danger_ratio == 0.0

    def test_on_window_safe(self):
        s = SessionMemory()
        s.on_window("Safe", 0.2, "SAFE", 0.05, -0.5, 0.0, 0.0)
        assert s.total_windows == 1
        assert s.total_safe_windows == 1
        assert s.total_danger_windows == 0

    def test_on_window_danger(self):
        s = SessionMemory()
        s.on_window("Danger", 0.8, "DANGER", 0.4, -2.0, 0.0, 0.0)
        assert s.total_danger_windows == 1
        assert s.current_consecutive_danger == 1

    def test_danger_ratio(self):
        s = SessionMemory()
        for _ in range(3):
            s.on_window("Danger", 0.8, "DANGER", 0.4, -2.0, 0.0, 0.0)
        for _ in range(7):
            s.on_window("Safe", 0.2, "SAFE", 0.05, -0.5, 0.0, 0.0)
        assert abs(s.danger_ratio - 0.3) < 0.001

    def test_max_consecutive_danger(self):
        s = SessionMemory()
        for _ in range(5):
            s.on_window("Danger", 0.8, "DANGER", 0.4, -2.0, 0.0, 0.0)
        s.on_window("Safe", 0.2, "SAFE", 0.05, -0.5, 0.0, 0.0)
        for _ in range(3):
            s.on_window("Danger", 0.8, "DANGER", 0.4, -2.0, 0.0, 0.0)
        assert s.max_consecutive_danger == 5
        assert s.current_consecutive_danger == 3

    def test_watch_and_critical_counters(self):
        s = SessionMemory()
        s.on_window("Safe", 0.35, "WATCH", 0.1, -0.5, 0.0, 0.0)
        s.on_window("Danger", 0.9, "CRITICAL", 0.5, -3.0, 2.0, 500.0)
        assert s.total_watch_windows == 1
        assert s.total_critical_windows == 1

    def test_microsleep_tracking(self):
        s = SessionMemory()
        s.on_window("Danger", 0.9, "CRITICAL", 0.5, -3.0, 2.0, 500.0)
        assert len(s.microsleep_timestamps) == 2
        assert s.total_microsleep_ms == 500.0

    def test_calibration_tracking(self):
        s = SessionMemory()
        s.on_calibration(0.28, "accept")
        s.on_calibration(0.30, "use_with_caution")
        assert s.calibration_count == 2
        assert s.last_calibration_ear_mean == 0.30
        assert s.last_calibration_verdict == "use_with_caution"

    def test_alert_tracking(self):
        s = SessionMemory()
        s.on_alert(triggered=True)
        s.on_alert(triggered=False)
        s.on_alert(triggered=True)
        assert s.total_alerts_triggered == 2
        assert s.total_alerts_suppressed == 1

    def test_auto_recalibration_tracking(self):
        s = SessionMemory()
        s.on_auto_recalibration()
        s.on_auto_recalibration()
        assert s.total_recalibrations_auto == 2

    def test_perclos_trend_slope_increasing(self):
        s = SessionMemory()
        for i in range(20):
            perclos = 0.05 + i * 0.02  # increasing
            s.on_window("Safe", 0.2, "SAFE", perclos, -0.5, 0.0, 0.0)
        assert s.perclos_trend_slope > 0

    def test_perclos_trend_slope_stable(self):
        s = SessionMemory()
        for _ in range(20):
            s.on_window("Safe", 0.2, "SAFE", 0.1, -0.5, 0.0, 0.0)
        assert abs(s.perclos_trend_slope) < 0.001

    def test_perclos_trend_slope_insufficient_data(self):
        s = SessionMemory()
        for _ in range(5):
            s.on_window("Safe", 0.2, "SAFE", 0.1, -0.5, 0.0, 0.0)
        assert s.perclos_trend_slope == 0.0

    def test_avg_prob_danger(self):
        s = SessionMemory()
        s.on_window("Safe", 0.2, "SAFE", 0.1, -0.5, 0.0, 0.0)
        s.on_window("Danger", 0.8, "DANGER", 0.4, -2.0, 0.0, 0.0)
        assert abs(s.avg_prob_danger - 0.5) < 0.001

    def test_summary_contains_all_keys(self):
        s = SessionMemory(operator_id="test-sum")
        s.on_window("Safe", 0.2, "SAFE", 0.1, -0.5, 0.0, 0.0)
        summary = s.summary()
        expected_keys = {
            "operator_id", "duration_min", "total_windows",
            "danger_ratio", "avg_prob", "max_consec_danger",
            "total_microsleeps", "total_microsleep_ms",
            "perclos_trend", "calibrations",
            "auto_recalibrations", "alerts_triggered",
            "c29_overrides", "c29_boosts", "c29_boost_total",
        }
        assert expected_keys == set(summary.keys())
        assert summary["operator_id"] == "test-sum"
        assert summary["total_windows"] == 1


# ── M2: OperatorStore ───────────────────────────────────────────────────


class TestOperatorStoreM2:

    @pytest.fixture
    def store(self, tmp_path):
        db = str(tmp_path / "test_operators.db")
        s = OperatorStore(db_path=db)
        yield s
        s.close()

    def test_profile_not_found(self, store):
        assert store.get_profile("nonexistent") is None

    def test_upsert_creates_profile(self, store):
        b = _FakeBaseline()
        store.upsert_profile_from_calibration("op-1", b)
        p = store.get_profile("op-1")
        assert p is not None
        assert abs(p["ear_mean"] - 0.28) < 0.001

    def test_upsert_updates_with_ema(self, store):
        b1 = _FakeBaseline(ear_mean=0.28)
        store.upsert_profile_from_calibration("op-1", b1)

        b2 = _FakeBaseline(ear_mean=0.32)
        store.upsert_profile_from_calibration("op-1", b2)

        p = store.get_profile("op-1")
        # EMA: 0.3 * 0.32 + 0.7 * 0.28 = 0.292
        assert abs(p["ear_mean"] - 0.292) < 0.001

    def test_save_and_get_session(self, store):
        b = _FakeBaseline()
        store.upsert_profile_from_calibration("op-1", b)

        summary = {
            "duration_min": 30.0,
            "total_windows": 100,
            "danger_ratio": 0.15,
            "avg_prob": 0.3,
            "max_consec_danger": 5,
            "total_microsleeps": 2,
            "total_microsleep_ms": 400.0,
            "perclos_trend": 0.001,
            "calibrations": 1,
        }
        store.save_session("op-1", summary, "2026-03-11T10:00:00")

        sessions = store.get_recent_sessions("op-1")
        assert len(sessions) == 1
        assert sessions[0]["total_windows"] == 100
        assert sessions[0]["danger_ratio"] == 0.15

    def test_warm_start_requires_2_sessions(self, store):
        b = _FakeBaseline()
        store.upsert_profile_from_calibration("op-1", b)

        # No sessions yet
        assert store.get_warm_start_baseline("op-1") is None

        # Save 1 session
        store.save_session("op-1", {"danger_ratio": 0.1, "duration_min": 30}, "t1")
        assert store.get_warm_start_baseline("op-1") is None

        # Save 2nd session
        store.save_session("op-1", {"danger_ratio": 0.2, "duration_min": 25}, "t2")
        ws = store.get_warm_start_baseline("op-1")
        assert ws is not None
        assert "ear_mean" in ws

    def test_multiple_sessions_ordered(self, store):
        b = _FakeBaseline()
        store.upsert_profile_from_calibration("op-1", b)

        for i in range(5):
            store.save_session(
                "op-1",
                {"danger_ratio": i * 0.1, "duration_min": 10 + i},
                f"2026-03-11T{10 + i}:00:00",
            )

        sessions = store.get_recent_sessions("op-1", limit=3)
        assert len(sessions) == 3

    def test_profile_sessions_count(self, store):
        b = _FakeBaseline()
        store.upsert_profile_from_calibration("op-1", b)
        store.save_session("op-1", {"danger_ratio": 0.1, "duration_min": 30}, "t1")
        store.save_session("op-1", {"danger_ratio": 0.2, "duration_min": 40}, "t2")

        p = store.get_profile("op-1")
        assert p["sessions_count"] == 2


# ── M3: FeatureLogger ───────────────────────────────────────────────────


class TestFeatureLoggerM3:

    def test_disabled_creates_no_file(self, tmp_path):
        fl = FeatureLogger(
            output_dir=str(tmp_path),
            feature_names=FEATURE_NAMES,
            enabled=False,
        )
        fl.log(1000.0, "Safe", 0.2, "SAFE", "high", {})
        fl.close()
        assert len(list(tmp_path.iterdir())) == 0

    def test_enabled_creates_csv(self, tmp_path):
        fl = FeatureLogger(
            output_dir=str(tmp_path),
            feature_names=FEATURE_NAMES,
            operator_id="op-test",
            enabled=True,
        )
        fl.log(1000.0, "Safe", 0.2, "SAFE", "high", {"ear_mean": -0.5})
        fl.log(2000.0, "Danger", 0.8, "DANGER", "low", {"ear_mean": -2.0})
        fl.close()

        csvs = list(tmp_path.glob("*.csv"))
        assert len(csvs) == 1
        assert "op-test" in csvs[0].name
        assert fl.count == 2

    def test_csv_header_correct(self, tmp_path):
        fl = FeatureLogger(
            output_dir=str(tmp_path),
            feature_names=FEATURE_NAMES,
            enabled=True,
        )
        fl.close()

        csvs = list(tmp_path.glob("*.csv"))
        with open(csvs[0], "r") as f:
            header = f.readline().strip().split(",")
        assert header[:6] == [
            "timestamp_ms", "operator_id", "label",
            "prob_danger", "alert_level", "confidence",
        ]
        assert len(header) == 6 + len(FEATURE_NAMES)

    def test_log_when_disabled_is_noop(self):
        fl = FeatureLogger(
            output_dir="/tmp/fake",
            feature_names=FEATURE_NAMES,
            enabled=False,
        )
        fl.log(1000.0, "Safe", 0.2, "SAFE", "high", {})
        assert fl.count == 0
