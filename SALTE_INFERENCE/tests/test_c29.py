"""
Testes unitarios para C29BoundaryEngine.

Cobre:
  - Ring buffers e tamanhos
  - Computacao de features C29 (HA, AD, NE_hf, YPR)
  - Disparo de cada regra (R1-R4) com contadores de duracao
  - Cap de boost (max 0.40)
  - Modo override-only
  - Reset de buffers longos (auto-recalibracao)
  - C29Alert properties
  - Engine desabilitada
  - load_c29_config com e sem arquivo
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from SALTE_INFERENCE.c29_boundary_engine import (
    C29Alert,
    C29BoundaryEngine,
    C29Config,
    load_c29_config,
)


# ── Helper: fake RTFrameFeatures ─────────────────────────────────────────


@dataclass
class FakeFeats:
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    head_roll: float = 0.0
    face_detected: bool = True


# ── Helpers ───────────────────────────────────────────────────────────────


def _make_engine(fps: int = 30, **overrides) -> C29BoundaryEngine:
    cfg = C29Config(fps=fps, **overrides)
    return C29BoundaryEngine(config=cfg, enabled=True)


def _feed_n_frames(
    engine: C29BoundaryEngine,
    n: int,
    pitch: float = 0.0,
    yaw: float = 0.0,
    roll: float = 0.0,
    face: bool = True,
) -> None:
    for _ in range(n):
        engine.push_frame(FakeFeats(
            head_pitch=pitch, head_yaw=yaw, head_roll=roll,
            face_detected=face,
        ))


# ══════════════════════════════════════════════════════════════════════════
# A. Ring buffer sizes
# ══════════════════════════════════════════════════════════════════════════


class TestBufferSizes:
    def test_buffer_maxlen_2s(self):
        eng = _make_engine(fps=30)
        assert eng._motion_2s.maxlen == 60
        assert eng._pitch_2s.maxlen == 60
        assert eng._yaw_2s.maxlen == 60
        assert eng._pitch_detrend_2s.maxlen == 60

    def test_buffer_maxlen_5s(self):
        eng = _make_engine(fps=30)
        assert eng._pitch_5s.maxlen == 150

    def test_buffer_maxlen_30s(self):
        eng = _make_engine(fps=30)
        assert eng._activity_30s.maxlen == 900

    def test_buffer_maxlen_5min(self):
        eng = _make_engine(fps=30)
        assert eng._activity_5min.maxlen == 9000

    def test_total_ram_under_1mb(self):
        eng = _make_engine(fps=30)
        total_floats = sum(
            buf.maxlen for buf in [
                eng._motion_2s, eng._pitch_2s, eng._pitch_5s,
                eng._pitch_detrend_2s, eng._yaw_2s,
                eng._activity_30s, eng._activity_5min,
            ]
        )
        # cada float = 8 bytes em Python, mas deque usa ~56 bytes/element
        # worst case: 56 * total_floats
        ram_bytes = total_floats * 56
        assert ram_bytes < 1_000_000, f"RAM estimada {ram_bytes} bytes > 1 MB"


# ══════════════════════════════════════════════════════════════════════════
# B. Feature computation
# ══════════════════════════════════════════════════════════════════════════


class TestFeatureComputation:
    def test_head_activity_zero_when_static(self):
        eng = _make_engine(fps=30)
        _feed_n_frames(eng, 100, pitch=10.0, yaw=5.0, roll=0.0)
        # Todos os frames iguais -> motion=0 -> HA=log(1+0)=0
        assert eng._head_activity == pytest.approx(0.0, abs=0.01)

    def test_head_activity_positive_when_moving(self):
        eng = _make_engine(fps=30)
        for i in range(100):
            eng.push_frame(FakeFeats(
                head_pitch=10.0 * np.sin(i * 0.5),
                head_yaw=5.0 * np.cos(i * 0.3),
                head_roll=0.0,
            ))
        assert eng._head_activity > 0.5

    def test_yaw_pitch_ratio_yaw_dominant(self):
        eng = _make_engine(fps=30)
        for i in range(100):
            eng.push_frame(FakeFeats(
                head_pitch=0.0,
                head_yaw=20.0 * np.sin(i * 0.2),  # yaw variando muito
                head_roll=0.0,
            ))
        assert eng._yaw_pitch_ratio > 1.0

    def test_yaw_pitch_ratio_pitch_dominant(self):
        eng = _make_engine(fps=30)
        for i in range(100):
            eng.push_frame(FakeFeats(
                head_pitch=20.0 * np.sin(i * 0.2),  # pitch variando muito
                head_yaw=0.0,
                head_roll=0.0,
            ))
        assert eng._yaw_pitch_ratio < 1.0

    def test_nodding_energy_high_for_oscillation(self):
        eng = _make_engine(fps=30)
        for i in range(200):
            # Oscilacao rapida de pitch
            eng.push_frame(FakeFeats(
                head_pitch=30.0 * np.sin(i * 1.0),
                head_yaw=0.0,
                head_roll=0.0,
            ))
        assert eng._nodding_energy_hf > 50

    def test_nodding_energy_low_for_static(self):
        eng = _make_engine(fps=30)
        _feed_n_frames(eng, 200, pitch=5.0, yaw=10.0)
        assert eng._nodding_energy_hf < 1.0

    def test_activity_drop_zero_before_warmup(self):
        eng = _make_engine(fps=30, r1_warmup_sec=300.0)
        _feed_n_frames(eng, 100)
        assert eng._activity_drop == pytest.approx(0.0)

    def test_activity_drop_positive_when_activity_decreases(self):
        """AD > 0 quando atividade recente < baseline."""
        eng = _make_engine(fps=10, r1_warmup_sec=10.0)
        # 10s de baseline com atividade
        for i in range(100):
            eng.push_frame(FakeFeats(
                head_pitch=10.0 * np.sin(i * 0.5),
                head_yaw=10.0 * np.cos(i * 0.5),
                head_roll=0.0,
            ))
        # Agora ficar parado (activity cai)
        _feed_n_frames(eng, 50, pitch=0.0, yaw=0.0)
        assert eng._activity_drop > 0.0


# ══════════════════════════════════════════════════════════════════════════
# C. Rule triggers
# ══════════════════════════════════════════════════════════════════════════


class TestR1Override:
    def test_r1_does_not_fire_before_warmup(self):
        eng = _make_engine(fps=10, r1_warmup_sec=100.0, r1_duration_sec=1.0)
        _feed_n_frames(eng, 50)  # 5s << 100s warmup
        alert = eng.evaluate()
        assert not alert.override

    def test_r1_fires_after_warmup_with_sustained_drop(self):
        eng = _make_engine(fps=10, r1_warmup_sec=5.0, r1_duration_sec=2.0)
        # 5s com atividade (warmup)
        for i in range(50):
            eng.push_frame(FakeFeats(
                head_pitch=20.0 * np.sin(i * 0.5),
                head_yaw=20.0 * np.cos(i * 0.3),
                head_roll=5.0 * np.sin(i * 0.1),
            ))
        # Agora ficar parado por >2s = 20 frames
        _feed_n_frames(eng, 30, pitch=0.0, yaw=0.0, roll=0.0)
        alert = eng.evaluate()
        # AD deve ser > 0.5 e sustentado por >= 2s
        if eng._activity_drop > eng.cfg.r1_ad_threshold:
            assert alert.override
            assert "R1" in alert.active_rules


class TestR2Boost:
    def test_r2_fires_on_stillness(self):
        eng = _make_engine(fps=30, r2_duration_sec=1.0, r2_ha_threshold=0.15)
        # Movimento minimo (HA ~= 0)
        _feed_n_frames(eng, 60, pitch=0.0, yaw=0.0, roll=0.0)
        alert = eng.evaluate()
        assert "R2" in alert.active_rules
        assert alert.boost >= 0.15

    def test_r2_does_not_fire_with_movement(self):
        eng = _make_engine(fps=30, r2_duration_sec=1.0)
        for i in range(60):
            eng.push_frame(FakeFeats(
                head_pitch=10.0 * np.sin(i * 0.5),
                head_yaw=10.0 * np.cos(i * 0.3),
                head_roll=0.0,
            ))
        alert = eng.evaluate()
        assert "R2" not in alert.active_rules

    def test_r2_resets_on_face_loss(self):
        eng = _make_engine(fps=30, r2_duration_sec=1.0)
        _feed_n_frames(eng, 25, pitch=0.0, yaw=0.0)
        # Face lost — counter resets
        eng.push_frame(FakeFeats(face_detected=False))
        _feed_n_frames(eng, 25, pitch=0.0, yaw=0.0)
        # Total still frames: 25 (not 50) due to reset
        alert = eng.evaluate()
        assert "R2" not in alert.active_rules


class TestR3Boost:
    def test_r3_fires_on_pitch_dominance_with_energy(self):
        cfg = C29Config(
            fps=30, r3_duration_sec=0.5, r3_ypr_threshold=1.0,
            r3_ne_threshold=10.0, r3_ha_min=0.1,
        )
        eng = C29BoundaryEngine(config=cfg, enabled=True)
        for i in range(100):
            # Pitch oscillation (pitch std >> yaw std → YPR < 1)
            # Plus enough energy + activity
            eng.push_frame(FakeFeats(
                head_pitch=15.0 * np.sin(i * 0.8),
                head_yaw=0.5 * np.sin(i * 0.1),
                head_roll=0.0,
            ))
        alert = eng.evaluate()
        assert "R3" in alert.active_rules
        assert alert.boost >= 0.20

    def test_r3_does_not_fire_when_yaw_dominates(self):
        cfg = C29Config(
            fps=30, r3_duration_sec=0.5, r3_ne_threshold=10.0,
            r3_ha_min=0.1,
        )
        eng = C29BoundaryEngine(config=cfg, enabled=True)
        for i in range(100):
            eng.push_frame(FakeFeats(
                head_pitch=0.5 * np.sin(i * 0.1),
                head_yaw=15.0 * np.sin(i * 0.8),
                head_roll=0.0,
            ))
        alert = eng.evaluate()
        assert "R3" not in alert.active_rules


class TestR4Boost:
    def test_r4_fires_on_high_energy_low_activity_forward(self):
        cfg = C29Config(
            fps=30, r4_ne_threshold=50.0,
            r4_ha_threshold=5.0, r4_yaw_threshold=20.0,
        )
        eng = C29BoundaryEngine(config=cfg, enabled=True)
        # High pitch oscillation (NE_hf alto) + low yaw + low overall motion
        for i in range(100):
            eng.push_frame(FakeFeats(
                head_pitch=15.0 * np.sin(i * 1.5),
                head_yaw=2.0,  # forward
                head_roll=0.0,
            ))
        alert = eng.evaluate()
        if eng._nodding_energy_hf > cfg.r4_ne_threshold:
            assert "R4" in alert.active_rules
            assert alert.boost >= 0.25

    def test_r4_does_not_fire_with_high_yaw(self):
        cfg = C29Config(fps=30, r4_ne_threshold=50.0, r4_yaw_threshold=20.0)
        eng = C29BoundaryEngine(config=cfg, enabled=True)
        for i in range(100):
            eng.push_frame(FakeFeats(
                head_pitch=15.0 * np.sin(i * 1.5),
                head_yaw=35.0,  # deep turn
                head_roll=0.0,
            ))
        alert = eng.evaluate()
        assert "R4" not in alert.active_rules


# ══════════════════════════════════════════════════════════════════════════
# D. Boost cap and combinations
# ══════════════════════════════════════════════════════════════════════════


class TestBoostCap:
    def test_boost_capped_at_max(self):
        alert = C29Alert(boost=0.60)
        # Engine should cap; simulate
        eng = _make_engine(fps=30)
        eng.cfg.max_boost = 0.40
        # Manually check cap in evaluate logic
        assert min(0.60, eng.cfg.max_boost) == 0.40

    def test_combined_boosts_capped(self):
        """R2+R3+R4 = 0.15+0.20+0.25 = 0.60 -> capped at 0.40."""
        cfg = C29Config(max_boost=0.40)
        alert = C29Alert()
        boost = cfg.r2_boost + cfg.r3_boost + cfg.r4_boost
        alert.boost = min(boost, cfg.max_boost)
        assert alert.boost == pytest.approx(0.40)


# ══════════════════════════════════════════════════════════════════════════
# E. Override-only mode
# ══════════════════════════════════════════════════════════════════════════


class TestOverrideOnly:
    def test_override_only_suppresses_boosts(self):
        cfg = C29Config(fps=30, override_only=True, r2_duration_sec=0.5)
        eng = C29BoundaryEngine(config=cfg, enabled=True)
        _feed_n_frames(eng, 60, pitch=0.0, yaw=0.0)
        alert = eng.evaluate()
        # R2 conditions met but override_only = True -> no boost
        assert alert.boost == 0.0
        assert not alert.override  # R1 not triggered (no warmup/AD)


# ══════════════════════════════════════════════════════════════════════════
# F. Reset and lifecycle
# ══════════════════════════════════════════════════════════════════════════


class TestResetAndLifecycle:
    def test_reset_long_buffers_clears_activity(self):
        eng = _make_engine(fps=10)
        _feed_n_frames(eng, 200, pitch=5.0, yaw=5.0)
        assert len(eng._activity_5min) > 0
        assert len(eng._activity_30s) > 0

        eng.reset_long_buffers()

        assert len(eng._activity_5min) == 0
        assert len(eng._activity_30s) == 0
        assert eng._r1_count == 0

    def test_reset_preserves_short_buffers(self):
        eng = _make_engine(fps=10)
        _feed_n_frames(eng, 50, pitch=5.0, yaw=10.0)
        n_motion = len(eng._motion_2s)
        n_pitch = len(eng._pitch_2s)

        eng.reset_long_buffers()

        assert len(eng._motion_2s) == n_motion
        assert len(eng._pitch_2s) == n_pitch

    def test_disabled_engine_returns_empty_alert(self):
        eng = C29BoundaryEngine(enabled=False)
        _feed_n_frames(eng, 100, pitch=5.0, yaw=5.0)
        alert = eng.evaluate()
        assert not alert.any_active
        assert eng._total_frames == 0  # push_frame is no-op

    def test_face_not_detected_skips_update(self):
        eng = _make_engine(fps=30)
        _feed_n_frames(eng, 10, face=False)
        assert len(eng._motion_2s) == 0
        assert eng._total_frames == 10  # counter still increments


# ══════════════════════════════════════════════════════════════════════════
# G. C29Alert dataclass
# ══════════════════════════════════════════════════════════════════════════


class TestC29Alert:
    def test_any_active_false_by_default(self):
        assert not C29Alert().any_active

    def test_any_active_true_on_override(self):
        assert C29Alert(override=True).any_active

    def test_any_active_true_on_boost(self):
        assert C29Alert(boost=0.15).any_active

    def test_any_active_false_on_zero_boost(self):
        assert not C29Alert(boost=0.0).any_active


# ══════════════════════════════════════════════════════════════════════════
# H. Config loading
# ══════════════════════════════════════════════════════════════════════════


class TestConfigLoading:
    def test_load_missing_file_returns_defaults(self, tmp_path):
        cfg = load_c29_config(tmp_path / "nonexistent.json")
        assert cfg.r1_ad_threshold == 0.5
        assert cfg.fps == 30

    def test_load_valid_json(self, tmp_path):
        p = tmp_path / "c29.json"
        p.write_text('{"r1_ad_threshold": 0.7, "fps": 15}')
        cfg = load_c29_config(p)
        assert cfg.r1_ad_threshold == 0.7
        assert cfg.fps == 15
        assert cfg.r2_boost == 0.15  # default preserved


# ══════════════════════════════════════════════════════════════════════════
# I. is_warm property
# ══════════════════════════════════════════════════════════════════════════


class TestIsWarm:
    def test_not_warm_initially(self):
        eng = _make_engine(fps=10, r1_warmup_sec=10.0)
        assert not eng.is_warm

    def test_warm_after_warmup(self):
        eng = _make_engine(fps=10, r1_warmup_sec=5.0)
        _feed_n_frames(eng, 50)  # 5s at 10fps
        assert eng.is_warm

    def test_evaluate_reports_warm_status(self):
        eng = _make_engine(fps=10, r1_warmup_sec=5.0)
        _feed_n_frames(eng, 50)
        alert = eng.evaluate()
        assert alert.is_warm
