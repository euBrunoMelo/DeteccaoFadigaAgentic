"""Testes unitários para o módulo parallel (P1-P3)."""

import threading
import time

import numpy as np
import pytest

from SALTE_INFERENCE.parallel import (
    FrameGrabber,
    FrameGrabberConfig,
    PerformanceMonitor,
    PipelineResult,
    PipelineWorker,
)


# ── Helpers ───────────────────────────────────────────────────────────────


class FakeCapture:
    """Simulador de cv2.VideoCapture para testes."""

    def __init__(self, num_frames: int = 100, delay: float = 0.001):
        self._num_frames = num_frames
        self._delay = delay
        self._count = 0
        self._opened = True

    def isOpened(self):
        return self._opened

    def read(self):
        if self._count >= self._num_frames:
            return False, np.empty(0)
        self._count += 1
        time.sleep(self._delay)
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    def release(self):
        self._opened = False


class SlowCapture(FakeCapture):
    """Captura lenta para testar drops."""

    def __init__(self):
        super().__init__(num_frames=1000, delay=0.01)


# ── P1: FrameGrabber ────────────────────────────────────────────────────


class TestFrameGrabberP1:

    def test_start_and_stop(self):
        cap = FakeCapture(num_frames=50)
        g = FrameGrabber(cap, FrameGrabberConfig(queue_size=2))
        g.start()
        assert g.is_running
        time.sleep(0.1)
        g.stop()
        assert not g.is_running

    def test_get_frame(self):
        cap = FakeCapture(num_frames=50)
        g = FrameGrabber(cap, FrameGrabberConfig(queue_size=2))
        g.start()
        time.sleep(0.05)

        frame = g.get_frame(timeout=1.0)
        assert frame is not None
        assert frame.shape == (480, 640, 3)

        g.stop()

    def test_stats_tracking(self):
        cap = FakeCapture(num_frames=50, delay=0.001)
        g = FrameGrabber(cap, FrameGrabberConfig(queue_size=2))
        g.start()
        time.sleep(0.2)
        g.stop()

        stats = g.stats
        assert stats["frames_grabbed"] > 0

    def test_drops_when_slow_consumer(self):
        cap = FakeCapture(num_frames=100, delay=0.001)
        g = FrameGrabber(
            cap, FrameGrabberConfig(queue_size=1, drop_old_frames=True),
        )
        g.start()
        # Don't consume — let the queue fill up
        time.sleep(0.2)
        g.stop()

        stats = g.stats
        assert stats["frames_dropped"] > 0

    def test_none_when_no_frames(self):
        cap = FakeCapture(num_frames=0)
        g = FrameGrabber(cap, FrameGrabberConfig(queue_size=2))
        g.start()
        time.sleep(0.05)

        frame = g.get_frame(timeout=0.05)
        assert frame is None
        g.stop()

    def test_double_start_is_safe(self):
        cap = FakeCapture(num_frames=10)
        g = FrameGrabber(cap, FrameGrabberConfig(queue_size=2))
        g.start()
        g.start()  # should not create second thread
        assert g.is_running
        g.stop()

    def test_double_stop_is_safe(self):
        cap = FakeCapture(num_frames=10)
        g = FrameGrabber(cap, FrameGrabberConfig(queue_size=2))
        g.start()
        g.stop()
        g.stop()  # should not raise


# ── P2: PipelineWorker ───────────────────────────────────────────────────


class TestPipelineWorkerP2:

    def _make_process_fn(self, delay_ms: float = 1.0):
        def process_fn(frame: np.ndarray) -> PipelineResult:
            time.sleep(delay_ms / 1000)
            return PipelineResult(
                frame=frame,
                raw_text="EAR:0.3",
                status_text="Safe (0.20)",
                color=(0, 255, 0),
                overlay2="",
            )
        return process_fn

    def test_start_and_stop(self):
        w = PipelineWorker(self._make_process_fn())
        w.start()
        assert w.is_running
        w.stop()
        assert not w.is_running

    def test_process_frame(self):
        w = PipelineWorker(self._make_process_fn(delay_ms=1))
        w.start()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        w.put_frame(frame)
        time.sleep(0.1)

        result = w.get_result(timeout=1.0)
        assert result is not None
        assert result.status_text == "Safe (0.20)"
        w.stop()

    def test_stats_tracking(self):
        w = PipelineWorker(self._make_process_fn(delay_ms=0))
        w.start()
        time.sleep(0.1)  # let worker thread start

        w.put_frame(np.zeros((10, 10, 3), dtype=np.uint8))
        time.sleep(0.5)  # generous wait

        stats = w.stats
        w.stop()

        assert stats["frames_processed"] >= 1
        assert stats["avg_process_ms"] >= 0

    def test_slow_worker_drops_old_input(self):
        def slow_fn(frame):
            time.sleep(0.1)
            return PipelineResult(
                frame=frame, raw_text="", status_text="",
                color=(0, 0, 0), overlay2="",
            )

        w = PipelineWorker(slow_fn, queue_size=1)
        w.start()

        # Push many frames quickly
        for _ in range(10):
            w.put_frame(np.zeros((10, 10, 3), dtype=np.uint8))
            time.sleep(0.01)

        time.sleep(0.3)
        w.stop()
        # Should have processed some but not all
        assert w.stats["frames_processed"] < 10


# ── P3: PerformanceMonitor ───────────────────────────────────────────────


class TestPerformanceMonitorP3:

    def test_no_report_before_interval(self):
        pm = PerformanceMonitor(report_interval_sec=60.0)
        report = pm.maybe_report(
            {"frames_grabbed": 100, "frames_dropped": 5},
            {"avg_process_ms": 20, "frames_processed": 50},
        )
        assert report is None

    def test_report_after_interval(self):
        pm = PerformanceMonitor(report_interval_sec=0.0)
        pm.on_capture()
        pm.on_process_complete(latency_ms=20.0)
        pm.on_display()

        report = pm.maybe_report(
            {"frames_grabbed": 100, "frames_dropped": 5},
            {"avg_process_ms": 20, "frames_processed": 50},
        )
        assert report is not None
        assert "[perf]" in report
        assert "FPS:" in report
        assert "Drops:" in report

    def test_fps_computation(self):
        pm = PerformanceMonitor(report_interval_sec=0.0)
        # Simulate captures at ~100 fps
        for _ in range(10):
            pm.on_capture()
            time.sleep(0.01)

        report = pm.maybe_report(
            {"frames_grabbed": 10, "frames_dropped": 0},
            {"avg_process_ms": 10},
        )
        assert report is not None

    def test_latency_tracking(self):
        pm = PerformanceMonitor(report_interval_sec=0.0)
        for i in range(10):
            pm.on_process_complete(latency_ms=20.0 + i)

        report = pm.maybe_report(
            {"frames_grabbed": 10, "frames_dropped": 0},
            {"avg_process_ms": 25},
        )
        assert report is not None
        assert "Latency:" in report


# ── P2 Integration: PipelineWorker with process_fn ──────────────────────


class TestPipelineWorkerIntegrationP2:
    """Testes do PipelineWorker com process_fn simulando o pipeline real."""

    def test_worker_returns_warmup_result(self):
        """process_fn que simula warmup retorna PipelineResult(is_warmup=True)."""
        def fake_process(frame):
            return PipelineResult(
                frame=frame, raw_text="EAR:0.3", status_text="Calibrating...",
                color=(0, 200, 255), overlay2="",
                is_warmup=True, warmup_progress=0.5, warmup_elapsed=10.0,
            )
        worker = PipelineWorker(fake_process)
        worker.start()
        worker.put_frame(np.zeros((480, 640, 3), dtype=np.uint8))
        time.sleep(0.1)
        result = worker.get_result(timeout=0.5)
        worker.stop()
        assert result is not None
        assert result.is_warmup is True
        assert result.warmup_progress == 0.5

    def test_worker_returns_inference_result(self):
        """process_fn que simula inferência retorna PipelineResult com output."""
        def fake_process(frame):
            return PipelineResult(
                frame=frame, raw_text="EAR:0.3",
                status_text="Safe (0.20) [high]",
                color=(0, 255, 0), overlay2="Alert:SAFE",
                is_warmup=False,
                znorm_text="EAR_z:0.12 MAR_z:-0.34",
                is_calibrated=True,
            )
        worker = PipelineWorker(fake_process)
        worker.start()
        worker.put_frame(np.zeros((10, 10, 3), dtype=np.uint8))
        time.sleep(0.1)
        result = worker.get_result(timeout=0.5)
        worker.stop()
        assert result is not None
        assert result.is_warmup is False
        assert result.is_calibrated is True
        assert "EAR_z" in result.znorm_text

    def test_force_calibrate_event(self):
        """threading.Event propaga corretamente entre threads."""
        import threading
        event = threading.Event()
        assert not event.is_set()
        event.set()
        assert event.is_set()
        event.clear()
        assert not event.is_set()

    def test_worker_stats_reflect_real_processing(self):
        """worker.stats tem avg_process_ms > 0 após processar frames."""
        def slow_process(frame):
            time.sleep(0.01)
            return PipelineResult(
                frame=frame, raw_text="", status_text="",
                color=(0, 0, 0), overlay2="",
            )
        worker = PipelineWorker(slow_process)
        worker.start()
        for _ in range(3):
            worker.put_frame(np.zeros((10, 10, 3), dtype=np.uint8))
            time.sleep(0.05)
        time.sleep(0.2)
        stats = worker.stats
        worker.stop()
        assert stats["frames_processed"] >= 1
        assert stats["avg_process_ms"] > 0

    def test_pipeline_result_new_fields(self):
        """PipelineResult tem os novos campos znorm_text e is_calibrated."""
        result = PipelineResult(
            frame=np.zeros((10, 10, 3), dtype=np.uint8),
            raw_text="test", status_text="test",
            color=(0, 0, 0), overlay2="",
            znorm_text="EAR_z:0.5",
            is_calibrated=True,
        )
        assert result.znorm_text == "EAR_z:0.5"
        assert result.is_calibrated is True

    def test_pipeline_result_default_fields(self):
        """PipelineResult defaults: znorm_text='', is_calibrated=False."""
        result = PipelineResult(
            frame=np.zeros((10, 10, 3), dtype=np.uint8),
            raw_text="test", status_text="test",
            color=(0, 0, 0), overlay2="",
        )
        assert result.znorm_text == ""
        assert result.is_calibrated is False
