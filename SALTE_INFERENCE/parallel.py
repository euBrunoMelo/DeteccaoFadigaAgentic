"""
Parallelization para DeteccaoFadiga.

Componentes:
  P1. FrameGrabber       — captura de câmera em thread dedicada
  P2. PipelineWorker     — processamento em thread separada
  P3. PerformanceMonitor — métricas de FPS, latência e drops
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from queue import Queue, Empty, Full
from typing import Any, Callable, Deque, Dict, Optional, Tuple

import numpy as np


# ── P1: Captura Paralela ─────────────────────────────────────────────────


@dataclass
class FrameGrabberConfig:
    """Configuração do capturador paralelo."""
    queue_size: int = 2          # Fila pequena: sempre pegar o frame mais recente
    target_fps: int = 30
    drop_old_frames: bool = True # Se fila cheia, descarta o mais antigo


class FrameGrabber:
    """
    P1: Captura de câmera em thread dedicada.

    A thread de captura roda em loop contínuo e coloca frames numa Queue.
    O consumidor (pipeline) pega frames sem bloquear a câmera.

    Se a fila está cheia e drop_old_frames=True, descarta o frame mais
    antigo para garantir que o consumidor sempre processe dados recentes.

    Benefícios:
    - Câmera nunca para de capturar (evita timeout de driver)
    - Pipeline pode demorar mais que 1/FPS sem dropar frames
    - Frame mais recente sempre disponível (baixa latência end-to-end)
    """

    def __init__(
        self,
        capture: Any,  # cv2.VideoCapture ou PiCamera2Capture
        config: Optional[FrameGrabberConfig] = None,
    ) -> None:
        self.cfg = config or FrameGrabberConfig()
        self._capture = capture
        self._queue: Queue = Queue(maxsize=self.cfg.queue_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frames_grabbed: int = 0
        self._frames_dropped: int = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        """Inicia a thread de captura."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="FrameGrabber",
        )
        self._thread.start()

    def stop(self) -> None:
        """Para a thread de captura e libera recursos."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def get_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """
        Retorna o frame mais recente, ou None se não há frame disponível.
        Não bloqueia por mais que `timeout` segundos.
        """
        try:
            frame = self._queue.get(timeout=timeout)
            return frame
        except Empty:
            return None

    def _capture_loop(self) -> None:
        """Loop de captura na thread dedicada."""
        while self._running:
            ret, frame = self._capture.read()
            if not ret:
                time.sleep(0.01)
                continue

            with self._lock:
                self._frames_grabbed += 1

            if self._queue.full() and self.cfg.drop_old_frames:
                try:
                    self._queue.get_nowait()  # descarta o mais antigo
                    with self._lock:
                        self._frames_dropped += 1
                except Empty:
                    pass

            try:
                self._queue.put_nowait(frame)
            except Full:
                with self._lock:
                    self._frames_dropped += 1

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "frames_grabbed": self._frames_grabbed,
                "frames_dropped": self._frames_dropped,
                "queue_size": self._queue.qsize(),
            }


# ── P2: Pipeline Worker ──────────────────────────────────────────────────


@dataclass
class PipelineResult:
    """Resultado do processamento de um frame pelo worker."""
    frame: np.ndarray                  # frame original (para overlay)
    raw_text: str                      # EAR/MAR/Face para display
    status_text: str                   # label + prob + confidence
    color: Tuple[int, int, int]        # cor do overlay
    overlay2: str                      # PERCLOS + blinks + alert
    output: Optional[Any] = None       # FatigueOutput (se disponível)
    window_feats: Optional[Dict] = None
    is_warmup: bool = False
    warmup_progress: float = 0.0
    warmup_elapsed: float = 0.0
    znorm_text: str = ""               # "EAR_z:0.12 MAR_z:-0.34 Pitch_z:0.01"
    is_calibrated: bool = False
    c29_text: str = ""                  # "C29:R3 +0.20" or "C29:R1 OVERRIDE"


class PipelineWorker:
    """
    P2: Executa o pipeline de inferência em thread separada.

    Arquitetura:
      Thread Principal (Main):  FrameGrabber -> display/overlay + key input
      Thread Worker:            frame_queue -> Feature -> Calibrate -> Window -> Model -> result_queue

    A main thread alimenta frames via put_frame() e consome
    resultados via get_result(). O worker processa na velocidade
    que conseguir — se for mais lento que o FPS da câmera,
    frames intermediários são descartados pelo FrameGrabber.
    """

    def __init__(
        self,
        process_fn: Callable[[np.ndarray], PipelineResult],
        queue_size: int = 2,
    ) -> None:
        self._process_fn = process_fn
        self._input_queue: Queue = Queue(maxsize=queue_size)
        self._output_queue: Queue = Queue(maxsize=queue_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frames_processed: int = 0
        self._total_process_time_ms: float = 0.0
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="PipelineWorker",
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def put_frame(self, frame: np.ndarray) -> bool:
        """Envia frame para processamento. Retorna False se fila cheia."""
        if self._input_queue.full():
            try:
                self._input_queue.get_nowait()  # descarta antigo
            except Empty:
                pass
        try:
            self._input_queue.put_nowait(frame)
            return True
        except Full:
            return False

    def get_result(self, timeout: float = 0.001) -> Optional[PipelineResult]:
        """Retorna resultado processado ou None."""
        try:
            return self._output_queue.get(timeout=timeout)
        except Empty:
            return None

    def _worker_loop(self) -> None:
        while self._running:
            try:
                frame = self._input_queue.get(timeout=0.1)
            except Empty:
                continue

            t0 = time.monotonic()
            result = self._process_fn(frame)
            dt = (time.monotonic() - t0) * 1000

            with self._lock:
                self._frames_processed += 1
                self._total_process_time_ms += dt

            # Colocar resultado na output queue
            if self._output_queue.full():
                try:
                    self._output_queue.get_nowait()
                except Empty:
                    pass
            try:
                self._output_queue.put_nowait(result)
            except Full:
                pass

    @property
    def avg_process_time_ms(self) -> float:
        with self._lock:
            if self._frames_processed == 0:
                return 0.0
            return self._total_process_time_ms / self._frames_processed

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            avg_ms = (
                self._total_process_time_ms / self._frames_processed
                if self._frames_processed > 0 else 0.0
            )
            return {
                "frames_processed": self._frames_processed,
                "avg_process_ms": round(avg_ms, 1),
                "input_queue": self._input_queue.qsize(),
                "output_queue": self._output_queue.qsize(),
            }


# ── P3: Performance Monitor ──────────────────────────────────────────────


class PerformanceMonitor:
    """
    P3: Métricas de performance do pipeline paralelo.

    Rastreia:
    - FPS real de captura, processamento e display
    - Latência end-to-end (captura -> resultado disponível)
    - Taxa de frame drop
    - Utilização de fila (indicador de backpressure)

    Imprime relatório periódico no console.
    """

    def __init__(self, report_interval_sec: float = 30.0) -> None:
        self._report_interval = report_interval_sec
        self._last_report_time = time.monotonic()

        # FPS tracking
        self._capture_timestamps: Deque[float] = deque(maxlen=100)
        self._process_timestamps: Deque[float] = deque(maxlen=100)
        self._display_timestamps: Deque[float] = deque(maxlen=100)

        # Latência
        self._latencies_ms: Deque[float] = deque(maxlen=100)

    def on_capture(self) -> None:
        self._capture_timestamps.append(time.monotonic())

    def on_process_complete(self, latency_ms: float) -> None:
        self._process_timestamps.append(time.monotonic())
        self._latencies_ms.append(latency_ms)

    def on_display(self) -> None:
        self._display_timestamps.append(time.monotonic())

    def _compute_fps(self, timestamps: Deque[float]) -> float:
        if len(timestamps) < 2:
            return 0.0
        dt = timestamps[-1] - timestamps[0]
        if dt < 1e-6:
            return 0.0
        return (len(timestamps) - 1) / dt

    def maybe_report(
        self,
        grabber_stats: Dict,
        worker_stats: Dict,
    ) -> Optional[str]:
        """Gera relatório se o intervalo foi atingido."""
        now = time.monotonic()
        if now - self._last_report_time < self._report_interval:
            return None

        self._last_report_time = now

        cap_fps = self._compute_fps(self._capture_timestamps)
        proc_fps = self._compute_fps(self._process_timestamps)
        disp_fps = self._compute_fps(self._display_timestamps)

        avg_lat = (
            float(np.mean(list(self._latencies_ms)))
            if self._latencies_ms else 0.0
        )
        p95_lat = (
            float(np.percentile(list(self._latencies_ms), 95))
            if len(self._latencies_ms) >= 5 else 0.0
        )

        total_grabbed = grabber_stats.get("frames_grabbed", 0)
        total_dropped = grabber_stats.get("frames_dropped", 0)
        drop_rate = (
            total_dropped / max(total_grabbed, 1) * 100
        )

        report = (
            f"[perf] FPS: capture={cap_fps:.1f} "
            f"process={proc_fps:.1f} display={disp_fps:.1f} | "
            f"Latency: avg={avg_lat:.0f}ms p95={p95_lat:.0f}ms | "
            f"Drops: {total_dropped}/{total_grabbed} ({drop_rate:.1f}%) | "
            f"Worker: {worker_stats.get('avg_process_ms', 0):.0f}ms/frame"
        )
        print(report)
        return report
