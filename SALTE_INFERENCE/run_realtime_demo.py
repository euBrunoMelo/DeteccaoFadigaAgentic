"""
Loop principal de inferência em tempo real com calibração per-subject.

V2: ONNXFaceMeshBackend, best_model.onnx (19 features), inference_config.json.
V3: FIX-RT-3 — HeadPoseNeutralizer (C33): zera 4 features de head pose
    para tornar o modelo agnóstico a pose em produção. Toggleável via
    --no-neutralize-pose para testes A/B com dados de lab.

Pipeline: picamera2/cv2 -> ONNXFaceMesh -> features -> scale_features (JSON)
          -> HeadPoseNeutralizer -> MLP V3 -> Safe/Danger.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Union

import cv2
import numpy as np

try:
    from picamera2 import Picamera2

    HAS_PICAMERA2 = True
except ImportError:
    Picamera2 = None
    HAS_PICAMERA2 = False

from .feature_extractor_rt import (
    DummyBackend,
    ONNXFaceMeshBackend,
    RealTimeFeatureExtractor,
)
from .model_loader import (
    load_best_model,
    predict_fatigue,
    scale_features,
)
from .subject_calibrator_rt import (
    CalibrationConfig,
    CalibratedFrame,
    RTSubjectCalibrator,
)
from .window_factory_rt import OnlineWindowFactory, RTWindowConfig
from . import guardrails
from .guardrails import (
    AlertLevel,
    BehaviorGuardRails,
    validate_calibration,
)
from .reflection import (
    AutoRecalibrationManager,
    DriftReflector,
    DriftStatus,
    PredictionReflector,
)
from .memory import FeatureLogger, OperatorStore, SessionMemory
from .parallel import (
    FrameGrabber,
    FrameGrabberConfig,
    PerformanceMonitor,
    PipelineResult,
    PipelineWorker,
)
from .agents import (
    OcularAgent,
    BlinkAgent,
    PosturalAgent,
    SupervisorAgent,
    SupervisorConfig,
)
from .c29_boundary_engine import (
    C29BoundaryEngine,
    C29Config,
    load_c29_config,
)


# ── FIX-RT-3: HeadPoseNeutralizer (C33) ─────────────────────────────────────


class HeadPoseNeutralizer:
    """
    FIX-RT-3: Neutraliza features de head pose para deploy em produção.

    Zera as 4 features de pose no vetor escalonado (z-score=0 = centro da
    distribuição de treino), tornando a classificação 100% dependente das
    15 features oculares/blink.

    ORDEM CRÍTICA: aplicar DEPOIS do SelectiveScaler para que o valor
    final seja exatamente 0.0.  Se aplicado ANTES, o scaler computa
    (0 - mean) / std ≠ 0, quebrando a neutralização.

    Toggleável via ``enabled`` para testes A/B (--no-neutralize-pose).

    Features neutralizadas (índices 4-7 em FEATURE_NAMES_21):
        pitch_mean, pitch_std, yaw_std, roll_std

    Ref: SALTE_COMPLETE_HISTORY §Bug #2, FeatureFactoryV4.md §4.2.
    Constraint: C33 — Head Pose Neutralization.
    """

    POSE_FEATURE_NAMES = {"pitch_mean", "pitch_std", "yaw_std", "roll_std"}

    def __init__(self, feature_names: list[str], enabled: bool = True) -> None:
        self.enabled = enabled
        self._pose_indices = [
            i for i, name in enumerate(feature_names)
            if name in self.POSE_FEATURE_NAMES
        ]
        if enabled and self._pose_indices:
            print(
                f"[neutralizer] Zeroing {len(self._pose_indices)} pose "
                f"features at indices {self._pose_indices}"
            )
        elif enabled and not self._pose_indices:
            print(
                "[neutralizer] WARNING: nenhuma feature de pose encontrada "
                "em feature_names — neutralizer é no-op"
            )

    def neutralize(self, vec: np.ndarray) -> np.ndarray:
        """Zera features de pose in-place (cópia). Retorna vetor modificado."""
        if not self.enabled:
            return vec
        v = vec.copy()
        for idx in self._pose_indices:
            v[idx] = 0.0
        return v


# ── PiCamera2 capture wrapper ───────────────────────────────────────────────


class PiCamera2Capture:
    """Wrapper picamera2 com interface compatível com cv2.VideoCapture."""

    def __init__(
        self,
        resolution: tuple[int, int] = (640, 480),
        fps: int = 30,
    ) -> None:
        if not HAS_PICAMERA2:
            raise RuntimeError(
                "picamera2 não está instalado. "
                "Instale com `sudo apt install python3-picamera2` "
                "ou use --camera-index para webcam USB."
            )
        self.cam = Picamera2()
        config = self.cam.create_video_configuration(
            main={"size": resolution, "format": "RGB888"},
            controls={"FrameRate": fps},
        )
        self.cam.configure(config)
        self.cam.start()
        self._opened = True

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> tuple[bool, np.ndarray]:
        if not self._opened:
            return False, np.empty(0)
        frame_rgb = self.cam.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        return True, frame_bgr

    def release(self) -> None:
        if self._opened:
            self.cam.stop()
            self._opened = False


# ── P2: process_fn factory (closure) ──────────────────────────────────────────


def _build_process_fn(
    *,
    extractor: RealTimeFeatureExtractor,
    calibrator: RTSubjectCalibrator,
    window_factory: OnlineWindowFactory,
    model,
    config,
    neutralizer: HeadPoseNeutralizer,
    threshold: float,
    feature_names: list,
    behavior_guard: BehaviorGuardRails,
    drift_reflector: DriftReflector,
    pred_reflector: PredictionReflector,
    recal_manager: AutoRecalibrationManager,
    supervisor,  # Optional[SupervisorAgent]
    c29_engine: C29BoundaryEngine,
    session: SessionMemory,
    feat_logger: FeatureLogger,
    op_store: OperatorStore,
    operator_id: str,
    training_stats: dict,
    fps: int,
    warmup_sec: float,
    min_warmup_sec: float,
    debug: bool,
    force_calibrate_event: threading.Event,
) -> Callable[[np.ndarray], PipelineResult]:
    """
    Constrói a process_fn para o PipelineWorker via closure.

    Todos os objetos stateful (extractor, calibrator, window_factory,
    behavior_guard, drift_reflector, etc.) ficam capturados na closure.
    Como o PipelineWorker chama process_fn de uma ÚNICA thread worker,
    o acesso serial a esses objetos é garantido — sem race conditions.

    A main thread NÃO acessa nenhum desses objetos diretamente.
    A comunicação é exclusivamente via PipelineResult (output queue).
    """

    def process_fn(frame: np.ndarray) -> PipelineResult:
        nonlocal calibrator, window_factory, drift_reflector, pred_reflector

        session.total_frames += 1
        feats = extractor.process_frame(frame)
        c29_engine.push_frame(feats)

        raw_text = (
            f"EAR:{feats.ear_avg:.3f} "
            f"MAR:{feats.mar:.3f} "
            f"Face:{int(feats.face_detected)}"
        )

        # ── Checar se main thread pediu force calibrate ──
        if force_calibrate_event.is_set():
            force_calibrate_event.clear()
            if not calibrator.is_calibrated:
                elapsed = len(calibrator._warmup_buffer) / fps
                if elapsed >= min_warmup_sec:
                    calibrator.force_calibrate()
                    print(f"[calibration] Forced at {elapsed:.0f}s")

        # ── Warm-up phase ──
        if not calibrator.is_calibrated:
            calibrated = calibrator.push(feats)
            progress = calibrator.warmup_progress

            if calibrator.is_calibrated:
                b = calibrator.baseline
                verdict = validate_calibration(b)

                if verdict.recommendation == "retry":
                    print(f"[guardrail] Calibração REJEITADA: {verdict.issues}")
                    print("[guardrail] Reiniciando warm-up...")
                    calibrator = RTSubjectCalibrator(
                        CalibrationConfig(fps=fps, search_sec=warmup_sec)
                    )
                    return PipelineResult(
                        frame=frame, raw_text=raw_text,
                        status_text="Recalibrating...",
                        color=(0, 200, 255), overlay2="",
                        is_warmup=True, warmup_progress=0.0,
                        warmup_elapsed=0.0,
                    )

                if verdict.recommendation == "use_with_caution":
                    print("[guardrail] Calibração ACEITA com ressalvas:")
                    for issue in verdict.issues:
                        print(f"  [guardrail] {issue}")

                print("[calibration] Baseline computed!")
                print(
                    f"[calibration]   EAR: mean={b.ear_mean:.4f}, "
                    f"std={b.ear_std:.4f}"
                )

                behavior_guard.on_calibration_complete()
                session.on_calibration(b.ear_mean, verdict.recommendation)
                op_store.upsert_profile_from_calibration(operator_id, b)
                window_factory.set_perclos_baseline(b.ear_mean)

                if calibrated is None:
                    calibrated = calibrator.calibrate(feats)
            else:
                elapsed = len(calibrator._warmup_buffer) / fps
                return PipelineResult(
                    frame=frame, raw_text=raw_text,
                    status_text="Calibrating...",
                    color=(0, 200, 255), overlay2="",
                    is_warmup=True,
                    warmup_progress=progress,
                    warmup_elapsed=elapsed,
                )
        else:
            calibrated = calibrator.calibrate(feats)

        # ── Z-norm text for display ──
        znorm_text = ""
        if calibrated is not None:
            znorm_text = (
                f"EAR_z:{calibrated.ear_avg_znorm:.2f} "
                f"MAR_z:{calibrated.mar_znorm:.2f} "
                f"Pitch_z:{calibrated.head_pitch_znorm:.2f}"
            )

        # ── Window aggregation ──
        window_feats = window_factory.push(calibrated)

        status_text = "Calibrated - waiting for window..."
        color = (255, 255, 255)
        overlay2 = ""
        output = None

        if window_feats is not None:
            vec_raw = np.array(
                [window_feats[name] for name in feature_names],
                dtype=np.float32,
            )
            vec = scale_features(vec_raw, config)
            vec = neutralizer.neutralize(vec)

            prob_danger, label = predict_fatigue(
                vec, model, config, threshold_override=threshold
            )

            # Guardrails (G1-G2: validação estrutural)
            output = guardrails.validate_and_wrap(
                prob_danger=prob_danger, label=label,
                window_feats=window_feats,
                feature_names=feature_names,
                config=config,
                timestamp_ms=calibrated.timestamp_ms,
                threshold=threshold,
            )

            # C29 Boundary Rules (Layer 5) — BEFORE G3 so behavior_guard
            # sees the final label/prob (including C29 boost/override)
            c29_alert_result = c29_engine.evaluate()
            if c29_alert_result.any_active:
                _mlp_label = output.label
                _mlp_prob = output.prob_danger
                if c29_alert_result.override:
                    output.label = "Danger"
                    output.prob_danger = max(output.prob_danger, 0.80)
                    if output.alert_level.value < AlertLevel.DANGER.value:
                        output.alert_level = AlertLevel.DANGER
                elif c29_alert_result.boost > 0:
                    output.prob_danger = min(
                        output.prob_danger + c29_alert_result.boost, 0.99,
                    )
                    if output.prob_danger >= threshold:
                        output.label = "Danger"
                        if output.alert_level.value < AlertLevel.DANGER.value:
                            output.alert_level = AlertLevel.DANGER
                c29_engine.log_alert(
                    c29_alert_result, _mlp_label, _mlp_prob,
                    output.label, output.prob_danger,
                )
                session.on_c29_alert(
                    c29_alert_result.override, c29_alert_result.boost,
                )

            # G3: behavior_guard agora vê label FINAL (pós-C29)
            output = behavior_guard.process(output)

            # Reflection (R1-R3)
            drift_report = drift_reflector.push_window(window_feats)
            reflection = pred_reflector.push(
                output.prob_danger, output.label,
                c29_active=c29_alert_result.any_active,
            )

            if reflection.pattern != "stable":
                print(
                    f"[reflection] Padrão: {reflection.pattern} | "
                    f"{reflection.suggestion}"
                )
            if reflection.confidence_modifier < 1.0:
                output.confidence = "low"

            if drift_report is not None:
                if drift_report.status != DriftStatus.STABLE:
                    print(
                        f"[reflection] Drift {drift_report.status.value}: "
                        f"{drift_report.drifted_features}"
                    )

                recal_decision = recal_manager.evaluate(
                    drift_report=drift_report,
                    pred_reflection=reflection,
                )
                if recal_decision.should_recalibrate:
                    print(
                        f"[reflection] AUTO-RECALIBRAÇÃO: "
                        f"{recal_decision.reason} "
                        f"(urgência: {recal_decision.urgency})"
                    )
                    session.on_auto_recalibration()
                    if recal_decision.urgency == "immediate":
                        calibrator = RTSubjectCalibrator(
                            CalibrationConfig(fps=fps, search_sec=60.0)
                        )
                        window_factory = OnlineWindowFactory(
                            RTWindowConfig(fps=fps)
                        )
                        drift_reflector = DriftReflector(training_stats)
                        pred_reflector = PredictionReflector()
                        behavior_guard.on_calibration_complete()
                        c29_engine.reset_long_buffers()

            # Alerts
            alert_triggered = behavior_guard.should_sound_alert(output)
            if alert_triggered:
                print("[alert] SOUND ALERT triggered")
            session.on_alert(triggered=alert_triggered)

            # Memory
            session.on_window(
                label=output.label,
                prob_danger=output.prob_danger,
                alert_level_name=output.alert_level.name,
                perclos=output.perclos,
                ear_mean=window_feats.get("ear_mean", 0.0),
                microsleep_count=output.microsleep_count,
                microsleep_total_ms=window_feats.get(
                    "microsleep_total_ms", 0.0
                ),
            )
            feat_logger.log(
                timestamp_ms=output.timestamp_ms,
                label=output.label,
                prob_danger=output.prob_danger,
                alert_level=output.alert_level.name,
                confidence=output.confidence,
                window_feats=window_feats,
            )

            # Multi-Agent
            supervisor_decision = None
            if supervisor is not None:
                supervisor_decision = supervisor.decide(window_feats)
                if debug:
                    print(f"[agents] MLP: {output.label} ({output.prob_danger:.3f})")
                    print(f"[agents] Supervisor: {supervisor_decision.label}")

            # Build overlay strings
            status_text = (
                f"{output.label} ({output.prob_danger:.2f}) "
                f"[{output.confidence}]"
            )
            color = (0, 0, 255) if output.label == "Danger" else (0, 255, 0)
            if output.alert_level == AlertLevel.CRITICAL:
                color = (0, 0, 200)
            elif output.alert_level == AlertLevel.WATCH:
                color = (0, 165, 255)

            overlay2 = (
                f"PERCLOS:{output.perclos:.2f} "
                f"BlinkCount:{output.blink_count:.1f} "
                f"Microsleeps:{output.microsleep_count:.1f} "
                f"Alert:{output.alert_level.name}"
            )
            if (supervisor_decision is not None
                    and supervisor_decision.fatigue_type != "none"):
                overlay2 += f" Type:{supervisor_decision.fatigue_type}"

            # C29 overlay text
            c29_text = ""
            if c29_alert_result.any_active:
                _rules = ",".join(c29_alert_result.active_rules)
                if c29_alert_result.override:
                    c29_text = f"C29:{_rules} OVERRIDE"
                else:
                    c29_text = f"C29:{_rules} +{c29_alert_result.boost:.2f}"
                overlay2 += f" {c29_text}"

            print(
                f"[window] label={output.label} "
                f"prob={output.prob_danger:.3f} "
                f"alert={output.alert_level.name}"
            )

        return PipelineResult(
            frame=frame,
            raw_text=raw_text,
            status_text=status_text,
            color=color,
            overlay2=overlay2,
            output=output,
            window_feats=window_feats,
            znorm_text=znorm_text,
            is_calibrated=calibrator.is_calibrated,
            c29_text=c29_text if window_feats is not None else "",
        )

    return process_fn


# ── Main realtime loop ───────────────────────────────────────────────────────


def run_realtime(
    checkpoint_path: Union[Path, str],
    config_path: Union[Path, str, None],
    *,
    detector_path: Union[Path, str, None] = None,
    mesh_path: Union[Path, str, None] = None,
    camera_index: int = 0,
    use_picamera: bool = False,
    threshold_override: float | None = None,
    warmup_sec: float = 120.0,
    min_warmup_sec: float = 30.0,
    fps: int = 30,
    headless: bool = False,
    debug: bool = False,
    neutralize_pose: bool = True,
    log_features: bool = False,
    operator_id: str | None = None,
    parallel: bool = False,
    no_agents: bool = False,
    disable_c29: bool = False,
    c29_override_only: bool = False,
    c29_log: bool = False,
) -> None:
    """
    Loop realtime com calibração per-subject.

    Fases:
    1. WARM-UP: coleta frames por warmup_sec segundos
    2. CALIBRATED: inferência com Safe/Danger overlay

    Args:
        neutralize_pose: Se True (default), zera as 4 features de head pose
            após o scaler, tornando a predição agnóstica a pose (C33).
            Desativar com --no-neutralize-pose para testes com dados de lab.
    """
    model_dir = Path(checkpoint_path).parent
    if config_path is None:
        config_path = model_dir / "inference_config.json"
    if detector_path is None:
        detector_path = model_dir / "blazeface_detector.onnx"
    if mesh_path is None:
        mesh_path = model_dir / "face_mesh_landmark.onnx"

    model, config = load_best_model(checkpoint_path, config_path=config_path)
    threshold = threshold_override if threshold_override is not None else config.threshold
    feature_names = config.feature_names

    # ── Guardrails + Reflection: carregar training_stats e inicializar ────
    with open(str(config_path), encoding="utf-8") as _f:
        _raw_config = json.load(_f)
    training_stats = _raw_config.get("training_stats", {})

    behavior_guard = BehaviorGuardRails()
    drift_reflector = DriftReflector(training_stats)
    pred_reflector = PredictionReflector()
    recal_manager = AutoRecalibrationManager()
    print("[init] Guardrails + Reflection: initialized")
    # ─────────────────────────────────────────────────────────────────────

    # ── Memory: SessionMemory + OperatorStore + FeatureLogger ─────────
    _op_id = operator_id or f"op-{camera_index}"
    session = SessionMemory(operator_id=_op_id)
    session_started_at = datetime.now().isoformat()

    op_store = OperatorStore(db_path=str(model_dir / "operator_memory.db"))

    # Check warm-start
    warm_baseline = op_store.get_warm_start_baseline(_op_id)
    if warm_baseline is not None:
        print(f"[memory] Warm-start disponível para {_op_id}")
        print(f"[memory]   EAR histórico: {warm_baseline['ear_mean']:.4f}")
        warmup_sec = 30.0

    feat_logger = FeatureLogger(
        output_dir=str(model_dir / "logs"),
        feature_names=feature_names,
        operator_id=_op_id,
        enabled=log_features,
    )
    print(f"[init] Memory: SessionMemory + OperatorStore initialized")
    # ─────────────────────────────────────────────────────────────────────

    # ── Multi-Agent: Supervisor + 3 especialistas ──────────────────────
    supervisor = None
    if not no_agents:
        ocular_agent = OcularAgent()
        blink_agent = BlinkAgent()
        postural_agent = PosturalAgent(pose_neutralized=neutralize_pose)
        supervisor = SupervisorAgent(
            ocular_agent, blink_agent, postural_agent,
            SupervisorConfig(),
        )
        print("[init] Multi-Agent: Supervisor + 3 specialists initialized")
    # ─────────────────────────────────────────────────────────────────────

    # ── C29 Boundary Engine (Layer 5) ────────────────────────────────────
    c29_cfg_path = model_dir / "c29_boundary_config.json"
    c29_cfg = load_c29_config(c29_cfg_path)
    c29_cfg.fps = fps
    if c29_override_only:
        c29_cfg.override_only = True
    c29_log_path = str(model_dir / "c29_alerts.jsonl") if c29_log else None
    c29_engine = C29BoundaryEngine(
        config=c29_cfg,
        enabled=not disable_c29,
        log_path=c29_log_path,
    )
    # ─────────────────────────────────────────────────────────────────────

    # ── FIX-RT-3: inicializar neutralizer ────────────────────────────────
    neutralizer = HeadPoseNeutralizer(
        feature_names=feature_names,
        enabled=neutralize_pose,
    )
    if not neutralize_pose:
        print("[init] HeadPoseNeutralizer: DISABLED (--no-neutralize-pose)")
    # ─────────────────────────────────────────────────────────────────────

    print(f"[init] Scaler: JSON (inference_config)")
    print(f"[init] Threshold: {threshold} ({len(feature_names)} features)")

    try:
        backend = ONNXFaceMeshBackend(
            str(detector_path),
            str(mesh_path),
            min_face_score=0.5,
        )
        print("[init] Backend: ONNXFaceMeshBackend (BlazeFace + FaceMesh ONNX)")
    except (RuntimeError, FileNotFoundError) as e:
        print(f"[init] Backend fallback: DummyBackend ({e})")
        backend = DummyBackend()

    extractor = RealTimeFeatureExtractor(backend)
    calibrator = RTSubjectCalibrator(
        CalibrationConfig(fps=fps, search_sec=warmup_sec)
    )
    window_factory = OnlineWindowFactory(RTWindowConfig(fps=fps))

    if use_picamera:
        cap = PiCamera2Capture(fps=fps)
        print("[init] Camera: picamera2 (AI Camera)")
    else:
        cap = cv2.VideoCapture(camera_index)
        print(f"[init] Camera: cv2.VideoCapture({camera_index})")

    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir a câmera")

    print(f"[init] Headless: {headless}")
    print(
        f"[init] Warm-up: {warmup_sec}s "
        f"(press 'c' to calibrate early, 'q' to quit)"
    )

    frame_interval = 1.0 / max(fps, 1)

    # ── Sinais inter-thread ──────────────────────────────────────────
    force_calibrate_event = threading.Event()
    # ─────────────────────────────────────────────────────────────────

    if parallel:
        # ── PARALLEL MODE: FrameGrabber + PipelineWorker + PerformanceMonitor ──
        grabber = FrameGrabber(
            cap, FrameGrabberConfig(target_fps=fps),
        )
        grabber.start()
        perf_monitor = PerformanceMonitor(report_interval_sec=30.0)

        process_fn = _build_process_fn(
            extractor=extractor,
            calibrator=calibrator,
            window_factory=window_factory,
            model=model,
            config=config,
            neutralizer=neutralizer,
            threshold=threshold,
            feature_names=feature_names,
            behavior_guard=behavior_guard,
            drift_reflector=drift_reflector,
            pred_reflector=pred_reflector,
            recal_manager=recal_manager,
            supervisor=supervisor,
            c29_engine=c29_engine,
            session=session,
            feat_logger=feat_logger,
            op_store=op_store,
            operator_id=_op_id,
            training_stats=training_stats,
            fps=fps,
            warmup_sec=warmup_sec,
            min_warmup_sec=min_warmup_sec,
            debug=debug,
            force_calibrate_event=force_calibrate_event,
        )

        worker = PipelineWorker(process_fn=process_fn, queue_size=2)
        worker.start()
        print("[init] Parallel: FrameGrabber + PipelineWorker + PerformanceMonitor started")

        try:
            last_result: Optional[PipelineResult] = None

            while True:
                # 1. Grab frame (from FrameGrabber thread)
                frame = grabber.get_frame(timeout=0.1)
                if frame is not None:
                    perf_monitor.on_capture()
                    worker.put_frame(frame)

                # 2. Get processed result (non-blocking)
                result = worker.get_result(timeout=0.005)
                if result is not None:
                    perf_monitor.on_process_complete(
                        worker.avg_process_time_ms
                    )
                    last_result = result

                # 3. Display (always show latest result)
                if not headless and last_result is not None:
                    if last_result.is_warmup:
                        _draw_warmup_overlay(
                            last_result.frame,
                            last_result.warmup_progress,
                            warmup_sec,
                            last_result.warmup_elapsed,
                            last_result.raw_text,
                        )
                    else:
                        _draw_inference_overlay_parallel(
                            last_result.frame,
                            last_result.status_text,
                            last_result.color,
                            last_result.raw_text,
                            last_result.overlay2,
                            last_result.znorm_text,
                            last_result.is_calibrated,
                        )
                    cv2.imshow("SALTE Realtime Demo", last_result.frame)
                    perf_monitor.on_display()

                # 4. Keyboard input (main thread only)
                if not headless:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("c"):
                        force_calibrate_event.set()
                    elif key == ord("q"):
                        break
                else:
                    time.sleep(0.005)

                # 5. Performance report (with REAL worker stats)
                perf_monitor.maybe_report(
                    grabber.stats,
                    worker.stats,
                )

        finally:
            worker.stop()
            grabber.stop()
            cap.release()
            if not headless:
                cv2.destroyAllWindows()

            # ── Memory: salvar sessão e fechar ────────────────────────
            summary = session.summary()
            print(f"[session] Resumo final: {summary}")
            op_store.save_session(_op_id, summary, session_started_at)
            op_store.close()
            feat_logger.close()
            c29_engine.close()
            print(f"[memory] Sessão salva para {_op_id}")
            # ──────────────────────────────────────────────────────────

    else:
        # ── SERIAL MODE (código original, backward-compatible) ─────────
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                session.total_frames += 1

                feats = extractor.process_frame(frame)
                c29_engine.push_frame(feats)

                raw_text = (
                    f"EAR:{feats.ear_avg:.3f} "
                    f"MAR:{feats.mar:.3f} "
                    f"Face:{int(feats.face_detected)}"
                )

                if not headless:
                    key = cv2.waitKey(1) & 0xFF
                else:
                    key = 0
                    time.sleep(frame_interval)

                if not calibrator.is_calibrated:
                    calibrated = calibrator.push(feats)
                    progress = calibrator.warmup_progress

                    if key == ord("c"):
                        elapsed = len(calibrator._warmup_buffer) / fps
                        if elapsed >= min_warmup_sec:
                            calibrator.force_calibrate()
                            print(f"[calibration] Forced at {elapsed:.0f}s")
                        else:
                            print(
                                f"[calibration] Need at least {min_warmup_sec}s "
                                f"(current: {elapsed:.0f}s)"
                            )

                    if calibrator.is_calibrated:
                        b = calibrator.baseline

                        # ── G4: Validar qualidade da calibração ───────────
                        verdict = validate_calibration(b)

                        if verdict.recommendation == "retry":
                            print(
                                f"[guardrail] Calibração REJEITADA: "
                                f"{verdict.issues}"
                            )
                            print("[guardrail] Reiniciando warm-up...")
                            calibrator = RTSubjectCalibrator(
                                CalibrationConfig(
                                    fps=fps, search_sec=warmup_sec
                                )
                            )
                            continue

                        if verdict.recommendation == "use_with_caution":
                            print(
                                "[guardrail] Calibração ACEITA com ressalvas:"
                            )
                            for issue in verdict.issues:
                                print(f"  [guardrail] {issue}")
                        # ──────────────────────────────────────────────────

                        print("[calibration] Baseline computed!")
                        print(
                            f"[calibration]   EAR:   mean={b.ear_mean:.4f}, "
                            f"std={b.ear_std:.4f}"
                        )
                        print(
                            f"[calibration]   EAR P90 (debug): {b.ear_p90_raw:.4f}"
                        )
                        pf = window_factory.cfg.perclos_factor
                        print(
                            f"[calibration]   PERCLOS baseline (=ear_mean): "
                            f"{b.ear_mean:.4f}"
                        )
                        print(
                            f"[calibration]   PERCLOS factor: {pf} "
                            f"(offline=0.80, RT=0.65)"
                        )
                        print(
                            f"[calibration]   PERCLOS threshold "
                            f"(mean*{pf}): {b.ear_mean * pf:.4f}"
                        )
                        print(
                            f"[calibration]   MAR:   mean={b.mar_mean:.4f}, "
                            f"std={b.mar_std:.4f}"
                        )
                        print(
                            f"[calibration]   Pitch: mean={b.pitch_mean:.2f}, "
                            f"std={b.pitch_std:.2f}"
                        )
                        print(
                            f"[calibration]   Yaw:   mean={b.yaw_mean:.2f}, "
                            f"std={b.yaw_std:.2f}"
                        )
                        print(
                            f"[calibration]   Roll:  mean={b.roll_mean:.2f}, "
                            f"std={b.roll_std:.2f}"
                        )
                        print(f"[calibration]   Valid: {b.is_valid}")
                        print(
                            f"[calibration]   Segment: frames "
                            f"{b.segment_start}-{b.segment_end}"
                        )

                        behavior_guard.on_calibration_complete()

                        # ── Memory: registrar calibração ──────────────────
                        session.on_calibration(
                            b.ear_mean, verdict.recommendation,
                        )
                        op_store.upsert_profile_from_calibration(
                            _op_id, b,
                        )
                        # ──────────────────────────────────────────────────

                        window_factory.set_perclos_baseline(b.ear_mean)
                    else:
                        if not headless:
                            _draw_warmup_overlay(
                                frame, progress, warmup_sec,
                                len(calibrator._warmup_buffer) / fps,
                                raw_text,
                            )
                            cv2.imshow("SALTE Realtime Demo", frame)

                        if key == ord("q"):
                            break
                        continue

                    if calibrated is None:
                        calibrated = calibrator.calibrate(feats)

                else:
                    calibrated = calibrator.calibrate(feats)

                window_feats = window_factory.push(calibrated)

                status_text = "Calibrated - waiting for window..."
                color = (255, 255, 255)
                overlay2 = ""

                if window_feats is not None:
                    vec_raw = np.array(
                        [window_feats[name] for name in feature_names],
                        dtype=np.float32,
                    )

                    if debug:
                        print("\n[debug] Raw feature vector:")
                        for i, name in enumerate(feature_names):
                            print(f"  {name:30s} = {vec_raw[i]:12.6f}")

                    vec = scale_features(vec_raw, config)

                    if debug:
                        print("[debug] After scale_features:")
                        for i, name in enumerate(feature_names):
                            print(f"  {name:30s} = {vec[i]:12.6f}")

                    vec = neutralizer.neutralize(vec)

                    if debug and neutralizer.enabled:
                        print("[debug] After HeadPoseNeutralizer:")
                        for i, name in enumerate(feature_names):
                            print(f"  {name:30s} = {vec[i]:12.6f}")

                    prob_danger, label = predict_fatigue(
                        vec, model, config, threshold_override=threshold
                    )

                    output = guardrails.validate_and_wrap(
                        prob_danger=prob_danger,
                        label=label,
                        window_feats=window_feats,
                        feature_names=feature_names,
                        config=config,
                        timestamp_ms=calibrated.timestamp_ms,
                        threshold=threshold,
                    )

                    # C29 Boundary Rules (Layer 5) — BEFORE G3
                    c29_alert_result = c29_engine.evaluate()
                    if c29_alert_result.any_active:
                        _mlp_label = output.label
                        _mlp_prob = output.prob_danger
                        if c29_alert_result.override:
                            output.label = "Danger"
                            output.prob_danger = max(output.prob_danger, 0.80)
                            if output.alert_level.value < AlertLevel.DANGER.value:
                                output.alert_level = AlertLevel.DANGER
                        elif c29_alert_result.boost > 0:
                            output.prob_danger = min(
                                output.prob_danger + c29_alert_result.boost,
                                0.99,
                            )
                            if output.prob_danger >= threshold:
                                output.label = "Danger"
                                if output.alert_level.value < AlertLevel.DANGER.value:
                                    output.alert_level = AlertLevel.DANGER
                        c29_engine.log_alert(
                            c29_alert_result, _mlp_label, _mlp_prob,
                            output.label, output.prob_danger,
                        )
                        session.on_c29_alert(
                            c29_alert_result.override,
                            c29_alert_result.boost,
                        )

                    # G3: behavior_guard agora vê label FINAL (pós-C29)
                    output = behavior_guard.process(output)

                    drift_report = drift_reflector.push_window(window_feats)
                    reflection = pred_reflector.push(
                        output.prob_danger, output.label,
                        c29_active=c29_alert_result.any_active,
                    )

                    if reflection.pattern != "stable":
                        print(
                            f"[reflection] Padrão: {reflection.pattern} | "
                            f"{reflection.suggestion}"
                        )

                    if reflection.confidence_modifier < 1.0:
                        output.confidence = "low"

                    if drift_report is not None:
                        if drift_report.status != DriftStatus.STABLE:
                            print(
                                f"[reflection] Drift {drift_report.status.value}: "
                                f"{drift_report.drifted_features}"
                            )
                            print(
                                f"[reflection] Z-scores: "
                                f"{drift_report.details}"
                            )
                            print(
                                f"[reflection] Recomendação: "
                                f"{drift_report.recommendation}"
                            )

                        recal_decision = recal_manager.evaluate(
                            drift_report=drift_report,
                            pred_reflection=reflection,
                        )

                        if recal_decision.should_recalibrate:
                            print(
                                f"[reflection] AUTO-RECALIBRAÇÃO: "
                                f"{recal_decision.reason} "
                                f"(urgência: {recal_decision.urgency})"
                            )

                            session.on_auto_recalibration()

                            if recal_decision.urgency == "immediate":
                                calibrator = RTSubjectCalibrator(
                                    CalibrationConfig(
                                        fps=fps, search_sec=60.0
                                    )
                                )
                                window_factory = OnlineWindowFactory(
                                    RTWindowConfig(fps=fps)
                                )
                                drift_reflector = DriftReflector(
                                    training_stats
                                )
                                pred_reflector = PredictionReflector()
                                behavior_guard.on_calibration_complete()
                                c29_engine.reset_long_buffers()

                    alert_triggered = behavior_guard.should_sound_alert(
                        output
                    )
                    if alert_triggered:
                        print("[alert] SOUND ALERT triggered")
                    session.on_alert(triggered=alert_triggered)

                    session.on_window(
                        label=output.label,
                        prob_danger=output.prob_danger,
                        alert_level_name=output.alert_level.name,
                        perclos=output.perclos,
                        ear_mean=window_feats.get("ear_mean", 0.0),
                        microsleep_count=output.microsleep_count,
                        microsleep_total_ms=window_feats.get(
                            "microsleep_total_ms", 0.0
                        ),
                    )
                    feat_logger.log(
                        timestamp_ms=output.timestamp_ms,
                        label=output.label,
                        prob_danger=output.prob_danger,
                        alert_level=output.alert_level.name,
                        confidence=output.confidence,
                        window_feats=window_feats,
                    )

                    supervisor_decision = None
                    if supervisor is not None:
                        supervisor_decision = supervisor.decide(window_feats)

                        if debug:
                            print(f"[agents] MLP: {output.label} ({output.prob_danger:.3f})")
                            print(f"[agents] Supervisor: {supervisor_decision.label} "
                                  f"({supervisor_decision.combined_score:.3f})")
                            print(f"[agents] Type: {supervisor_decision.fatigue_type}")
                            print(f"[agents] Dominant: {supervisor_decision.dominant_agent}")
                            print(f"[agents] Agreement: {supervisor_decision.agent_agreement:.2f}")
                            for op in supervisor_decision.opinions:
                                print(f"[agents]   {op.agent_name}: {op.signal.name} "
                                      f"(score={op.score:.3f}, conf={op.confidence:.2f})")
                                print(f"[agents]     → {op.reasoning}")

                    status_text = (
                        f"{output.label} ({output.prob_danger:.2f}) "
                        f"[{output.confidence}]"
                    )
                    color = (0, 0, 255) if output.label == "Danger" else (0, 255, 0)
                    if output.alert_level == AlertLevel.CRITICAL:
                        color = (0, 0, 200)
                    elif output.alert_level == AlertLevel.WATCH:
                        color = (0, 165, 255)

                    overlay2 = (
                        f"PERCLOS:{output.perclos:.2f} "
                        f"BlinkCount:{output.blink_count:.1f} "
                        f"Microsleeps:{output.microsleep_count:.1f} "
                        f"Alert:{output.alert_level.name}"
                    )
                    if (supervisor_decision is not None
                            and supervisor_decision.fatigue_type != "none"):
                        overlay2 += f" Type:{supervisor_decision.fatigue_type}"
                    if c29_alert_result.any_active:
                        _rules = ",".join(c29_alert_result.active_rules)
                        if c29_alert_result.override:
                            overlay2 += f" C29:{_rules} OVERRIDE"
                        else:
                            overlay2 += f" C29:{_rules} +{c29_alert_result.boost:.2f}"

                    print(
                        f"[window] label={output.label} "
                        f"prob={output.prob_danger:.3f} "
                        f"alert={output.alert_level.name} "
                        f"ear_mean_z={window_feats.get('ear_mean', 0.0):.3f} "
                        f"perclos={output.perclos:.3f} "
                        f"blinks={output.blink_count:.0f} "
                        f"microsleeps={output.microsleep_count:.0f} "
                        f"confidence={output.confidence}"
                    )

                if not headless:
                    _draw_inference_overlay(
                        frame, status_text, color, raw_text,
                        overlay2, calibrator, calibrated,
                    )
                    cv2.imshow("SALTE Realtime Demo", frame)

                if key == ord("q"):
                    break

        finally:
            cap.release()
            if not headless:
                cv2.destroyAllWindows()

            # ── Memory: salvar sessão e fechar ────────────────────────
            summary = session.summary()
            print(f"[session] Resumo final: {summary}")
            op_store.save_session(_op_id, summary, session_started_at)
            op_store.close()
            feat_logger.close()
            c29_engine.close()
            print(f"[memory] Sessão salva para {_op_id}")
            # ──────────────────────────────────────────────────────────


# ── Overlay drawing helpers ──────────────────────────────────────────────────


def _draw_warmup_overlay(
    frame: np.ndarray,
    progress: float,
    warmup_sec: float,
    elapsed_sec: float,
    raw_text: str,
) -> None:
    bar_w = int(frame.shape[1] * 0.6)
    bar_h = 30
    bar_x = (frame.shape[1] - bar_w) // 2
    bar_y = frame.shape[0] // 2

    cv2.rectangle(
        frame, (bar_x, bar_y),
        (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1,
    )
    fill_w = int(bar_w * progress)
    cv2.rectangle(
        frame, (bar_x, bar_y),
        (bar_x + fill_w, bar_y + bar_h), (0, 200, 255), -1,
    )
    cv2.putText(
        frame,
        f"CALIBRATING... {elapsed_sec:.0f}/{warmup_sec:.0f}s  "
        f"(press 'c' to skip)",
        (bar_x, bar_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, raw_text, (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
    )


def _draw_inference_overlay(
    frame: np.ndarray,
    status_text: str,
    color: tuple[int, int, int],
    raw_text: str,
    overlay2: str,
    calibrator: RTSubjectCalibrator,
    calibrated: Optional[CalibratedFrame],
) -> None:
    cv2.putText(
        frame, status_text, (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, raw_text, (16, 64),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
    )

    if calibrator.is_calibrated and calibrated is not None:
        znorm_text = (
            f"EAR_z:{calibrated.ear_avg_znorm:.2f} "
            f"MAR_z:{calibrated.mar_znorm:.2f} "
            f"Pitch_z:{calibrated.head_pitch_znorm:.2f}"
        )
        cv2.putText(
            frame, znorm_text, (16, 96),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 255), 1, cv2.LINE_AA,
        )

    if overlay2:
        cv2.putText(
            frame, overlay2, (16, 128),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA,
        )


def _draw_inference_overlay_parallel(
    frame: np.ndarray,
    status_text: str,
    color: tuple[int, int, int],
    raw_text: str,
    overlay2: str,
    znorm_text: str,
    is_calibrated: bool,
) -> None:
    """Overlay para modo paralelo — usa znorm_text do PipelineResult."""
    cv2.putText(
        frame, status_text, (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, raw_text, (16, 64),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
    )

    if is_calibrated and znorm_text:
        cv2.putText(
            frame, znorm_text, (16, 96),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 255), 1, cv2.LINE_AA,
        )

    if overlay2:
        cv2.putText(
            frame, overlay2, (16, 128),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA,
        )


# ── CLI entry point ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SALTE Realtime Fatigue Detection (V3 — HeadPoseNeutralizer)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-dir", default="MODELS",
        help="Directory containing best_model.onnx, inference_config.json, "
             "blazeface_detector.onnx, face_mesh_landmark.onnx",
    )
    parser.add_argument(
        "--model", default=None,
        help="Path to best_model.onnx (default: {model-dir}/best_model.onnx)",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to inference_config.json (default: {model-dir}/inference_config.json)",
    )
    parser.add_argument(
        "--detector-model", default=None,
        help="Path to blazeface_detector.onnx",
    )
    parser.add_argument(
        "--mesh-model", default=None,
        help="Path to face_mesh_landmark.onnx",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Override decision threshold (default: from inference_config, 0.41)",
    )
    parser.add_argument(
        "--picamera", action="store_true",
        help="Use picamera2 (AI Camera IMX500 via CSI)",
    )
    parser.add_argument(
        "--camera-index", type=int, default=0,
        help="USB webcam index (ignored with --picamera)",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="No display output (for SSH / no X11)",
    )
    parser.add_argument(
        "--warmup", type=float, default=120.0,
        help="Warm-up duration in seconds (C6-V2)",
    )
    parser.add_argument(
        "--min-warmup", type=float, default=30.0,
        help="Minimum seconds before allowing forced calibration",
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Target FPS",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print feature vector for each window (raw, scaled, neutralized)",
    )
    # ── FIX-RT-3: CLI toggle (C33) ──────────────────────────────────────
    parser.add_argument(
        "--no-neutralize-pose", action="store_true",
        help="Desativar neutralização de head pose (para testes com dados de lab). "
             "Default: neutralização ATIVADA (4 features de pose zeradas).",
    )
    # ── Memory + Parallel ────────────────────────────────────────────
    parser.add_argument(
        "--log-features", action="store_true",
        help="Gravar features de cada janela em CSV para retraining",
    )
    parser.add_argument(
        "--operator-id", type=str, default=None,
        help="Operator ID for session tracking and warm-start",
    )
    parser.add_argument(
        "--parallel", action="store_true",
        help="Ativar captura paralela via FrameGrabber (thread separada)",
    )
    parser.add_argument(
        "--no-agents", action="store_true",
        help="Desativar análise multi-agente (usar apenas MLP)",
    )
    # ── C29 Boundary Rules ────────────────────────────────────────────
    parser.add_argument(
        "--disable-c29", action="store_true",
        help="Desativar todas as regras C29 de head pose",
    )
    parser.add_argument(
        "--c29-override-only", action="store_true",
        help="Apenas R1 (override), sem boosts R2-R4",
    )
    parser.add_argument(
        "--c29-log", action="store_true",
        help="Logar avaliações C29 em c29_alerts.jsonl",
    )
    # ────────────────────────────────────────────────────────────────────
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    checkpoint = Path(args.model) if args.model else model_dir / "best_model.onnx"
    config_path = Path(args.config) if args.config else model_dir / "inference_config.json"
    detector_path = Path(args.detector_model) if args.detector_model else model_dir / "blazeface_detector.onnx"
    mesh_path = Path(args.mesh_model) if args.mesh_model else model_dir / "face_mesh_landmark.onnx"

    run_realtime(
        checkpoint_path=checkpoint,
        config_path=config_path,
        detector_path=detector_path,
        mesh_path=mesh_path,
        camera_index=args.camera_index,
        use_picamera=args.picamera,
        threshold_override=args.threshold,
        warmup_sec=args.warmup,
        min_warmup_sec=args.min_warmup,
        fps=args.fps,
        headless=args.headless,
        debug=args.debug,
        neutralize_pose=not args.no_neutralize_pose,
        log_features=args.log_features,
        operator_id=args.operator_id,
        parallel=args.parallel,
        no_agents=args.no_agents,
        disable_c29=args.disable_c29,
        c29_override_only=args.c29_override_only,
        c29_log=args.c29_log,
    )


if __name__ == "__main__":
    main()