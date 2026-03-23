"""
SALTE Inference V2 — Pipeline de inferência em tempo real (RPi 5).

Módulos:
- feature_extractor_rt: ONNXFaceMeshBackend (BlazeFace + FaceMesh ONNX), EAR/MAR/head pose
- subject_calibrator_rt: Calibração per-subject Z-Norm (C5 + C6-V2)
- window_factory_rt: Agregação de janelas com 19 features (TEV7 Agentic V3)
- model_loader: best_model.onnx, inference_config.json (scaler via JSON)
- guardrails: Validação estruturada de saída, entrada, calibração e comportamento (G1-G4)
- reflection: Detecção de drift, crítico de predição, auto-recalibração (R1-R3)
- memory: Memória de curto prazo (SessionMemory), longo prazo (OperatorStore), logger (M1-M3)
- parallel: Captura paralela (FrameGrabber), worker (PipelineWorker), métricas (P1-P3)
- agents: Multi-Agent Collaboration — OcularAgent, BlinkAgent, PosturalAgent, SupervisorAgent (A1-A5)
- run_realtime_demo: Loop principal câmera → Safe/Danger (picamera2 / webcam)
- offline_eval: Ferramentas de validação offline (dev-only, requer scikit-learn)
"""
