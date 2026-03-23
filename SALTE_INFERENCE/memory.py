"""
Memory Management para DeteccaoFadiga.

Três componentes:
  M1. SessionMemory    — estado de curto prazo (sessão atual em RAM)
  M2. OperatorStore    — memória de longo prazo (SQLite, persiste entre sessões)
  M3. FeatureLogger    — gravação de features para retraining futuro
"""

from __future__ import annotations

import csv
import json
import sqlite3
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional

import numpy as np


# ── M1: Memória de Curto Prazo ────────────────────────────────────────────


@dataclass
class SessionMemory:
    """
    M1: Memória de curto prazo — estado completo da sessão em execução.

    Centraliza toda informação acumulada durante uma sessão de monitoramento.
    Substitui variáveis dispersas no loop principal por um objeto coerente
    e consultável. Vive exclusivamente em RAM — destruído ao encerrar.

    Ref: Agentic Design Patterns Ch.8 — Short-term memory / Session State.
    """

    # ── Identificação ──
    operator_id: str = "unknown"
    session_start_time: float = field(default_factory=time.monotonic)

    # ── Contadores globais ──
    total_frames: int = 0
    total_windows: int = 0
    total_danger_windows: int = 0
    total_safe_windows: int = 0
    total_watch_windows: int = 0
    total_critical_windows: int = 0

    # ── Streaks ──
    max_consecutive_danger: int = 0
    current_consecutive_danger: int = 0

    # ── Calibração ──
    calibration_count: int = 0
    calibration_timestamps: List[float] = field(default_factory=list)
    last_calibration_ear_mean: float = 0.0
    last_calibration_verdict: str = ""   # "accept" | "retry" | "use_with_caution"

    # ── Trends (janela deslizante para análise temporal) ──
    prob_history: Deque[float] = field(
        default_factory=lambda: deque(maxlen=120)  # últimas 120 janelas (~30 min)
    )
    perclos_history: Deque[float] = field(
        default_factory=lambda: deque(maxlen=120)
    )
    ear_mean_history: Deque[float] = field(
        default_factory=lambda: deque(maxlen=120)
    )

    # ── Alertas ──
    total_alerts_triggered: int = 0
    total_alerts_suppressed: int = 0
    total_recalibrations_auto: int = 0

    # ── Microsleeps ──
    microsleep_timestamps: List[float] = field(default_factory=list)
    total_microsleep_ms: float = 0.0

    # ── C29 Boundary Rules ──
    total_c29_overrides: int = 0
    total_c29_boosts: int = 0
    total_c29_boost_magnitude: float = 0.0

    def on_window(
        self,
        label: str,
        prob_danger: float,
        alert_level_name: str,
        perclos: float,
        ear_mean: float,
        microsleep_count: float,
        microsleep_total_ms: float,
    ) -> None:
        """Registra uma nova janela de inferência na memória."""
        self.total_windows += 1
        self.prob_history.append(prob_danger)
        self.perclos_history.append(perclos)
        self.ear_mean_history.append(ear_mean)

        if label == "Danger":
            self.total_danger_windows += 1
            self.current_consecutive_danger += 1
            self.max_consecutive_danger = max(
                self.max_consecutive_danger,
                self.current_consecutive_danger,
            )
        else:
            self.current_consecutive_danger = 0
            self.total_safe_windows += 1

        if alert_level_name == "WATCH":
            self.total_watch_windows += 1
        elif alert_level_name == "CRITICAL":
            self.total_critical_windows += 1

        if microsleep_count > 0:
            now = time.monotonic()
            for _ in range(int(microsleep_count)):
                self.microsleep_timestamps.append(now)
            self.total_microsleep_ms += microsleep_total_ms

    def on_calibration(
        self, ear_mean: float, verdict: str
    ) -> None:
        """Registra uma calibração."""
        self.calibration_count += 1
        self.calibration_timestamps.append(time.monotonic())
        self.last_calibration_ear_mean = ear_mean
        self.last_calibration_verdict = verdict

    def on_auto_recalibration(self) -> None:
        self.total_recalibrations_auto += 1

    def on_c29_alert(self, override: bool, boost: float) -> None:
        """Registra uma ativacao de regra C29."""
        if override:
            self.total_c29_overrides += 1
        elif boost > 0:
            self.total_c29_boosts += 1
            self.total_c29_boost_magnitude += boost

    def on_alert(self, triggered: bool) -> None:
        if triggered:
            self.total_alerts_triggered += 1
        else:
            self.total_alerts_suppressed += 1

    @property
    def session_duration_sec(self) -> float:
        return time.monotonic() - self.session_start_time

    @property
    def danger_ratio(self) -> float:
        if self.total_windows == 0:
            return 0.0
        return self.total_danger_windows / self.total_windows

    @property
    def avg_prob_danger(self) -> float:
        if not self.prob_history:
            return 0.0
        return float(np.mean(self.prob_history))

    @property
    def perclos_trend_slope(self) -> float:
        """Slope do PERCLOS ao longo do tempo. Positivo = piorando."""
        if len(self.perclos_history) < 10:
            return 0.0
        y = np.array(self.perclos_history)
        x = np.arange(len(y))
        # Regressão linear simples
        n = len(x)
        slope = (n * np.dot(x, y) - x.sum() * y.sum()) / (
            n * np.dot(x, x) - x.sum() ** 2 + 1e-12
        )
        return float(slope)

    def summary(self) -> Dict[str, object]:
        """Resumo da sessão para logging ou display."""
        return {
            "operator_id": self.operator_id,
            "duration_min": round(self.session_duration_sec / 60, 1),
            "total_windows": self.total_windows,
            "danger_ratio": round(self.danger_ratio, 3),
            "avg_prob": round(self.avg_prob_danger, 3),
            "max_consec_danger": self.max_consecutive_danger,
            "total_microsleeps": len(self.microsleep_timestamps),
            "total_microsleep_ms": round(self.total_microsleep_ms, 0),
            "perclos_trend": round(self.perclos_trend_slope, 6),
            "calibrations": self.calibration_count,
            "auto_recalibrations": self.total_recalibrations_auto,
            "alerts_triggered": self.total_alerts_triggered,
            "c29_overrides": self.total_c29_overrides,
            "c29_boosts": self.total_c29_boosts,
            "c29_boost_total": round(self.total_c29_boost_magnitude, 3),
        }


# ── M2: Memória de Longo Prazo ───────────────────────────────────────────


class OperatorStore:
    """
    M2: Memória de longo prazo — persiste perfis entre sessões via SQLite.

    Armazena:
    - Baselines de calibração (ear_mean, ear_std, etc.) por operador
    - Resumos de sessões anteriores (danger_ratio, microsleeps, duração)
    - Timestamps de última sessão para cálculo de descanso

    Ref: Agentic Design Patterns Ch.8 — Long-term memory / Persistent Storage.
    """

    DB_NAME = "operator_memory.db"

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path = db_path or self.DB_NAME
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self) -> None:
        cur = self._conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS operator_profiles (
                operator_id   TEXT PRIMARY KEY,
                ear_mean      REAL,
                ear_std       REAL,
                mar_mean      REAL,
                pitch_mean    REAL,
                yaw_mean      REAL,
                sessions_count INTEGER DEFAULT 0,
                total_danger_ratio REAL DEFAULT 0.0,
                avg_session_minutes REAL DEFAULT 0.0,
                last_session_end TEXT,
                created_at    TEXT DEFAULT (datetime('now')),
                updated_at    TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS session_logs (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                operator_id   TEXT NOT NULL,
                started_at    TEXT NOT NULL,
                ended_at      TEXT DEFAULT (datetime('now')),
                duration_min  REAL,
                total_windows INTEGER,
                danger_ratio  REAL,
                avg_prob      REAL,
                max_consec_danger INTEGER,
                total_microsleeps INTEGER,
                total_microsleep_ms REAL,
                perclos_trend REAL,
                calibrations  INTEGER,
                summary_json  TEXT,
                FOREIGN KEY (operator_id) REFERENCES operator_profiles(operator_id)
            );
        """)
        self._conn.commit()

    def get_profile(self, operator_id: str) -> Optional[Dict]:
        """Retorna perfil do operador ou None se não existe."""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT * FROM operator_profiles WHERE operator_id = ?",
            (operator_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def upsert_profile_from_calibration(
        self,
        operator_id: str,
        baseline,  # SubjectBaseline
    ) -> None:
        """
        Atualiza ou cria perfil com dados da calibração atual.
        Usa média exponencial com o baseline anterior para suavizar.
        """
        existing = self.get_profile(operator_id)
        alpha = 0.3  # peso da calibração nova vs histórica

        if existing and existing["ear_mean"] is not None:
            ear_mean = alpha * baseline.ear_mean + (1 - alpha) * existing["ear_mean"]
            ear_std = alpha * baseline.ear_std + (1 - alpha) * existing["ear_std"]
            mar_mean = alpha * baseline.mar_mean + (1 - alpha) * existing["mar_mean"]
            pitch_mean = alpha * baseline.pitch_mean + (1 - alpha) * existing["pitch_mean"]
            yaw_mean = alpha * baseline.yaw_mean + (1 - alpha) * existing["yaw_mean"]
        else:
            ear_mean = baseline.ear_mean
            ear_std = baseline.ear_std
            mar_mean = baseline.mar_mean
            pitch_mean = baseline.pitch_mean
            yaw_mean = baseline.yaw_mean

        cur = self._conn.cursor()
        cur.execute("""
            INSERT INTO operator_profiles
                (operator_id, ear_mean, ear_std, mar_mean, pitch_mean, yaw_mean)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(operator_id) DO UPDATE SET
                ear_mean = ?, ear_std = ?, mar_mean = ?,
                pitch_mean = ?, yaw_mean = ?,
                updated_at = datetime('now')
        """, (
            operator_id, ear_mean, ear_std, mar_mean, pitch_mean, yaw_mean,
            ear_mean, ear_std, mar_mean, pitch_mean, yaw_mean,
        ))
        self._conn.commit()

    def save_session(
        self,
        operator_id: str,
        session_summary: Dict,
        started_at: str,
    ) -> None:
        """Persiste o resumo de uma sessão encerrada."""
        cur = self._conn.cursor()
        cur.execute("""
            INSERT INTO session_logs
                (operator_id, started_at, duration_min, total_windows,
                 danger_ratio, avg_prob, max_consec_danger,
                 total_microsleeps, total_microsleep_ms, perclos_trend,
                 calibrations, summary_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            operator_id,
            started_at,
            session_summary.get("duration_min", 0),
            session_summary.get("total_windows", 0),
            session_summary.get("danger_ratio", 0),
            session_summary.get("avg_prob", 0),
            session_summary.get("max_consec_danger", 0),
            session_summary.get("total_microsleeps", 0),
            session_summary.get("total_microsleep_ms", 0),
            session_summary.get("perclos_trend", 0),
            session_summary.get("calibrations", 0),
            json.dumps(session_summary),
        ))
        # Atualizar contadores no perfil
        cur.execute("""
            UPDATE operator_profiles SET
                sessions_count = sessions_count + 1,
                total_danger_ratio = (
                    total_danger_ratio * (sessions_count - 1) + ?
                ) / sessions_count,
                avg_session_minutes = (
                    avg_session_minutes * (sessions_count - 1) + ?
                ) / sessions_count,
                last_session_end = datetime('now'),
                updated_at = datetime('now')
            WHERE operator_id = ?
        """, (
            session_summary.get("danger_ratio", 0),
            session_summary.get("duration_min", 0),
            operator_id,
        ))
        self._conn.commit()

    def get_recent_sessions(
        self, operator_id: str, limit: int = 10
    ) -> List[Dict]:
        """Retorna as últimas N sessões do operador."""
        cur = self._conn.cursor()
        cur.execute("""
            SELECT * FROM session_logs
            WHERE operator_id = ?
            ORDER BY ended_at DESC LIMIT ?
        """, (operator_id, limit))
        return [dict(row) for row in cur.fetchall()]

    def get_warm_start_baseline(
        self, operator_id: str
    ) -> Optional[Dict]:
        """
        Retorna baseline suavizado do operador para warm-start.

        Se o operador já tem perfil com pelo menos 2 sessões,
        retorna os valores históricos como ponto de partida.
        Isso permite reduzir warm-up de 120s para 30s.
        """
        profile = self.get_profile(operator_id)
        if profile is None:
            return None
        if profile.get("sessions_count", 0) < 2:
            return None
        return {
            "ear_mean": profile["ear_mean"],
            "ear_std": profile["ear_std"],
            "mar_mean": profile["mar_mean"],
            "pitch_mean": profile["pitch_mean"],
            "yaw_mean": profile["yaw_mean"],
        }

    def close(self) -> None:
        self._conn.close()


# ── M3: Feature Logger para Retraining ───────────────────────────────────


class FeatureLogger:
    """
    M3: Gravação de features para retraining futuro (data flywheel).

    Grava um CSV com 19 features + label + prob + metadados por janela.
    O CSV fica no mesmo diretório dos modelos.
    Opt-in: desativado por padrão (--log-features para ativar).

    Formato: um arquivo por sessão, nomeado com timestamp.
    Não grava frames brutos (privacidade — apenas features numéricas).
    """

    def __init__(
        self,
        output_dir: str,
        feature_names: List[str],
        operator_id: str = "unknown",
        enabled: bool = False,
    ) -> None:
        self._enabled = enabled
        self._feature_names = feature_names
        self._operator_id = operator_id
        self._file = None
        self._writer = None
        self._count = 0

        if not enabled:
            return

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        filepath = out_path / f"features_{operator_id}_{ts}.csv"

        self._file = open(filepath, "w", newline="", encoding="utf-8")
        header = (
            ["timestamp_ms", "operator_id", "label", "prob_danger",
             "alert_level", "confidence"]
            + feature_names
        )
        self._writer = csv.writer(self._file)
        self._writer.writerow(header)
        print(f"[feature_log] Logging to {filepath}")

    def log(
        self,
        timestamp_ms: float,
        label: str,
        prob_danger: float,
        alert_level: str,
        confidence: str,
        window_feats: Dict[str, float],
    ) -> None:
        """Grava uma linha no CSV."""
        if not self._enabled or self._writer is None:
            return

        row = [
            f"{timestamp_ms:.1f}",
            self._operator_id,
            label,
            f"{prob_danger:.6f}",
            alert_level,
            confidence,
        ]
        for name in self._feature_names:
            row.append(f"{window_feats.get(name, 0.0):.6f}")

        self._writer.writerow(row)
        self._count += 1

        # Flush a cada 100 linhas para não perder dados em crash
        if self._count % 100 == 0:
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            print(f"[feature_log] {self._count} janelas gravadas")

    @property
    def count(self) -> int:
        return self._count
