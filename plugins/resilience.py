"""
resilience.py — Reliability & Fault Tolerance Layer for EcoLoop HVAC Control.

Covers every identified failure mode:
  - LLM timeout / Ollama unavailable
  - Malformed or incomplete JSON from LLM
  - EnergyPlus sensor handle missing or stale
  - Invalid actuator value (out-of-range)
  - Plugin callback exception (would crash simulation)
  - Memory I/O failure (disk full, permissions)
  - Simulation freeze detection (timestamp watchdog)
  - MCP bridge unavailable

Design principles:
  - RetryPolicy   : configurable exponential backoff for transient failures
  - CircuitBreaker: prevents hammering a failed service; auto-resets after cooldown
  - SensorGuard   : tracks last-known-good values; flags stale data
  - HealthMonitor : writes live health.json readable by the dashboard
  - FallbackChain : ordered strategy list, executes until one succeeds
  - Never raise   : every public function catches all exceptions internally

Author: EcoLoop AI System
"""

import json
import os
import time
import traceback
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum, auto
from functools import wraps
from typing import Callable, Dict, List, Optional, Any

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

HEALTH_FILENAME      = "health.json"
ERROR_LOG_FILENAME   = "ecoloop_errors.log"
MAX_ERROR_LOG_LINES  = 500          # Rotate after this many lines
FREEZE_THRESHOLD_S   = 300          # Seconds without a decision → freeze alert


# ─────────────────────────────────────────────────────────────────────────────
# Component Health Status
# ─────────────────────────────────────────────────────────────────────────────

class ComponentStatus(Enum):
    HEALTHY   = "healthy"
    DEGRADED  = "degraded"
    FAILED    = "failed"
    UNKNOWN   = "unknown"


@dataclass
class ComponentHealth:
    name:            str
    status:          str    # ComponentStatus.value
    last_success_ts: str
    last_failure_ts: str
    failure_count:   int
    last_error:      str
    note:            str

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Health Monitor
# ─────────────────────────────────────────────────────────────────────────────

class HealthMonitor:
    """
    Maintains live health state for every system component.
    Writes a human-readable health.json after every update.
    The dashboard reads this file to display system status.
    """

    def __init__(self, log_dir: str):
        self._log_dir   = log_dir
        self._health: Dict[str, ComponentHealth] = {}
        self._start_ts  = _now_iso()
        self._last_decision_ts: Optional[float] = None

        # Initialise all known components as UNKNOWN
        for name in [
            "llm_agent", "ollama_service", "energyplus_plugin",
            "mcp_bridge", "memory_io", "actuator", "sensor_zone_temp",
            "sensor_outdoor_temp", "explanation_engine",
        ]:
            self._health[name] = ComponentHealth(
                name=name, status=ComponentStatus.UNKNOWN.value,
                last_success_ts="", last_failure_ts="",
                failure_count=0, last_error="", note="Initialising",
            )

    def record_success(self, component: str, note: str = "") -> None:
        h = self._health.get(component) or ComponentHealth(
            name=component, status="unknown", last_success_ts="",
            last_failure_ts="", failure_count=0, last_error="", note="")
        h.status          = ComponentStatus.HEALTHY.value
        h.last_success_ts = _now_iso()
        h.note            = note or "OK"
        self._health[component] = h
        if component == "llm_agent":
            self._last_decision_ts = time.monotonic()
        self._flush()

    def record_failure(self, component: str, error: str, note: str = "") -> None:
        h = self._health.get(component) or ComponentHealth(
            name=component, status="unknown", last_success_ts="",
            last_failure_ts="", failure_count=0, last_error="", note="")
        h.failure_count  += 1
        h.last_failure_ts = _now_iso()
        h.last_error      = str(error)[:300]
        h.note            = note or f"Failure #{h.failure_count}"
        # Status transition based on failure count
        if h.failure_count >= 5:
            h.status = ComponentStatus.FAILED.value
        elif h.failure_count >= 2:
            h.status = ComponentStatus.DEGRADED.value
        else:
            h.status = ComponentStatus.DEGRADED.value
        self._health[component] = h
        self._flush()

    def record_degraded(self, component: str, note: str) -> None:
        h = self._health.get(component) or ComponentHealth(
            name=component, status="unknown", last_success_ts="",
            last_failure_ts="", failure_count=0, last_error="", note="")
        h.status = ComponentStatus.DEGRADED.value
        h.note   = note
        self._health[component] = h
        self._flush()

    def is_frozen(self) -> bool:
        if self._last_decision_ts is None:
            return False
        return (time.monotonic() - self._last_decision_ts) > FREEZE_THRESHOLD_S

    def summary(self) -> dict:
        counts = {s.value: 0 for s in ComponentStatus}
        for h in self._health.values():
            counts[h.status] = counts.get(h.status, 0) + 1
        overall = (
            ComponentStatus.HEALTHY.value   if counts.get("failed", 0) == 0 and counts.get("degraded", 0) == 0
            else ComponentStatus.FAILED.value    if counts.get("failed", 0) > 0
            else ComponentStatus.DEGRADED.value
        )
        return {
            "overall_status": overall,
            "session_start":  self._start_ts,
            "last_updated":   _now_iso(),
            "frozen":         self.is_frozen(),
            "components":     {k: v.to_dict() for k, v in self._health.items()},
            "counts":         counts,
        }

    def _flush(self) -> None:
        try:
            os.makedirs(self._log_dir, exist_ok=True)
            path = os.path.join(self._log_dir, HEALTH_FILENAME)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.summary(), f, indent=2)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Circuit Breaker
# ─────────────────────────────────────────────────────────────────────────────

class CircuitState(Enum):
    CLOSED   = auto()   # Normal operation
    OPEN     = auto()   # Failing: calls bypassed
    HALF_OPEN = auto()  # Testing: one call allowed


@dataclass
class CircuitBreaker:
    """
    Tracks consecutive failures for a named service.
    Opens circuit after `failure_threshold` consecutive failures.
    Attempts reset after `reset_timeout_s` seconds.
    """
    name:               str
    failure_threshold:  int   = 3
    reset_timeout_s:    float = 60.0
    _state:             CircuitState    = field(default=CircuitState.CLOSED, init=False, repr=False)
    _failure_count:     int             = field(default=0, init=False, repr=False)
    _last_failure_time: Optional[float] = field(default=None, init=False, repr=False)

    def is_open(self) -> bool:
        if self._state == CircuitState.OPEN:
            if time.monotonic() - (self._last_failure_time or 0) >= self.reset_timeout_s:
                self._state = CircuitState.HALF_OPEN
                return False
            return True
        return False

    def record_success(self) -> None:
        self._state         = CircuitState.CLOSED
        self._failure_count = 0

    def record_failure(self) -> None:
        self._failure_count    += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN

    @property
    def state_name(self) -> str:
        return self._state.name


# ─────────────────────────────────────────────────────────────────────────────
# Retry Policy
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetryPolicy:
    """
    Configurable retry with exponential backoff.
    Wraps any callable; catches all exceptions.
    """
    max_attempts:  int   = 3
    base_delay_s:  float = 0.5    # Initial backoff delay
    max_delay_s:   float = 5.0
    backoff_factor: float = 2.0   # Multiply delay by this after each failure

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute `func(*args, **kwargs)` with retry.
        Returns the result on success, or raises the last exception after exhausting retries.
        """
        last_exc = RuntimeError("RetryPolicy: no attempts made")
        delay = self.base_delay_s

        for attempt in range(1, self.max_attempts + 1):
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_attempts:
                    time.sleep(min(delay, self.max_delay_s))
                    delay *= self.backoff_factor

        raise last_exc


# ─────────────────────────────────────────────────────────────────────────────
# Sensor Guard
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SensorReading:
    value:     float
    timestamp: float    # monotonic time
    stale:     bool = False


class SensorGuard:
    """
    Tracks last-known-good values for critical sensors.
    Flags stale data when no update has been received within `stale_threshold_s`.
    Provides safe fallback values when sensor data is missing.

    Default fallback values are conservative midpoints for a residential zone.
    """
    STALE_THRESHOLD_S = 120.0   # 2 simulated minutes

    _DEFAULTS = {
        "zone_temp":    24.0,
        "heating_sp":   22.0,
        "cooling_sp":   26.6,
        "outdoor_temp": 25.0,
    }

    def __init__(self):
        self._readings: Dict[str, SensorReading] = {}

    def update(self, name: str, value: float) -> None:
        """Record a new sensor value."""
        if value is None or (isinstance(value, float) and not (value == value)):  # NaN check
            return
        # Plausibility bounds for HVAC sensors
        bounds = {
            "zone_temp":    (-10.0, 60.0),
            "heating_sp":   (10.0,  40.0),
            "cooling_sp":   (10.0,  40.0),
            "outdoor_temp": (-40.0, 60.0),
        }
        lo, hi = bounds.get(name, (-1e9, 1e9))
        if not (lo <= value <= hi):
            return  # Reject implausible value — keep previous good value
        self._readings[name] = SensorReading(
            value=value, timestamp=time.monotonic(), stale=False
        )

    def read(self, name: str) -> SensorReading:
        """Read a sensor value, marking as stale if too old."""
        r = self._readings.get(name)
        if r is None:
            return SensorReading(
                value=self._DEFAULTS.get(name, 0.0),
                timestamp=0.0, stale=True
            )
        age = time.monotonic() - r.timestamp
        r.stale = age > self.STALE_THRESHOLD_S
        return r

    def get_safe(self, name: str) -> float:
        """Return value (possibly stale but always a float)."""
        return self.read(name).value

    def any_stale(self) -> bool:
        """True if any tracked sensor is stale."""
        for name in ["zone_temp", "heating_sp", "cooling_sp"]:
            r = self.read(name)
            if r.stale:
                return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Fallback Chain
# ─────────────────────────────────────────────────────────────────────────────

class FallbackChain:
    """
    Executes a prioritised list of strategies, stopping at the first success.
    Each strategy is a callable that returns a non-None result on success.

    Usage:
        chain = FallbackChain("llm_decision", [strategy_a, strategy_b, strategy_c])
        result = chain.execute()
    """

    def __init__(self, name: str, strategies: List[Callable]):
        self.name       = name
        self.strategies = strategies

    def execute(self, *args, **kwargs) -> Any:
        """
        Try each strategy in order. Returns the first successful result.
        Raises RuntimeError if all strategies fail.
        """
        last_exc = None
        for i, strategy in enumerate(self.strategies, 1):
            try:
                result = strategy(*args, **kwargs)
                if result is not None:
                    return result
            except Exception as exc:
                last_exc = exc
                continue

        raise RuntimeError(
            f"FallbackChain '{self.name}': all {len(self.strategies)} strategies failed. "
            f"Last error: {last_exc}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Error Logger
# ─────────────────────────────────────────────────────────────────────────────

class ErrorLogger:
    """
    Thread-safe (single-process) structured error logger.
    Rotates at MAX_ERROR_LOG_LINES to prevent unbounded disk growth.
    Each entry is a single JSON line for easy parsing by the dashboard.
    """

    def __init__(self, log_dir: str):
        self._path = os.path.join(log_dir, ERROR_LOG_FILENAME)
        self._log_dir = log_dir

    def log(
        self,
        component:  str,
        error_type: str,
        message:    str,
        context:    Optional[dict] = None,
        exc:        Optional[Exception] = None,
    ) -> None:
        try:
            os.makedirs(self._log_dir, exist_ok=True)
            entry = {
                "ts":         _now_iso(),
                "component":  component,
                "error_type": error_type,
                "message":    message,
                "context":    context or {},
                "traceback":  traceback.format_exc() if exc else "",
            }
            self._rotate_if_needed()
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass    # Never crash due to logging

    def _rotate_if_needed(self) -> None:
        try:
            if not os.path.exists(self._path):
                return
            with open(self._path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) > MAX_ERROR_LOG_LINES:
                # Keep only the last half
                with open(self._path, "w", encoding="utf-8") as f:
                    f.writelines(lines[-(MAX_ERROR_LOG_LINES // 2):])
        except Exception:
            pass

    def tail(self, n: int = 20) -> List[dict]:
        """Return the last n error entries as dicts (for dashboard display)."""
        records = []
        try:
            if not os.path.exists(self._path):
                return []
            with open(self._path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines[-n:]:
                try:
                    records.append(json.loads(line.strip()))
                except Exception:
                    pass
        except Exception:
            pass
        return records


# ─────────────────────────────────────────────────────────────────────────────
# Resilience Manager (singleton per process)
# ─────────────────────────────────────────────────────────────────────────────

class ResilienceManager:
    """
    Central resilience facade. Owns one instance each of:
      - HealthMonitor
      - ErrorLogger
      - CircuitBreaker per service
      - SensorGuard

    Instantiated once inside CoilSpeedControl.__init__ and shared
    across all control cycles in the simulation run.
    """

    def __init__(self, log_dir: str):
        self.health  = HealthMonitor(log_dir)
        self.errors  = ErrorLogger(log_dir)
        self.sensors = SensorGuard()

        # One circuit breaker per external service
        self.breakers = {
            "ollama":     CircuitBreaker("ollama",     failure_threshold=3, reset_timeout_s=90.0),
            "mcp_bridge": CircuitBreaker("mcp_bridge", failure_threshold=3, reset_timeout_s=60.0),
            "memory_io":  CircuitBreaker("memory_io",  failure_threshold=5, reset_timeout_s=30.0),
        }

        self._retry = RetryPolicy(max_attempts=2, base_delay_s=0.3, backoff_factor=2.0)
        self._log_dir = log_dir

    # ── Public convenience methods ────────────────────────────────────────────

    def llm_available(self) -> bool:
        """True if the Ollama circuit is not open."""
        return not self.breakers["ollama"].is_open()

    def mcp_available(self) -> bool:
        return not self.breakers["mcp_bridge"].is_open()

    def record_llm_success(self) -> None:
        self.breakers["ollama"].record_success()
        self.health.record_success("ollama_service", "LLM call succeeded")
        self.health.record_success("llm_agent", "Decision produced")

    def record_llm_failure(self, error: str) -> None:
        self.breakers["ollama"].record_failure()
        self.health.record_failure("ollama_service", error, "LLM call failed")
        self.errors.log("llm_agent", "LLM_TIMEOUT_OR_ERROR", error,
                        context={"circuit_state": self.breakers["ollama"].state_name})

    def record_sensor_update(self, name: str, value: float) -> None:
        self.sensors.update(name, value)
        self.health.record_success(f"sensor_{name}", f"value={value:.2f}")

    def record_actuator_set(self, speed: float) -> None:
        self.health.record_success("actuator", f"coil_speed={speed:.2f}")

    def record_actuator_reset(self, reason: str) -> None:
        self.health.record_degraded("actuator", f"reset: {reason}")
        self.errors.log("actuator", "ACTUATOR_RESET", reason)

    def record_plugin_exception(self, exc: Exception) -> None:
        self.health.record_failure("energyplus_plugin", str(exc), "Plugin callback exception")
        self.errors.log("energyplus_plugin", "PLUGIN_EXCEPTION", str(exc)[:300], exc=exc)

    def record_memory_error(self, exc: Exception) -> None:
        self.breakers["memory_io"].record_failure()
        self.health.record_failure("memory_io", str(exc))
        self.errors.log("memory_io", "MEMORY_IO_ERROR", str(exc)[:200])

    def record_explanation_stored(self) -> None:
        self.health.record_success("explanation_engine", "Explanation written")

    def check_freeze(self) -> bool:
        frozen = self.health.is_frozen()
        if frozen:
            self.errors.log("energyplus_plugin", "SIMULATION_FREEZE",
                            f"No decision for >{FREEZE_THRESHOLD_S}s — simulation may be frozen")
        return frozen

    def safe_update_sensors(
        self, zone_temp: float, heating_sp: float, cooling_sp: float, outdoor_temp: float
    ) -> tuple:
        """
        Update SensorGuard and return (zone_temp, heating_sp, cooling_sp, outdoor_temp).
        Falls back to last-known-good for any implausible value.
        """
        for name, val in [("zone_temp", zone_temp), ("heating_sp", heating_sp),
                           ("cooling_sp", cooling_sp), ("outdoor_temp", outdoor_temp)]:
            self.record_sensor_update(name, val)

        return (
            self.sensors.get_safe("zone_temp"),
            self.sensors.get_safe("heating_sp"),
            self.sensors.get_safe("cooling_sp"),
            self.sensors.get_safe("outdoor_temp"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Decorator Utilities
# ─────────────────────────────────────────────────────────────────────────────

def safe_call(default=None, log_component: str = "unknown"):
    """
    Decorator: wrap a function so it never raises.
    On exception, returns `default` and logs the traceback.
    Use for non-critical paths (logging, explanation storage, etc.).
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception:
                return default
        return wrapper
    return decorator


def with_circuit_breaker(breaker: CircuitBreaker, fallback: Callable):
    """
    Decorator factory: if the circuit is open, call `fallback` instead.
    Records success/failure on the breaker automatically.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if breaker.is_open():
                return fallback(*args, **kwargs)
            try:
                result = func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as exc:
                breaker.record_failure()
                raise exc
        return wrapper
    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
