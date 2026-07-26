"""
performance_tracer.py — High-Resolution Runtime Tracing & Performance Profiling.

Measures stage-by-stage latency, LLM HTTP POST duration, total cycle time,
and summary metrics using high-resolution monotonic timers (time.perf_counter()).

Saves measurements to:
  - logs/runtime_trace.csv
  - logs/runtime_summary.json

Design principles:
  - Zero performance impact on HVAC control logic
  - Context manager timing (with tracer.stage("stage_name"): ...)
  - Non-blocking logging (never crashes simulation)
  - Real measured numbers only — zero fabrication
"""

import csv
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Stage Names & CSV Headers
# ─────────────────────────────────────────────────────────────────────────────

STAGES = [
    "ep_callback",
    "sensor_collection",
    "mcp_sync",
    "memory_loading",
    "energy_optimizer",
    "comfort_optimizer",
    "confidence_engine",
    "planning_agent",
    "ollama_request",
    "validator",
    "decision_log_write",
    "explanation_engine",
    "memory_save",
    "actuator_update",
]

CSV_HEADERS = ["timestamp", "cycle_number"] + [f"{s}_ms" for s in STAGES] + ["total_cycle_ms"]


# ─────────────────────────────────────────────────────────────────────────────
# Tracer Class
# ─────────────────────────────────────────────────────────────────────────────

class PerformanceTracer:
    """
    High-resolution runtime performance profiling engine.
    Collects per-stage timings in milliseconds for each control cycle.
    """

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.trace_csv_path = os.path.join(log_dir, "runtime_trace.csv")
        self.summary_json_path = os.path.join(log_dir, "runtime_summary.json")

        self.cycle_count = 0
        self.cycle_timings: List[float] = []
        self.llm_timings: List[float] = []
        self.confidence_scores: List[float] = []
        self.fallback_count = 0
        self.circuit_breaker_trips = 0
        self.success_count = 0
        self.start_wall_time = time.time()

        self._current_stage_timings: Dict[str, float] = {}
        self._ensure_csv_header()

    def _ensure_csv_header(self) -> None:
        """Create CSV header if file does not exist."""
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            if not os.path.exists(self.trace_csv_path):
                with open(self.trace_csv_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(CSV_HEADERS)
        except Exception:
            pass

    def start_cycle(self) -> None:
        """Reset per-stage timing dictionary for a new cycle."""
        self._current_stage_timings = {s: 0.0 for s in STAGES}

    @contextmanager
    def stage(self, name: str):
        """Context manager to measure execution time of a specific stage in ms."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self._current_stage_timings[name] = round(dt_ms, 3)

    def record_stage_ms(self, name: str, duration_ms: float) -> None:
        """Manually record a stage duration in milliseconds."""
        self._current_stage_timings[name] = round(duration_ms, 3)

    def finish_cycle(
        self,
        timestamp: str,
        confidence: float = 0.0,
        outcome: str = "SUCCESS",
        llm_ok: bool = True,
        circuit_open: bool = False,
    ) -> float:
        """
        Finalize timings for the cycle, log to CSV, update summary JSON, and print console trace.
        """
        self.cycle_count += 1
        total_cycle_ms = sum(self._current_stage_timings.values())
        self._current_stage_timings["total_cycle_ms"] = round(total_cycle_ms, 3)

        self.cycle_timings.append(total_cycle_ms)
        if self._current_stage_timings.get("ollama_request"):
            self.llm_timings.append(self._current_stage_timings["ollama_request"])

        if confidence > 0:
            self.confidence_scores.append(confidence)

        if outcome == "FALLBACK" or not llm_ok:
            self.fallback_count += 1
        else:
            self.success_count += 1

        if circuit_open:
            self.circuit_breaker_trips += 1

        # Write to CSV
        try:
            row = [timestamp, self.cycle_count]
            for s in STAGES:
                row.append(self._current_stage_timings.get(s, 0.0))
            row.append(round(total_cycle_ms, 3))

            with open(self.trace_csv_path, "a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception:
            pass

        # Update Summary JSON
        self._update_summary_json()

        # Console output
        self._print_console_trace(timestamp, total_cycle_ms, confidence, outcome)

        return total_cycle_ms

    def _update_summary_json(self) -> None:
        """Compute aggregate stats and save to logs/runtime_summary.json."""
        try:
            avg_latency_ms = sum(self.cycle_timings) / len(self.cycle_timings) if self.cycle_timings else 0.0
            avg_llm_latency_ms = sum(self.llm_timings) / len(self.llm_timings) if self.llm_timings else 0.0
            avg_confidence = sum(self.confidence_scores) / len(self.confidence_scores) if self.confidence_scores else 0.0
            success_rate_pct = (self.success_count / self.cycle_count * 100.0) if self.cycle_count > 0 else 0.0
            total_runtime_s = round(time.time() - self.start_wall_time, 2)

            # Dynamic energy savings calculation from simulation summary JSONs (no hardcoded fallbacks)
            energy_savings_pct = None
            eco_sum_p = os.path.join(self.log_dir, "ecoloop_summary.json")
            base_sum_p = os.path.join(self.log_dir, "baseline_summary.json")

            if os.path.exists(eco_sum_p) and os.path.exists(base_sum_p):
                try:
                    with open(eco_sum_p, "r", encoding="utf-8") as f1, open(base_sum_p, "r", encoding="utf-8") as f2:
                        eco_data = json.load(f1)
                        base_data = json.load(f2)
                        eco_kwh = eco_data.get("total_facility_electricity_kwh")
                        base_kwh = base_data.get("total_facility_electricity_kwh")
                        if eco_kwh is not None and base_kwh is not None and base_kwh > 0:
                            energy_savings_pct = round((base_kwh - eco_kwh) / base_kwh * 100.0, 2)
                except Exception as exc:
                    print(f"[TRACER WARNING] Could not compute energy savings from summary files: {exc}")

            summary = {
                "total_cycles": self.cycle_count,
                "average_latency_ms": round(avg_latency_ms, 2),
                "average_llm_latency_ms": round(avg_llm_latency_ms, 2),
                "min_cycle_latency_ms": round(min(self.cycle_timings), 2) if self.cycle_timings else 0.0,
                "max_cycle_latency_ms": round(max(self.cycle_timings), 2) if self.cycle_timings else 0.0,
                "average_confidence": round(avg_confidence, 3),
                "fallback_count": self.fallback_count,
                "circuit_breaker_trips": self.circuit_breaker_trips,
                "success_rate_pct": round(success_rate_pct, 2),
                "energy_savings_pct": energy_savings_pct,
                "total_runtime_s": total_runtime_s,
                "last_updated_ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            with open(self.summary_json_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        except Exception:
            pass

    def _print_console_trace(
        self, timestamp: str, total_ms: float, confidence: float, outcome: str
    ) -> None:
        """Print a structured console trace for immediate developer visibility."""
        try:
            avg_ms = sum(self.cycle_timings) / len(self.cycle_timings) if self.cycle_timings else total_ms
            min_ms = min(self.cycle_timings) if self.cycle_timings else total_ms
            max_ms = max(self.cycle_timings) if self.cycle_timings else total_ms

            print("\n" + "=" * 80)
            print(f"[TIMER] ECOLOOP RUNTIME TRACE -- Cycle #{self.cycle_count} @ {timestamp} [{outcome}]")
            print("=" * 80)

            stage_labels = [
                ("ep_callback",        "1. EnergyPlus Callback  "),
                ("sensor_collection",  "2. Sensor Collection    "),
                ("mcp_sync",           "3. MCP Synchronization  "),
                ("memory_loading",     "4. Memory Loading       "),
                ("energy_optimizer",   "5. Energy Optimizer     "),
                ("comfort_optimizer",  "6. Comfort Optimizer    "),
                ("confidence_engine",  "7. Confidence Engine    "),
                ("planning_agent",     "8. Planning Agent       "),
                ("ollama_request",     "   [Ollama HTTP POST]  "),
                ("validator",          "9. Validator Agent      "),
                ("decision_log_write", "10. Decision Log Write  "),
                ("explanation_engine", "11. Explanation Engine  "),
                ("memory_save",        "12. Memory Save (Disk)  "),
                ("actuator_update",    "13. Actuator Update     "),
            ]

            for key, label in stage_labels:
                dur = self._current_stage_timings.get(key, 0.0)
                print(f"  {label} : {dur:8.2f} ms")

            print("-" * 80)
            print(
                f"TOTAL CYCLE LATENCY : {total_ms:8.2f} ms  "
                f"(Avg: {avg_ms:.2f} ms, Min: {min_ms:.2f} ms, Max: {max_ms:.2f} ms) | Conf: {confidence:.2f}"
            )
            print("=" * 80 + "\n")
        except Exception:
            pass
