"""
agent_memory.py — Persistent Short-Term Reasoning Memory for EcoLoop Agent System.

Implements a fixed-capacity ring buffer backed by JSON on disk.
Every cycle writes a MemoryRecord. Every reasoning step reads the last N records.
Serialization is JSON so the file is human-readable for debugging.

Design principles:
    - Zero external dependencies (stdlib only)
    - Atomic write via temp-file rename (no partial reads)
    - summarize_for_prompt() injects memory directly into LLM context window
    - capacity=12 covers one full day at hourly throttling

Author: EcoLoop AI System
"""

import collections
import json
import os
import tempfile
from dataclasses import dataclass, field, asdict
from typing import List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CAPACITY = 12          # Number of cycles to retain in short-term memory
MEMORY_FILENAME  = "agent_memory.json"


# ─────────────────────────────────────────────────────────────────────────────
# Memory Record — one per completed control cycle
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MemoryRecord:
    """
    Full audit snapshot of one control cycle.
    Stored in the ring buffer and serialized to JSON.
    """
    timestamp:          str
    zone_temp:          float
    heating_sp:         float
    cooling_sp:         float
    outdoor_temp:       float
    action:             str
    coil_speed:         float
    reasoning:          str         # Truncated LLM chain-of-thought
    confidence:         float       # 0.0 – 1.0
    energy_kwh:         float       # Estimated energy this cycle
    comfort_deviation:  float       # |zone_temp - nearest_setpoint|
    outcome:            str         # "SUCCESS" | "CORRECTED" | "ROLLED_BACK" | "FALLBACK"
    success:            bool        # True if no violations triggered
    violations:         List[str]   # List of ViolationType names that fired
    # Economizer 4-Stage Pipeline & Telemetry Fields
    economizer_recommended:        bool  = False
    economizer_mode:               str   = "NO_ACTION"
    temperature_advantage:        float = 0.0
    estimated_runtime_saved_hours: float = 0.0
    estimated_energy_saved_kwh:   float = 0.0
    planner_accepted:              bool  = False
    validator_overrode:            bool  = False
    final_free_cooling_used:       bool  = False
    economizer_confidence:         float = 0.0
    # Demand Response 4-Stage Pipeline Fields
    is_peak_window:                bool  = False
    tariff_period:                 str   = "NORMAL"
    tariff_inr_kwh:                float = 10.0
    dr_recommended:                bool  = False
    dr_planner_accepted:           bool  = False
    dr_validator_overrode:         bool  = False
    dr_final_used:                 bool  = False
    dr_cost_saved_inr:             float = 0.0
    # Predictive Pre-Cooling 4-Stage Pipeline Fields
    precool_recommended:           bool  = False
    predicted_peak_outdoor_temp:  float = 0.0
    precool_planner_accepted:      bool  = False
    precool_validator_overrode:    bool  = False
    precool_final_used:            bool  = False

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "MemoryRecord":
        return MemoryRecord(
            timestamp         = d.get("timestamp", ""),
            zone_temp         = float(d.get("zone_temp", 0.0)),
            heating_sp        = float(d.get("heating_sp", 0.0)),
            cooling_sp        = float(d.get("cooling_sp", 0.0)),
            outdoor_temp      = float(d.get("outdoor_temp", 0.0)),
            action            = d.get("action", "off"),
            coil_speed        = float(d.get("coil_speed", 0.0)),
            reasoning         = d.get("reasoning", ""),
            confidence        = float(d.get("confidence", 0.0)),
            energy_kwh        = float(d.get("energy_kwh", 0.0)),
            comfort_deviation = float(d.get("comfort_deviation", 0.0)),
            outcome           = d.get("outcome", "FALLBACK"),
            success           = bool(d.get("success", False)),
            violations        = d.get("violations", []),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Short-Term Memory Container
# ─────────────────────────────────────────────────────────────────────────────

class ShortTermMemory:
    """
    Fixed-capacity ring buffer of MemoryRecord instances.
    Persists to JSON after every write. Loads from JSON on construction.

    Thread-safety: Not required — EnergyPlus plugins are single-threaded.
    """

    def __init__(self, capacity: int = DEFAULT_CAPACITY, persist_path: Optional[str] = None):
        self._capacity = capacity
        self._buffer: collections.deque = collections.deque(maxlen=capacity)
        self._persist_path = persist_path
        if persist_path:
            self._load()

    # ── Public Interface ──────────────────────────────────────────────────────

    def add(self, record: MemoryRecord) -> None:
        """Append a new record and persist to disk."""
        self._buffer.append(record)
        if self._persist_path:
            self._save()

    def get_recent(self, n: int = 6) -> List[MemoryRecord]:
        """Return the last n records, oldest first."""
        records = list(self._buffer)
        return records[-n:] if len(records) >= n else records

    def __len__(self) -> int:
        return len(self._buffer)

    def is_empty(self) -> bool:
        return len(self._buffer) == 0

    # ── Prompt Injection ──────────────────────────────────────────────────────

    def summarize_for_prompt(self, n: int = 6) -> str:
        """
        Render the last N cycles as a compact, structured block for injection
        into the LLM reasoning prompt. Provides agent temporal awareness.
        """
        recent = self.get_recent(n)
        if not recent:
            return "=== SHORT-TERM MEMORY ===\n(No prior cycles recorded — first invocation)"

        lines = ["=== SHORT-TERM MEMORY (last {} cycles) ===".format(len(recent))]
        for i, r in enumerate(recent, 1):
            comfort_str = f"{r.comfort_deviation:.3f}C deviation"
            energy_str  = f"{r.energy_kwh:.5f} kWh"
            flag        = "OK" if r.success else "FAIL"
            lines.append(
                f"  [{i}] {r.timestamp:>14}  "
                f"zone={r.zone_temp:.1f}C  outdoor={r.outdoor_temp:.1f}C  "
                f"action={r.action:<6}  speed={r.coil_speed:.1f}  "
                f"conf={r.confidence:.2f}  {comfort_str}  "
                f"{energy_str}  outcome={r.outcome}  [{flag}]"
            )
            if r.violations:
                lines.append(f"           WARN violations: {', '.join(r.violations)}")
            if r.reasoning:
                lines.append(f"           reasoning: {r.reasoning[:120]}...")
        return "\n".join(lines)

    def action_sequence(self) -> List[str]:
        """Return ordered list of recent action labels for oscillation detection."""
        return [r.action for r in self._buffer]

    def speed_sequence(self) -> List[float]:
        """Return ordered list of recent coil speeds."""
        return [r.coil_speed for r in self._buffer]

    def recent_energy_total(self, n: int = 6) -> float:
        """Sum of energy_kwh over the last n cycles."""
        return sum(r.energy_kwh for r in self.get_recent(n))

    def recent_avg_comfort_deviation(self, n: int = 6) -> float:
        """Mean comfort deviation over the last n cycles."""
        records = self.get_recent(n)
        if not records:
            return 0.0
        return sum(r.comfort_deviation for r in records) / len(records)

    def recent_success_rate(self, n: int = 6) -> float:
        """Fraction of recent cycles that were successful (no violations)."""
        records = self.get_recent(n)
        if not records:
            return 1.0
        return sum(1 for r in records if r.success) / len(records)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        """Atomically write buffer to JSON via temp-file rename."""
        if not self._persist_path:
            return
        try:
            os.makedirs(os.path.dirname(self._persist_path), exist_ok=True)
            records = [r.to_dict() for r in self._buffer]
            payload = {"capacity": self._capacity, "records": records}
            dir_path = os.path.dirname(self._persist_path)
            with tempfile.NamedTemporaryFile(
                mode="w", dir=dir_path, delete=False,
                suffix=".tmp", encoding="utf-8"
            ) as tf:
                json.dump(payload, tf, indent=2)
                tmp_name = tf.name
            os.replace(tmp_name, self._persist_path)
        except Exception:
            pass    # Never crash the EnergyPlus simulation due to memory I/O

    def _load(self) -> None:
        """Load existing memory from JSON on startup. Silent on missing file."""
        if not self._persist_path or not os.path.exists(self._persist_path):
            return
        try:
            with open(self._persist_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            for d in payload.get("records", []):
                self._buffer.append(MemoryRecord.from_dict(d))
        except Exception:
            self._buffer.clear()
