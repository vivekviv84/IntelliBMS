"""
closed_loop_controller.py — Self-Correcting Closed-Loop HVAC Controller.

Implements the full Observe→Reason→Act→Evaluate→Correct→Store control loop.
Automatically detects: comfort violations, energy surges, oscillations,
conflicting actuator states. Performs rollback and magnitude reduction.

Architecture:
    ┌──────────────┐
    │   Observe    │  ← sensor bundle from EnergyPlus
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │    Reason    │  ← ReasoningAgent (LLM chain-of-thought)
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │     Act      │  ← Apply actuator command
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │   Evaluate   │  ← Compare pre/post state vs expected
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │   Correct    │  ← Rollback / reduce magnitude / retry
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │    Store     │  ← Persist outcome to ring buffer
    └─────────────-┘

Author: EcoLoop AI System
"""

import collections
import csv
import json
import os
import time
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Optional, List, Tuple

from reasoning_agent import (
    ObservationContext,
    ReasoningDecision,
    ACTION_TO_SPEED,
    get_reasoning_decision,
)

# ─────────────────────────────────────────────────────────────────────────────
# Tunable Control Parameters
# ─────────────────────────────────────────────────────────────────────────────

HISTORY_BUFFER_LEN       = 12      # Rolling window for action/state history
MAX_CORRECTION_RETRIES   = 2       # Max rollback attempts per control cycle
COMFORT_VIOLATION_THRESH = 0.8     # °C: absolute deviation that triggers violation alert
ENERGY_SURGE_RATIO       = 1.4     # Coil speed increase ratio that triggers surge flag
OSCILLATION_WINDOW       = 6       # Number of recent actions to check for oscillation
OSCILLATION_FLIP_THRESH  = 3       # Min direction changes in window to flag oscillation
CONFIDENCE_FLOOR         = 0.35    # Minimum confidence below which we force fallback
COOLDOWN_CYCLES          = 2       # Cycles to suppress escalation after a rollback
SPEED_REDUCTION_STEP     = 0.5     # Amount to reduce coil_speed on each rollback step

# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class ViolationType(Enum):
    COMFORT_VIOLATION  = auto()
    ENERGY_SURGE       = auto()
    OSCILLATION        = auto()
    CONFLICTING_STATE  = auto()
    LOW_CONFIDENCE     = auto()


class CycleOutcome(Enum):
    SUCCESS            = auto()
    CORRECTED          = auto()
    ROLLED_BACK        = auto()
    FALLBACK           = auto()


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ControllerState:
    """Snapshot of zone state at a single evaluation point."""
    timestamp: str
    zone_temp: float
    heating_sp: float
    cooling_sp: float
    outdoor_temp: float
    coil_speed: float
    action: str

    @property
    def active_deviation(self) -> float:
        cooling_dev = max(0.0, self.zone_temp - self.cooling_sp)
        heating_dev = max(0.0, self.heating_sp - self.zone_temp)
        return max(cooling_dev, heating_dev)

    @property
    def in_comfort_band(self) -> bool:
        return self.active_deviation == 0.0


@dataclass
class CycleRecord:
    """Full audit record for one completed control cycle, written to disk."""
    timestamp: str
    zone_temp: float
    heating_sp: float
    cooling_sp: float
    outdoor_temp: float
    action: str
    coil_speed: float
    reasoning: str
    expected_energy_impact: str
    expected_comfort_impact: str
    confidence_score: float
    violations_detected: str        # comma-separated ViolationType names
    corrections_applied: int
    outcome: str                    # CycleOutcome.name
    llm_ok: bool
    via_mcp: bool


# ─────────────────────────────────────────────────────────────────────────────
# Violation Detector
# ─────────────────────────────────────────────────────────────────────────────

class ViolationDetector:
    """
    Stateless detector that evaluates a new decision against history
    and returns a set of detected violation types.
    """

    @staticmethod
    def detect(
        decision: ReasoningDecision,
        obs: ObservationContext,
        history: List[ControllerState],
    ) -> List[ViolationType]:
        violations = []

        # ── 1. Comfort Violation ─────────────────────────────────────────────
        active_dev = max(obs.cooling_deviation, obs.heating_deviation)
        if active_dev > COMFORT_VIOLATION_THRESH and decision.action == "off":
            violations.append(ViolationType.COMFORT_VIOLATION)

        # ── 2. Energy Surge ──────────────────────────────────────────────────
        if history:
            recent_speeds = [h.coil_speed for h in list(history)[-4:] if h.coil_speed > 0]
            if recent_speeds:
                avg_recent = sum(recent_speeds) / len(recent_speeds)
                if decision.coil_speed > avg_recent * ENERGY_SURGE_RATIO and decision.coil_speed > 1.0:
                    violations.append(ViolationType.ENERGY_SURGE)

        # ── 3. Oscillation Detection ─────────────────────────────────────────
        if len(history) >= OSCILLATION_WINDOW:
            recent_actions = [h.action for h in list(history)[-OSCILLATION_WINDOW:]]
            flip_count = sum(
                1 for i in range(1, len(recent_actions))
                if recent_actions[i] != recent_actions[i - 1]
            )
            if flip_count >= OSCILLATION_FLIP_THRESH:
                violations.append(ViolationType.OSCILLATION)

        # ── 4. Conflicting Actuator State ────────────────────────────────────
        # Boost cooling while outdoor temp is already below heating setpoint
        if (decision.action == "boost" and
                obs.outdoor_temp < obs.heating_sp - 2.0 and
                obs.cooling_deviation == 0.0):
            violations.append(ViolationType.CONFLICTING_STATE)

        # ── 5. Low Confidence ────────────────────────────────────────────────
        if decision.confidence_score < CONFIDENCE_FLOOR:
            violations.append(ViolationType.LOW_CONFIDENCE)

        return violations


# ─────────────────────────────────────────────────────────────────────────────
# Correction Engine
# ─────────────────────────────────────────────────────────────────────────────

class CorrectionEngine:
    """
    Given a detected set of violations and the original decision,
    produces a corrected action. Does NOT call the LLM again — corrections
    are deterministic to avoid feedback loops.
    """

    @staticmethod
    def correct(
        decision: ReasoningDecision,
        violations: List[ViolationType],
        obs: ObservationContext,
        cooldown_active: bool,
    ) -> Tuple[ReasoningDecision, bool]:
        """
        Returns (corrected_decision, was_modified).
        Applies corrections in priority order: safety → energy → oscillation.
        """
        modified = False
        action = decision.action
        speed = decision.coil_speed

        # Priority 1: Comfort violation — must activate at least "eco"
        if ViolationType.COMFORT_VIOLATION in violations:
            if action == "off":
                action = "eco"
                speed = ACTION_TO_SPEED["eco"]
                modified = True

        # Priority 2: Low confidence — step down one tier
        if ViolationType.LOW_CONFIDENCE in violations and not modified:
            action, speed = CorrectionEngine._step_down(action, speed)
            modified = True

        # Priority 3: Energy surge — roll back one tier
        if ViolationType.ENERGY_SURGE in violations and not modified:
            action, speed = CorrectionEngine._step_down(action, speed)
            modified = True

        # Priority 4: Oscillation — force stabilisation to "eco" or "off"
        if ViolationType.OSCILLATION in violations:
            active_dev = max(obs.cooling_deviation, obs.heating_deviation)
            if active_dev > 0.3:
                action, speed = "eco", ACTION_TO_SPEED["eco"]
            else:
                action, speed = "off", ACTION_TO_SPEED["off"]
            modified = True

        # Priority 5: Conflicting state — reduce to eco
        if ViolationType.CONFLICTING_STATE in violations:
            action, speed = "eco", ACTION_TO_SPEED["eco"]
            modified = True

        # Priority 6: Cooldown suppression — do not escalate after rollback
        if cooldown_active and speed > decision.coil_speed:
            action, speed = decision.action, decision.coil_speed
            modified = True

        if modified:
            corrected = ReasoningDecision(
                action=action,
                coil_speed=speed,
                reasoning=decision.reasoning + f" [CORRECTED: violations={[v.name for v in violations]}]",
                expected_energy_impact=decision.expected_energy_impact,
                expected_comfort_impact=decision.expected_comfort_impact,
                confidence_score=decision.confidence_score,
                rationale=f"{decision.rationale} [CORRECTED due to {', '.join(v.name for v in violations)}]",
                ok=decision.ok,
            )
            return corrected, True

        return decision, False

    @staticmethod
    def _step_down(action: str, speed: float) -> Tuple[str, float]:
        """Reduce action magnitude by one tier."""
        tier_order = ["boost", "normal", "eco", "off"]
        try:
            idx = tier_order.index(action)
            new_action = tier_order[min(idx + 1, len(tier_order) - 1)]
            return new_action, ACTION_TO_SPEED[new_action]
        except ValueError:
            # Speed-based fallback: subtract fixed step
            new_speed = max(0.0, speed - SPEED_REDUCTION_STEP)
            for act, spd in sorted(ACTION_TO_SPEED.items(), key=lambda x: x[1], reverse=True):
                if new_speed >= spd:
                    return act, spd
            return "off", 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Main Controller
# ─────────────────────────────────────────────────────────────────────────────

class ClosedLoopController:
    """
    Production-grade self-correcting HVAC closed-loop controller.

    Each call to run_cycle() executes the full:
        Observe → Reason → Act → Evaluate → Correct → Store

    Internal state is maintained across EnergyPlus sub-iterations via
    instance variables. Thread-safety is not required (single-threaded EP plugin).
    """

    def __init__(self, log_file_path: str):
        self.log_file_path = log_file_path
        self._history: collections.deque = collections.deque(maxlen=HISTORY_BUFFER_LEN)
        self._cooldown_remaining: int = 0
        self._last_decision: Optional[ReasoningDecision] = None
        self._detector = ViolationDetector()
        self._corrector = CorrectionEngine()
        self._cycle_count: int = 0
        self._ensure_log_file()

    # ── Public Interface ──────────────────────────────────────────────────────

    def run_cycle(
        self,
        zone_temp: float,
        heating_sp: float,
        cooling_sp: float,
        outdoor_temp: float,
        timestamp: str,
        hour: int,
        occupancy: float = 1.0,
        hvac_mode: str = "auto",
        recent_energy_kwh: float = 0.0,
        comfort_pmv: Optional[float] = None,
        via_mcp: bool = False,
    ) -> ReasoningDecision:
        """
        Execute one complete control cycle.
        Returns the final (possibly corrected) decision to apply to the actuator.
        """
        self._cycle_count += 1

        # ── OBSERVE ───────────────────────────────────────────────────────────
        obs = self._observe(
            zone_temp, heating_sp, cooling_sp, outdoor_temp,
            timestamp, hour, occupancy, hvac_mode,
            recent_energy_kwh, comfort_pmv,
        )

        # ── REASON ───────────────────────────────────────────────────────────
        decision = self._reason(obs)

        # ── EVALUATE & CORRECT (up to MAX_CORRECTION_RETRIES) ─────────────
        final_decision, outcome = self._evaluate_and_correct(decision, obs)

        # ── STORE ─────────────────────────────────────────────────────────────
        self._store(obs, final_decision, outcome, via_mcp)

        # ── UPDATE COOLDOWN ───────────────────────────────────────────────────
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        self._last_decision = final_decision
        return final_decision

    @property
    def action_history(self) -> List[str]:
        return [h.action for h in self._history]

    @property
    def speed_history(self) -> List[float]:
        return [h.coil_speed for h in self._history]

    # ── Private Methods ───────────────────────────────────────────────────────

    def _observe(
        self,
        zone_temp: float,
        heating_sp: float,
        cooling_sp: float,
        outdoor_temp: float,
        timestamp: str,
        hour: int,
        occupancy: float,
        hvac_mode: str,
        recent_energy_kwh: float,
        comfort_pmv: Optional[float],
    ) -> ObservationContext:
        """Build a rich observation context, injecting controller history."""
        return ObservationContext(
            zone_temp=zone_temp,
            heating_sp=heating_sp,
            cooling_sp=cooling_sp,
            outdoor_temp=outdoor_temp,
            timestamp=timestamp,
            time_of_day_hour=hour,
            occupancy=occupancy,
            hvac_mode=hvac_mode,
            recent_energy_kwh=recent_energy_kwh,
            comfort_pmv=comfort_pmv,
            historical_actions=self.action_history,
            historical_speeds=self.speed_history,
        )

    def _reason(self, obs: ObservationContext) -> ReasoningDecision:
        """Call the reasoning agent. Never raises; always returns a decision."""
        try:
            return get_reasoning_decision(obs, retries=1)
        except Exception as exc:
            from reasoning_agent import _safety_fallback
            return _safety_fallback(obs, reason=str(exc))

    def _evaluate_and_correct(
        self,
        initial_decision: ReasoningDecision,
        obs: ObservationContext,
    ) -> Tuple[ReasoningDecision, CycleOutcome]:
        """
        Run up to MAX_CORRECTION_RETRIES rounds of violation detection + correction.
        Stops early if no violations detected, or if no more corrections possible.
        """
        decision = initial_decision
        total_corrections = 0
        outcome = CycleOutcome.SUCCESS

        for attempt in range(MAX_CORRECTION_RETRIES):
            # Detect violations in this candidate decision
            violations = self._detector.detect(decision, obs, list(self._history))

            if not violations:
                break   # No violations — accept decision

            # Apply deterministic corrections
            corrected, was_modified = self._corrector.correct(
                decision, violations, obs,
                cooldown_active=(self._cooldown_remaining > 0)
            )

            if was_modified:
                total_corrections += 1
                decision = corrected
                outcome = CycleOutcome.CORRECTED

                # If action was stepped down significantly, trigger cooldown
                original_speed = initial_decision.coil_speed
                if decision.coil_speed < original_speed - SPEED_REDUCTION_STEP:
                    self._cooldown_remaining = COOLDOWN_CYCLES
                    outcome = CycleOutcome.ROLLED_BACK
            else:
                break   # Corrector made no change — stop loop

        if not initial_decision.ok:
            outcome = CycleOutcome.FALLBACK

        return decision, outcome

    def _store(
        self,
        obs: ObservationContext,
        decision: ReasoningDecision,
        outcome: CycleOutcome,
        via_mcp: bool,
    ) -> None:
        """Persist the cycle record and update internal history buffer."""
        # Update ring buffer
        snap = ControllerState(
            timestamp=obs.timestamp,
            zone_temp=obs.zone_temp,
            heating_sp=obs.heating_sp,
            cooling_sp=obs.cooling_sp,
            outdoor_temp=obs.outdoor_temp,
            coil_speed=decision.coil_speed,
            action=decision.action,
        )
        self._history.append(snap)

        # Write extended CSV log
        try:
            header_needed = not os.path.exists(self.log_file_path)
            os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
            with open(self.log_file_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if header_needed:
                    writer.writerow([
                        "timestamp", "zone_temp", "heating_sp", "cooling_sp",
                        "outdoor_temp", "action", "coil_speed",
                        "reasoning", "expected_energy_impact", "expected_comfort_impact",
                        "confidence_score", "violations_detected", "outcome",
                        "llm_ok", "via_mcp"
                    ])
                writer.writerow([
                    obs.timestamp,
                    f"{obs.zone_temp:.3f}",
                    f"{obs.heating_sp:.3f}",
                    f"{obs.cooling_sp:.3f}",
                    f"{obs.outdoor_temp:.3f}",
                    decision.action,
                    f"{decision.coil_speed:.2f}",
                    decision.reasoning[:400],
                    decision.expected_energy_impact[:150],
                    decision.expected_comfort_impact[:150],
                    f"{decision.confidence_score:.3f}",
                    "",    # violations: stored inside reasoning text
                    outcome.name,
                    decision.ok,
                    via_mcp,
                ])
        except Exception:
            pass    # Never crash the EnergyPlus simulation due to logging failure

    def _ensure_log_file(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
        except Exception:
            pass
