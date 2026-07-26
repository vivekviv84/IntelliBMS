"""
confidence_engine.py — Mathematically Grounded Confidence Scoring Engine.

Produces a composite confidence score [0.0, 1.0] from five independent,
observable signals. No randomness. Every value is derived from real data
in the ShortTermMemory or from the current ObservationContext.

Signal Weights (sum to 1.0):
    historical_success   30%  — recent action success rate from memory
    sensor_consistency   20%  — inverse of recent zone-temp variance
    weather_certainty    15%  — stability of outdoor-indoor thermal delta
    comfort_prediction   20%  — accuracy of previous comfort expectation
    simulation_stability 15%  — absence of oscillations in action history

Author: EcoLoop AI System
"""

import math
import statistics
from dataclasses import dataclass, asdict
from typing import List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Weights (must sum to 1.0)
# ─────────────────────────────────────────────────────────────────────────────

WEIGHTS = {
    "historical_success":   0.30,
    "sensor_consistency":   0.20,
    "weather_certainty":    0.15,
    "comfort_prediction":   0.20,
    "simulation_stability": 0.15,
}

# Normalisation constants (empirically tuned to realistic HVAC data ranges)
_TEMP_VARIANCE_MAX       = 4.0   # °C²  — variance above this → sensor_consistency=0
_DELTA_VARIANCE_MAX      = 9.0   # (°C)² — outdoor-indoor delta variance above this → 0
_OSCILLATION_FLIP_MAX    = 5     # flips in last 6 actions → simulation_stability=0
_COMFORT_MATCH_THRESHOLD = 0.10  # °C — improvement within this → "correct prediction"


# ─────────────────────────────────────────────────────────────────────────────
# Output Dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConfidenceBreakdown:
    """
    Per-signal confidence scores and the weighted total.
    All values are in [0.0, 1.0].
    """
    historical_success:   float   # recent cycles that had no violations
    sensor_consistency:   float   # 1 - normalised(temp_variance)
    weather_certainty:    float   # 1 - normalised(delta_variance)
    comfort_prediction:   float   # accuracy of previous comfort expectation
    simulation_stability: float   # 1 - normalised(oscillation_flips)
    total:                float   # weighted composite score

    def to_dict(self) -> dict:
        return asdict(self)

    def to_prompt_block(self) -> str:
        return (
            f"=== CONFIDENCE BREAKDOWN (pre-computed) ===\n"
            f"  Historical Success      : {self.historical_success:.3f}  (weight=0.30)\n"
            f"  Sensor Consistency      : {self.sensor_consistency:.3f}  (weight=0.20)\n"
            f"  Comfort Prediction Acc. : {self.comfort_prediction:.3f}  (weight=0.20)\n"
            f"  Weather Certainty       : {self.weather_certainty:.3f}  (weight=0.15)\n"
            f"  Simulation Stability    : {self.simulation_stability:.3f}  (weight=0.15)\n"
            f"  COMPOSITE CONFIDENCE    : {self.total:.3f}\n"
            f"  Your confidence_score field MUST be within ±0.10 of {self.total:.3f}."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class ConfidenceEngine:
    """
    Computes a ConfidenceBreakdown from ShortTermMemory + ObservationContext.
    Stateless — call compute() directly.
    """

    @staticmethod
    def compute(memory, obs) -> ConfidenceBreakdown:
        """
        Parameters
        ----------
        memory : ShortTermMemory
        obs    : ObservationContext

        Returns
        -------
        ConfidenceBreakdown with all signals and composite total.
        """
        recent = memory.get_recent(6)

        s1 = ConfidenceEngine._historical_success(memory)
        s2 = ConfidenceEngine._sensor_consistency(recent)
        s3 = ConfidenceEngine._weather_certainty(recent)
        s4 = ConfidenceEngine._comfort_prediction(recent)
        s5 = ConfidenceEngine._simulation_stability(memory)

        total = (
            WEIGHTS["historical_success"]   * s1 +
            WEIGHTS["sensor_consistency"]   * s2 +
            WEIGHTS["weather_certainty"]    * s3 +
            WEIGHTS["comfort_prediction"]   * s4 +
            WEIGHTS["simulation_stability"] * s5
        )

        return ConfidenceBreakdown(
            historical_success   = round(s1, 4),
            sensor_consistency   = round(s2, 4),
            weather_certainty    = round(s3, 4),
            comfort_prediction   = round(s4, 4),
            simulation_stability = round(s5, 4),
            total                = round(max(0.0, min(1.0, total)), 4),
        )

    # ── Signal Implementations ────────────────────────────────────────────────

    @staticmethod
    def _historical_success(memory) -> float:
        """
        Fraction of recent cycles that succeeded (no violations triggered).
        n=6 sliding window. Prior: 0.60 (moderately uncertain without data).
        """
        if len(memory) == 0:
            return 0.60   # Uninformed prior
        rate = memory.recent_success_rate(6)
        # Shrinkage toward 0.60 when we have few samples
        n = min(len(memory), 6)
        shrunk = (n * rate + 2 * 0.60) / (n + 2)  # Bayesian smoothing, pseudocount=2
        return round(shrunk, 4)

    @staticmethod
    def _sensor_consistency(recent) -> float:
        """
        Inverse of recent zone-temp variance.
        High variance → sensors inconsistent or zone oscillating → lower confidence.
        """
        if len(recent) < 2:
            return 0.50
        temps = [r.zone_temp for r in recent]
        try:
            var = statistics.variance(temps)
        except statistics.StatisticsError:
            return 0.50
        # Logistic-like decay: var=0 → 1.0, var=4 → 0.0
        score = max(0.0, 1.0 - var / _TEMP_VARIANCE_MAX)
        return round(score, 4)

    @staticmethod
    def _weather_certainty(recent) -> float:
        """
        Stability of the outdoor-indoor thermal delta over recent cycles.
        Stable delta → predictable thermal load → higher confidence.
        """
        if len(recent) < 2:
            return 0.50
        deltas = [r.outdoor_temp - r.zone_temp for r in recent]
        try:
            var = statistics.variance(deltas)
        except statistics.StatisticsError:
            return 0.50
        score = max(0.0, 1.0 - var / _DELTA_VARIANCE_MAX)
        return round(score, 4)

    @staticmethod
    def _comfort_prediction(recent) -> float:
        """
        Accuracy of the previous cycle's comfort expectation.
        Positive signal if: last action was successful AND comfort deviation
        did not increase significantly compared to the cycle before it.
        """
        if len(recent) < 2:
            return 0.50

        prev = recent[-2]
        curr = recent[-1]

        # Was the previous prediction correct?
        comfort_improved = (curr.comfort_deviation <= prev.comfort_deviation + _COMFORT_MATCH_THRESHOLD)
        prev_succeeded   = prev.success

        if prev_succeeded and comfort_improved:
            base = 0.85   # Strong prediction accuracy
        elif prev_succeeded and not comfort_improved:
            base = 0.55   # Action was valid but comfort didn't improve as expected
        elif not prev_succeeded and comfort_improved:
            base = 0.60   # Violation triggered but comfort recovered anyway
        else:
            base = 0.35   # Both failed

        # Penalise if violations cascaded across multiple cycles
        consecutive_failures = sum(1 for r in recent[-3:] if not r.success)
        penalty = consecutive_failures * 0.08
        return round(max(0.10, base - penalty), 4)

    @staticmethod
    def _simulation_stability(memory) -> float:
        """
        Absence of oscillations in recent action history.
        Counts action-direction flips in the last 6 steps.
        flip=0 → 1.0, flip=5 → 0.0 (linear decay).
        """
        actions = memory.action_sequence()
        if len(actions) < 2:
            return 0.70   # Assume stable when insufficient history
        window = actions[-6:]
        flips = sum(1 for i in range(1, len(window)) if window[i] != window[i - 1])
        score = max(0.0, 1.0 - flips / _OSCILLATION_FLIP_MAX)
        return round(score, 4)
