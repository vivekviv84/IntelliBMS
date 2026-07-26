"""
economizer.py — Commercial BAS-Grade Economizer / Free Cooling Optimization Agent.

Models commercial Building Automation System (BAS) economizer logic (Honeywell, Siemens,
Schneider Electric, Johnson Controls). Intelligently evaluates outdoor drybulb suitability,
thermal advantage (Indoor - Outdoor temp), cooling demand hysteresis, and occupancy signals
to recommend free cooling whenever ambient conditions permit avoiding active compressor cooling.

Author: EcoLoop AI System
"""

import sys
import os
from dataclasses import dataclass, asdict

# Import central configuration
_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_PLUGIN_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import (
    ESTIMATED_COOLING_POWER_KW,
    DECISION_INTERVAL_HOURS,
    ECONOMIZER_TEMP_ADVANTAGE_MIN_C,
    ECONOMIZER_RUNTIME_FACTOR,
)


@dataclass
class EconomizerRecommendation:
    """
    Structured output from EconomizerModule evaluation.
    Encapsulates real-time free cooling recommendation and telemetry metrics.
    """
    recommended_mode:              str     # "FREE_COOLING" | "NO_ACTION"
    economizer_active:             bool    # True if FREE_COOLING is recommended
    estimated_energy_saved_kwh:    float   # Estimated kWh saved by avoiding compressor
    estimated_runtime_saved_hours: float   # Avoided compressor runtime in hours
    temperature_advantage:         float   # Zone temp - Outdoor temp (°C)
    confidence:                    float   # Deterministic confidence (0.0 to 1.0)
    reason:                        str     # Commercial BAS engineering explanation

    def to_dict(self) -> dict:
        return asdict(self)


class EconomizerModule:
    """
    Specialist agent evaluating outdoor free-cooling potential.
    Provides unbiased advisory recommendations to CoordinatorAgent.
    """

    def evaluate(self, obs) -> EconomizerRecommendation:
        """
        Evaluate real-time sensor observation using commercial BAS 6-step logic.
        Accepts ObservationContext or any object exposing zone_temp, cooling_sp,
        outdoor_temp, occupancy, and hvac_mode.
        """
        zone_temp    = float(getattr(obs, "zone_temp", 24.0))
        cooling_sp   = float(getattr(obs, "cooling_sp", 24.0))
        outdoor_temp = float(getattr(obs, "outdoor_temp", 25.0))
        occupancy    = float(getattr(obs, "occupancy", 1.0))
        hvac_mode    = str(getattr(obs, "hvac_mode", "auto")).lower()

        # Step 1 — Cooling Demand Hysteresis Check
        # Cooling is needed if zone temperature exceeds cooling setpoint + 0.3°C deadband
        cooling_demand_offset = 0.3
        cooling_needed = zone_temp > (cooling_sp + cooling_demand_offset)

        if not cooling_needed:
            return EconomizerRecommendation(
                recommended_mode="NO_ACTION",
                economizer_active=False,
                estimated_energy_saved_kwh=0.0,
                estimated_runtime_saved_hours=0.0,
                temperature_advantage=round(zone_temp - outdoor_temp, 2),
                confidence=0.90,
                reason="No active cooling demand exists (zone_temp is within comfort setpoint + hysteresis)."
            )

        # Step 2 — Temperature Advantage Check
        # Require at least ECONOMIZER_TEMP_ADVANTAGE_MIN_C (1.5°C) before free cooling is considered
        temp_advantage = zone_temp - outdoor_temp
        if temp_advantage < ECONOMIZER_TEMP_ADVANTAGE_MIN_C:
            return EconomizerRecommendation(
                recommended_mode="NO_ACTION",
                economizer_active=False,
                estimated_energy_saved_kwh=0.0,
                estimated_runtime_saved_hours=0.0,
                temperature_advantage=round(temp_advantage, 2),
                confidence=0.85,
                reason=(f"Insufficient temperature advantage ({temp_advantage:.2f}°C vs "
                        f"required {ECONOMIZER_TEMP_ADVANTAGE_MIN_C:.1f}°C min threshold).")
            )

        # Step 3 — Outdoor Air Suitability Check
        # Outdoor drybulb temperature must be <= Cooling Setpoint + 0.5°C tolerance
        outdoor_tolerance = 0.5
        outdoor_suitable = outdoor_temp <= (cooling_sp + outdoor_tolerance)
        if not outdoor_suitable:
            return EconomizerRecommendation(
                recommended_mode="NO_ACTION",
                economizer_active=False,
                estimated_energy_saved_kwh=0.0,
                estimated_runtime_saved_hours=0.0,
                temperature_advantage=round(temp_advantage, 2),
                confidence=0.85,
                reason=(f"Outdoor drybulb ({outdoor_temp:.2f}°C) exceeds economizer limit "
                        f"({cooling_sp + outdoor_tolerance:.2f}°C).")
            )

        # Step 4 — Occupancy Check
        if occupancy <= 0.0:
            return EconomizerRecommendation(
                recommended_mode="NO_ACTION",
                economizer_active=False,
                estimated_energy_saved_kwh=0.0,
                estimated_runtime_saved_hours=0.0,
                temperature_advantage=round(temp_advantage, 2),
                confidence=0.95,
                reason="Zone is unoccupied; economizer ventilation disabled."
            )

        # Step 5 — HVAC Mode Check
        if hvac_mode == "off":
            return EconomizerRecommendation(
                recommended_mode="NO_ACTION",
                economizer_active=False,
                estimated_energy_saved_kwh=0.0,
                estimated_runtime_saved_hours=0.0,
                temperature_advantage=round(temp_advantage, 2),
                confidence=1.00,
                reason="HVAC system is turned off."
            )

        # Step 6 — Final Decision: FREE_COOLING
        runtime_saved = DECISION_INTERVAL_HOURS
        energy_saved_kwh = runtime_saved * ESTIMATED_COOLING_POWER_KW * ECONOMIZER_RUNTIME_FACTOR

        # Deterministic Engineering Confidence Computation
        base_confidence = 0.35
        score_temp_adv  = min(0.35, max(0.0, (temp_advantage - ECONOMIZER_TEMP_ADVANTAGE_MIN_C) * 0.10))
        score_demand    = min(0.20, max(0.0, (zone_temp - (cooling_sp + cooling_demand_offset)) * 0.20))
        score_occ       = 0.10 if occupancy > 0 else 0.0

        raw_confidence  = base_confidence + score_temp_adv + score_demand + score_occ
        confidence      = round(max(0.0, min(1.0, raw_confidence)), 2)

        reason_str = (
            f"Favorable outdoor air ({outdoor_temp:.2f}°C) provides +{temp_advantage:.2f}°C "
            f"temperature advantage over zone air ({zone_temp:.2f}°C). "
            f"Free cooling recommended to avoid {runtime_saved:.1f}h compressor runtime "
            f"({energy_saved_kwh:.3f} kWh estimated saving)."
        )

        return EconomizerRecommendation(
            recommended_mode="FREE_COOLING",
            economizer_active=True,
            estimated_energy_saved_kwh=round(energy_saved_kwh, 3),
            estimated_runtime_saved_hours=round(runtime_saved, 2),
            temperature_advantage=round(temp_advantage, 2),
            confidence=confidence,
            reason=reason_str
        )
