"""
demand_response.py — Commercial BAS Peak Tariff / Demand Response Advisory Agent.

Evaluates Time-of-Use (ToU) commercial tariffs and recommends peak load-shedding
biases to the CoordinatorAgent and PlanningAgent.

Author: EcoLoop AI Control Architecture
"""

from dataclasses import dataclass
from typing import Optional

try:
    from config import (
        ELECTRICITY_TARIFF_INR_KWH,
        TOU_MULTIPLIERS,
        PEAK_WINDOWS,
        DR_MAX_COMFORT_DEV_C,
        ESTIMATED_COOLING_POWER_KW,
        DECISION_INTERVAL_HOURS,
    )
except ImportError:
    ELECTRICITY_TARIFF_INR_KWH = 10.0
    TOU_MULTIPLIERS = {"OFF_PEAK": 0.80, "NORMAL": 1.00, "PEAK": 1.35}
    PEAK_WINDOWS = [(18, 22)]
    DR_MAX_COMFORT_DEV_C = 0.8
    ESTIMATED_COOLING_POWER_KW = 1.5
    DECISION_INTERVAL_HOURS = 1.0


@dataclass
class DemandResponseRecommendation:
    """Dataclass encapsulating Demand Response peak tariff analysis and recommendations."""
    is_peak_window:             bool
    current_tariff_period:      str     # "OFF_PEAK" | "NORMAL" | "PEAK"
    current_tariff_inr_kwh:     float
    recommended_action_bias:    str     # "eco" | "none"
    dr_recommended:             bool
    comfort_deviation_c:        float
    estimated_cost_saved_inr:   float
    reason:                     str
    confidence:                 float


class DemandResponseModule:
    """
    Demand Response Advisory Agent.
    Evaluates ToU tariff windows and provides load-shedding recommendations.
    """

    def __init__(self):
        pass

    def evaluate(self, obs) -> DemandResponseRecommendation:
        """
        Evaluate current observation context against ToU commercial tariff schedule.
        """
        hour = getattr(obs, "time_of_day_hour", 12)
        zone_temp = getattr(obs, "zone_temp", 24.0)
        cooling_sp = getattr(obs, "cooling_sp", 24.0)

        # 1. Determine peak window status
        is_peak_window = any(start <= hour < end for start, end in PEAK_WINDOWS)

        # 2. Determine ToU tariff period
        if is_peak_window:
            tariff_period = "PEAK"
        elif hour >= 22 or hour < 6:
            tariff_period = "OFF_PEAK"
        else:
            tariff_period = "NORMAL"

        mult = TOU_MULTIPLIERS.get(tariff_period, 1.00)
        current_tariff_inr = round(ELECTRICITY_TARIFF_INR_KWH * mult, 2)

        # 3. Compute active cooling comfort deviation
        comfort_dev = max(0.0, zone_temp - cooling_sp) if zone_temp > cooling_sp else 0.0

        # 4. Apply Demand Response logic
        if is_peak_window:
            if comfort_dev <= DR_MAX_COMFORT_DEV_C:
                bias = "eco"
                dr_rec = True
                # Cost difference between peak compressor run vs eco load reduction
                cost_saved = round(
                    DECISION_INTERVAL_HOURS * ESTIMATED_COOLING_POWER_KW * 0.5 * (current_tariff_inr), 2
                )
                reason = (
                    f"Peak ToU tariff window active ({hour:02d}:00, ₹{current_tariff_inr:.2f}/kWh, +35% surcharge). "
                    f"Zone comfort deviation ({comfort_dev:.2f}°C) is within safety threshold ({DR_MAX_COMFORT_DEV_C:.1f}°C). "
                    f"Biasing planner toward 'eco' to shed peak load and avoid high tariff costs."
                )
                confidence = 0.95
            else:
                bias = "none"
                dr_rec = False
                cost_saved = 0.0
                reason = (
                    f"Peak ToU tariff active ({hour:02d}:00, ₹{current_tariff_inr:.2f}/kWh), but zone comfort deviation "
                    f"({comfort_dev:.2f}°C) exceeds safety limit ({DR_MAX_COMFORT_DEV_C:.1f}°C). Comfort safety override active."
                )
                confidence = 0.85
        else:
            bias = "none"
            dr_rec = False
            cost_saved = 0.0
            reason = (
                f"Standard ToU tariff period ({tariff_period}, ₹{current_tariff_inr:.2f}/kWh). No peak demand response required."
            )
            confidence = 1.00

        return DemandResponseRecommendation(
            is_peak_window=is_peak_window,
            current_tariff_period=tariff_period,
            current_tariff_inr_kwh=current_tariff_inr,
            recommended_action_bias=bias,
            dr_recommended=dr_rec,
            comfort_deviation_c=round(comfort_dev, 2),
            estimated_cost_saved_inr=cost_saved,
            reason=reason,
            confidence=confidence,
        )
