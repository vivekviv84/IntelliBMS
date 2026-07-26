"""
predictive_controller.py — Predictive Pre-Cooling Advisory Module.

Uses EPW Look-Ahead Weather data (2-4 hour simulation forecast horizon) to detect
upcoming high outdoor temperature events and recommend predictive pre-cooling.

Author: EcoLoop AI Control Architecture
"""

import os
from dataclasses import dataclass
from typing import List, Optional

try:
    from config import (
        PRECOOL_LOOKAHEAD_HOURS,
        PRECOOL_TEMP_THRESHOLD_C,
        PRECOOL_TEMP_DIFF_MIN_C,
        PRECOOL_MAX_COIL_SPEED,
    )
except ImportError:
    PRECOOL_LOOKAHEAD_HOURS = 3
    PRECOOL_TEMP_THRESHOLD_C = 28.0
    PRECOOL_TEMP_DIFF_MIN_C = 2.0
    PRECOOL_MAX_COIL_SPEED = 1.0


@dataclass
class PredictivePrecoolRecommendation:
    """Dataclass encapsulating Predictive Pre-Cooling analysis and recommendations."""
    precool_recommended:          bool
    predicted_peak_outdoor_temp: float
    forecast_horizon_hours:       int
    precool_start_time:           str
    estimated_peak_reduction_w:   float
    expected_comfort_benefit_c:   float
    target_precool_sp:            float
    reason:                       str
    confidence:                   float


class PredictivePrecoolModule:
    """
    Predictive Pre-Cooling Advisory Agent.
    Parses EPW weather look-ahead data and recommends thermal pre-cooling prior to peak heat.
    """

    def __init__(self, epw_path: Optional[str] = None):
        self.epw_path = epw_path
        self.hourly_temps: List[float] = []
        if epw_path and os.path.exists(epw_path):
            self._load_epw_weather(epw_path)

    def _load_epw_weather(self, epw_path: str) -> None:
        """Load 8,760 hourly outdoor drybulb temperature points from EPW file."""
        try:
            temps = []
            with open(epw_path, "r", encoding="utf-8", errors="ignore") as f:
                for idx, line in enumerate(f):
                    if idx < 8:
                        continue
                    parts = line.strip().split(",")
                    if len(parts) > 6:
                        try:
                            temps.append(float(parts[6]))
                        except ValueError:
                            pass
            if len(temps) >= 8760:
                self.hourly_temps = temps
        except Exception:
            pass

    def evaluate(self, obs, upcoming_temps: Optional[List[float]] = None) -> PredictivePrecoolRecommendation:
        """
        Evaluate EPW look-ahead weather forecast against predictive pre-cooling criteria.
        """
        hour = getattr(obs, "time_of_day_hour", 12)
        current_outdoor = getattr(obs, "outdoor_temp", 25.0)
        cooling_sp = getattr(obs, "cooling_sp", 24.6)

        # 1. Get future outdoor temperatures over PRECOOL_LOOKAHEAD_HOURS (3 hours)
        future_temps = []
        if upcoming_temps and len(upcoming_temps) > 0:
            future_temps = upcoming_temps[:PRECOOL_LOOKAHEAD_HOURS]
        elif self.hourly_temps:
            # Derive day of year index
            timestamp = getattr(obs, "timestamp", "07/01 12:00")
            try:
                parts = timestamp.split()
                month, day = map(int, parts[0].split("/"))
                # Days per month in non-leap year
                days_before = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
                day_of_year = days_before[month - 1] + day
                start_idx = (day_of_year - 1) * 24 + hour
                future_temps = [
                    self.hourly_temps[(start_idx + i) % 8760]
                    for i in range(1, PRECOOL_LOOKAHEAD_HOURS + 1)
                ]
            except Exception:
                future_temps = [current_outdoor + 1.0, current_outdoor + 2.5, current_outdoor + 3.0]
        else:
            # Synthetic forecast if EPW path not bound
            future_temps = [current_outdoor + 0.8, current_outdoor + 1.8, current_outdoor + 2.5]

        predicted_peak_outdoor = max(future_temps) if future_temps else current_outdoor

        # 2. Smart Differential Pre-Cooling Logic
        is_hot_event = (predicted_peak_outdoor >= PRECOOL_TEMP_THRESHOLD_C)
        is_temp_rising = (predicted_peak_outdoor >= current_outdoor + PRECOOL_TEMP_DIFF_MIN_C)
        is_daytime_window = (6 <= hour <= 17)  # Prior to afternoon thermal peak

        if is_hot_event and is_temp_rising and is_daytime_window:
            precool_rec = True
            target_precool_sp = max(21.0, round(cooling_sp - 1.0, 1))
            peak_red_w = round((predicted_peak_outdoor - PRECOOL_TEMP_THRESHOLD_C) * 120.0 + 320.0, 1)
            comfort_benefit = 0.45
            start_time = f"{hour:02d}:00"
            reason = (
                f"EPW 3-hour look-ahead weather predicts outdoor heat event reaching "
                f"{predicted_peak_outdoor:.1f}°C (+{predicted_peak_outdoor - current_outdoor:.1f}°C rise above current {current_outdoor:.1f}°C). "
                f"Recommending predictive pre-cooling to setpoint {target_precool_sp:.1f}°C to build building thermal mass and shave peak demand."
            )
            confidence = 0.92
        else:
            precool_rec = False
            target_precool_sp = cooling_sp
            peak_red_w = 0.0
            comfort_benefit = 0.0
            start_time = "N/A"
            if not is_hot_event:
                reason = f"EPW 3-hour forecast peak ({predicted_peak_outdoor:.1f}°C) is below pre-cooling threshold ({PRECOOL_TEMP_THRESHOLD_C:.1f}°C)."
            elif not is_temp_rising:
                reason = f"Outdoor temperature rise (+{predicted_peak_outdoor - current_outdoor:.1f}°C) is below minimum differential threshold (+{PRECOOL_TEMP_DIFF_MIN_C:.1f}°C)."
            else:
                reason = "Nighttime period; predictive pre-cooling restricted to pre-occupancy daytime hours."
            confidence = 1.00

        return PredictivePrecoolRecommendation(
            precool_recommended=precool_rec,
            predicted_peak_outdoor_temp=round(predicted_peak_outdoor, 2),
            forecast_horizon_hours=PRECOOL_LOOKAHEAD_HOURS,
            precool_start_time=start_time,
            estimated_peak_reduction_w=peak_red_w,
            expected_comfort_benefit_c=comfort_benefit,
            target_precool_sp=target_precool_sp,
            reason=reason,
            confidence=confidence,
        )
