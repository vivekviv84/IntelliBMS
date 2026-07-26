"""
test_enterprise_advisory_isolation.py — Standalone Unit Tests for Demand Response and Predictive Pre-Cooling.
Tests all ToU tariff rules and EPW look-ahead differential precooling logic.
"""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "plugins"))

from demand_response import DemandResponseModule
from predictive_controller import PredictivePrecoolModule
from reasoning_agent import ObservationContext


def run_isolation_tests():
    dr_module = DemandResponseModule()
    epw_p = os.path.join(os.path.dirname(__file__), "..", "weather", "IND_KA_Bengaluru.432950_ISHRAE2014.epw")
    pc_module = PredictivePrecoolModule(epw_p)

    print("==========================================================")
    print("  ECOLOOP ENTERPRISE ADVISORY MODULE ISOLATION UNIT TESTS ")
    print("==========================================================")

    # ─────────────────────────────────────────────────────────────────────────
    # Test 1: Demand Response — Peak Window with Low Comfort Deviation
    # Hour = 19 (Peak), ZoneTemp = 24.8°C, CoolingSP = 24.6°C (Dev = 0.2°C <= 0.8°C limit)
    # Expected: bias="eco", dr_recommended=True
    # ─────────────────────────────────────────────────────────────────────────
    obs1 = ObservationContext(
        zone_temp=24.8, heating_sp=21.0, cooling_sp=24.6, outdoor_temp=28.0,
        timestamp="07/01 19:00", time_of_day_hour=19, occupancy=1.0, hvac_mode="auto"
    )
    dr_rec1 = dr_module.evaluate(obs1)
    print("\n[TEST 1: Demand Response — Peak Window Load Shedding]")
    print(f"  Tariff Period : {dr_rec1.current_tariff_period} (₹{dr_rec1.current_tariff_inr_kwh:.2f}/kWh)")
    print(f"  Peak Window   : {dr_rec1.is_peak_window}")
    print(f"  Action Bias   : {dr_rec1.recommended_action_bias}")
    print(f"  DR Recommended: {dr_rec1.dr_recommended}")
    print(f"  Reason        : {dr_rec1.reason}")
    assert dr_rec1.is_peak_window is True, "Test 1 peak window failed!"
    assert dr_rec1.recommended_action_bias == "eco", "Test 1 action bias failed!"
    assert dr_rec1.dr_recommended is True, "Test 1 dr_recommended failed!"
    print("  STATUS        : ✅ PASSED")

    # ─────────────────────────────────────────────────────────────────────────
    # Test 2: Demand Response — Peak Window Comfort Safety Override
    # Hour = 19 (Peak), ZoneTemp = 25.6°C, CoolingSP = 24.6°C (Dev = 1.0°C > 0.8°C limit)
    # Expected: bias="none", dr_recommended=False (comfort safety priority)
    # ─────────────────────────────────────────────────────────────────────────
    obs2 = ObservationContext(
        zone_temp=25.6, heating_sp=21.0, cooling_sp=24.6, outdoor_temp=28.0,
        timestamp="07/01 19:00", time_of_day_hour=19, occupancy=1.0, hvac_mode="auto"
    )
    dr_rec2 = dr_module.evaluate(obs2)
    print("\n[TEST 2: Demand Response — Comfort Safety Override]")
    print(f"  Tariff Period : {dr_rec2.current_tariff_period}")
    print(f"  Comfort Dev   : {dr_rec2.comfort_deviation_c:.2f}°C")
    print(f"  Action Bias   : {dr_rec2.recommended_action_bias}")
    print(f"  DR Recommended: {dr_rec2.dr_recommended}")
    print(f"  Reason        : {dr_rec2.reason}")
    assert dr_rec2.recommended_action_bias == "none", "Test 2 safety override bias failed!"
    assert dr_rec2.dr_recommended is False, "Test 2 dr_recommended failed!"
    print("  STATUS        : ✅ PASSED")

    # ─────────────────────────────────────────────────────────────────────────
    # Test 3: Predictive Pre-Cooling — EPW Look-Ahead Heat Event
    # Hour = 10 (Daytime), CurrentOutdoor = 25.0°C, Future Peak = 29.5°C (+4.5°C rise >= 2°C & >= 28°C)
    # Expected: precool_recommended=True, target_precool_sp=23.6°C
    # ─────────────────────────────────────────────────────────────────────────
    obs3 = ObservationContext(
        zone_temp=24.6, heating_sp=21.0, cooling_sp=24.6, outdoor_temp=25.0,
        timestamp="07/01 10:00", time_of_day_hour=10, occupancy=1.0, hvac_mode="auto"
    )
    pc_rec3 = pc_module.evaluate(obs3, upcoming_temps=[26.5, 28.2, 29.5])
    print("\n[TEST 3: Predictive Pre-Cooling — Heat Event Look-Ahead]")
    print(f"  Peak Outdoor  : {pc_rec3.predicted_peak_outdoor_temp:.1f}°C")
    print(f"  Precool Rec   : {pc_rec3.precool_recommended}")
    print(f"  Target SP     : {pc_rec3.target_precool_sp:.1f}°C")
    print(f"  Est Peak Reduc: {pc_rec3.estimated_peak_reduction_w:.1f} W")
    print(f"  Reason        : {pc_rec3.reason}")
    assert pc_rec3.precool_recommended is True, "Test 3 precool failed!"
    assert pc_rec3.target_precool_sp == 23.6, "Test 3 target SP failed!"
    print("  STATUS        : ✅ PASSED")

    # ─────────────────────────────────────────────────────────────────────────
    # Test 4: Predictive Pre-Cooling — Mild Weather Forecast
    # Hour = 10 (Daytime), CurrentOutdoor = 22.0°C, Future Peak = 24.5°C (< 28°C threshold)
    # Expected: precool_recommended=False
    # ─────────────────────────────────────────────────────────────────────────
    obs4 = ObservationContext(
        zone_temp=24.6, heating_sp=21.0, cooling_sp=24.6, outdoor_temp=22.0,
        timestamp="07/01 10:00", time_of_day_hour=10, occupancy=1.0, hvac_mode="auto"
    )
    pc_rec4 = pc_module.evaluate(obs4, upcoming_temps=[23.0, 23.8, 24.5])
    print("\n[TEST 4: Predictive Pre-Cooling — Mild Weather]")
    print(f"  Peak Outdoor  : {pc_rec4.predicted_peak_outdoor_temp:.1f}°C")
    print(f"  Precool Rec   : {pc_rec4.precool_recommended}")
    print(f"  Reason        : {pc_rec4.reason}")
    assert pc_rec4.precool_recommended is False, "Test 4 precool failed!"
    print("  STATUS        : ✅ PASSED")

    print("\n==========================================================")
    print("  ALL ENTERPRISE ADVISORY MODULE TESTS PASSED 100% CLEANLY!")
    print("==========================================================")


if __name__ == "__main__":
    run_isolation_tests()
