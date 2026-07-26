"""
test_economizer_isolation.py — Standalone Isolation Unit Tests for Economizer Module.
Tests 5 commercial BAS engineering scenarios against economizer logic.
"""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure plugins directory is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "plugins"))

from economizer import EconomizerModule, EconomizerRecommendation
from reasoning_agent import ObservationContext


def run_isolation_tests():
    module = EconomizerModule()
    print("==========================================================")
    print("      ECOLOOP ECONOMIZER ISOLATION UNIT TESTS             ")
    print("==========================================================")

    # ─────────────────────────────────────────────────────────────────────────
    # Test 1: Ideal Free Cooling Scenario
    # Indoor = 26°C, Outdoor = 20°C, Cooling SP = 24°C, Occupied (1.0)
    # Expected: FREE_COOLING
    # ─────────────────────────────────────────────────────────────────────────
    obs1 = ObservationContext(
        zone_temp=26.0, heating_sp=21.0, cooling_sp=24.0, outdoor_temp=20.0,
        timestamp="07/01 14:00", time_of_day_hour=14, occupancy=1.0, hvac_mode="auto"
    )
    rec1 = module.evaluate(obs1)
    print("\n[TEST 1: Ideal Free Cooling]")
    print(f"  Inputs      : Indoor=26°C, Outdoor=20°C, CoolingSP=24°C, Occupied=1.0")
    print(f"  Recommended : {rec1.recommended_mode}")
    print(f"  Active      : {rec1.economizer_active}")
    print(f"  Temp Adv.   : +{rec1.temperature_advantage:.2f}°C")
    print(f"  Confidence  : {rec1.confidence:.2f}")
    print(f"  Est. Saved  : {rec1.estimated_energy_saved_kwh:.3f} kWh")
    print(f"  Reason      : {rec1.reason}")
    assert rec1.recommended_mode == "FREE_COOLING", "Test 1 failed!"
    assert rec1.economizer_active is True, "Test 1 active failed!"
    print("  STATUS      : ✅ PASSED")

    # ─────────────────────────────────────────────────────────────────────────
    # Test 2: Hot Outdoor Air Scenario
    # Indoor = 26°C, Outdoor = 32°C, Cooling SP = 24°C, Occupied (1.0)
    # Expected: NO_ACTION
    # ─────────────────────────────────────────────────────────────────────────
    obs2 = ObservationContext(
        zone_temp=26.0, heating_sp=21.0, cooling_sp=24.0, outdoor_temp=32.0,
        timestamp="07/01 14:00", time_of_day_hour=14, occupancy=1.0, hvac_mode="auto"
    )
    rec2 = module.evaluate(obs2)
    print("\n[TEST 2: Hot Outdoor Air]")
    print(f"  Inputs      : Indoor=26°C, Outdoor=32°C, CoolingSP=24°C, Occupied=1.0")
    print(f"  Recommended : {rec2.recommended_mode}")
    print(f"  Active      : {rec2.economizer_active}")
    print(f"  Temp Adv.   : {rec2.temperature_advantage:.2f}°C")
    print(f"  Reason      : {rec2.reason}")
    assert rec2.recommended_mode == "NO_ACTION", "Test 2 failed!"
    assert rec2.economizer_active is False, "Test 2 active failed!"
    print("  STATUS      : ✅ PASSED")

    # ─────────────────────────────────────────────────────────────────────────
    # Test 3: Unoccupied Scenario
    # Indoor = 26°C, Outdoor = 20°C, Cooling SP = 24°C, Unoccupied (0.0)
    # Expected: NO_ACTION
    # ─────────────────────────────────────────────────────────────────────────
    obs3 = ObservationContext(
        zone_temp=26.0, heating_sp=21.0, cooling_sp=24.0, outdoor_temp=20.0,
        timestamp="07/01 23:00", time_of_day_hour=23, occupancy=0.0, hvac_mode="auto"
    )
    rec3 = module.evaluate(obs3)
    print("\n[TEST 3: Unoccupied Zone]")
    print(f"  Inputs      : Indoor=26°C, Outdoor=20°C, CoolingSP=24°C, Occupied=0.0")
    print(f"  Recommended : {rec3.recommended_mode}")
    print(f"  Active      : {rec3.economizer_active}")
    print(f"  Reason      : {rec3.reason}")
    assert rec3.recommended_mode == "NO_ACTION", "Test 3 failed!"
    assert rec3.economizer_active is False, "Test 3 active failed!"
    print("  STATUS      : ✅ PASSED")

    # ─────────────────────────────────────────────────────────────────────────
    # Test 4: Borderline Temperature Advantage (Hysteresis Check)
    # Indoor = 25°C, Outdoor = 24.9°C (Advantage = 0.1°C < 1.5°C min)
    # Expected: NO_ACTION
    # ─────────────────────────────────────────────────────────────────────────
    obs4 = ObservationContext(
        zone_temp=25.0, heating_sp=21.0, cooling_sp=24.0, outdoor_temp=24.9,
        timestamp="07/01 14:00", time_of_day_hour=14, occupancy=1.0, hvac_mode="auto"
    )
    rec4 = module.evaluate(obs4)
    print("\n[TEST 4: Borderline Hysteresis]")
    print(f"  Inputs      : Indoor=25°C, Outdoor=24.9°C, CoolingSP=24°C, Occupied=1.0")
    print(f"  Recommended : {rec4.recommended_mode}")
    print(f"  Active      : {rec4.economizer_active}")
    print(f"  Temp Adv.   : {rec4.temperature_advantage:.2f}°C")
    print(f"  Reason      : {rec4.reason}")
    assert rec4.recommended_mode == "NO_ACTION", "Test 4 failed!"
    assert rec4.economizer_active is False, "Test 4 active failed!"
    print("  STATUS      : ✅ PASSED")

    # ─────────────────────────────────────────────────────────────────────────
    # Test 5: Already Comfortable (No Cooling Demand)
    # Indoor = 23.9°C <= 24.3°C, Outdoor = 20°C
    # Expected: NO_ACTION
    # ─────────────────────────────────────────────────────────────────────────
    obs5 = ObservationContext(
        zone_temp=23.9, heating_sp=21.0, cooling_sp=24.0, outdoor_temp=20.0,
        timestamp="07/01 14:00", time_of_day_hour=14, occupancy=1.0, hvac_mode="auto"
    )
    rec5 = module.evaluate(obs5)
    print("\n[TEST 5: Already Comfortable (No Demand)]")
    print(f"  Inputs      : Indoor=23.9°C, Outdoor=20°C, CoolingSP=24°C, Occupied=1.0")
    print(f"  Recommended : {rec5.recommended_mode}")
    print(f"  Active      : {rec5.economizer_active}")
    print(f"  Reason      : {rec5.reason}")
    assert rec5.recommended_mode == "NO_ACTION", "Test 5 failed!"
    assert rec5.economizer_active is False, "Test 5 active failed!"
    print("  STATUS      : ✅ PASSED")

    print("\n==========================================================")
    print("    ALL 5 ECOLOOP ISOLATION TESTS COMPLETED SUCCESSFULLY!  ")
    print("==========================================================")


if __name__ == "__main__":
    run_isolation_tests()
