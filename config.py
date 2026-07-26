"""
EcoLoop Configuration File
Central repository for environment variables, system constants, utility tariffs,
and commercial HVAC economizer engineering parameters.
"""
import os

# BESCOM Bengaluru Commercial Building Tariff (₹ / kWh)
# Category LT-3 / HT-2 Commercial (inclusive of base energy rate, true-up charges, and 9% electricity tax)
# Can be overridden via environment variable: export ELECTRICITY_TARIFF_INR_KWH=9.50
ELECTRICITY_TARIFF_INR_KWH = float(os.environ.get("ELECTRICITY_TARIFF_INR_KWH", "9.50"))

# Grid Carbon Intensity (kg CO2 / kWh)
CARBON_KG_PER_KWH = float(os.environ.get("CARBON_KG_PER_KWH", "0.233"))

# ─────────────────────────────────────────────────────────────────────────────
# Economizer & Commercial HVAC Engineering Parameters
# ─────────────────────────────────────────────────────────────────────────────

# Representative cooling power (kW) used for estimating avoided compressor energy.
# Configurable because compressor electrical capacity depends on building size and equipment.
ESTIMATED_COOLING_POWER_KW = float(os.environ.get("ESTIMATED_COOLING_POWER_KW", "1.5"))

# Hourly decision interval in hours (default: 1.0 hour)
DECISION_INTERVAL_HOURS = float(os.environ.get("DECISION_INTERVAL_HOURS", "1.0"))

# Minimum required indoor vs outdoor drybulb temperature advantage (°C) before free cooling is considered
ECONOMIZER_TEMP_ADVANTAGE_MIN_C = float(os.environ.get("ECONOMIZER_TEMP_ADVANTAGE_MIN_C", "1.5"))

# Economizer runtime factor (1.0 = 100% compressor power bypass during free cooling)
ECONOMIZER_RUNTIME_FACTOR = float(os.environ.get("ECONOMIZER_RUNTIME_FACTOR", "1.0"))

# ─────────────────────────────────────────────────────────────────────────────
# Demand Response & Time-of-Use (ToU) Tariff Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Enable Time-of-Use (ToU) tariff pricing model
ENABLE_TOU_TARIFF = True

# ToU Tariff Multipliers relative to base commercial tariff (ELECTRICITY_TARIFF_INR_KWH)
# OFF_PEAK: 0.80 (20% discount: 10 PM - 6 AM)
# NORMAL  : 1.00 (Base tariff: 6 AM - 6 PM)
# PEAK    : 1.35 (35% surcharge: 6 PM - 10 PM)
TOU_MULTIPLIERS = {
    "OFF_PEAK": 0.80,
    "NORMAL":   1.00,
    "PEAK":     1.35,
}

# Peak ToU hours window (start_hour, end_hour) -> 18:00 to 22:00 (6 PM - 10 PM)
PEAK_WINDOWS = [(18, 22)]

# Maximum allowed comfort deviation (°C) during peak tariff before allowing load shedding
DR_MAX_COMFORT_DEV_C = float(os.environ.get("DR_MAX_COMFORT_DEV_C", "0.8"))

# ─────────────────────────────────────────────────────────────────────────────
# Predictive Pre-Cooling Configuration (EPW Look-Ahead Weather Horizon)
# ─────────────────────────────────────────────────────────────────────────────

# Lookahead forecast horizon in hours (2-4 hours)
PRECOOL_LOOKAHEAD_HOURS = int(os.environ.get("PRECOOL_LOOKAHEAD_HOURS", "3"))

# Outdoor drybulb temperature threshold (°C) triggering pre-cooling recommendation
PRECOOL_TEMP_THRESHOLD_C = float(os.environ.get("PRECOOL_TEMP_THRESHOLD_C", "28.0"))

# Outdoor temperature rise differential threshold (°C) (future outdoor >= current outdoor + 2.0°C)
PRECOOL_TEMP_DIFF_MIN_C = float(os.environ.get("PRECOOL_TEMP_DIFF_MIN_C", "2.0"))

# Maximum coil speed ceiling during pre-cooling (never exceed 1.0)
PRECOOL_MAX_COIL_SPEED = float(os.environ.get("PRECOOL_MAX_COIL_SPEED", "1.0"))

