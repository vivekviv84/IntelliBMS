"""
reasoning_agent.py — Production-grade LLM Reasoning Agent for EcoLoop HVAC Control.

Replaces threshold-based rule-engine with a chain-of-thought engineering reasoner.
The LLM receives full observational context, must reason explicitly, and returns a
validated structured JSON schema including: action, reasoning, expected_energy_impact,
expected_comfort_impact, confidence_score.

Author: EcoLoop AI System
"""

import json
import os
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict
from typing import Optional, List

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_NAME = os.getenv("ECOLOOP_MODEL", "qwen2.5:3b")

ACTION_TO_SPEED = {
    "off":    0.0,
    "eco":    1.0,
    "normal": 1.3,
    "boost":  1.9,
}

VALID_ACTIONS = set(ACTION_TO_SPEED.keys())

# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ObservationContext:
    """Full sensor observation bundle passed to the reasoning agent."""
    zone_temp: float
    heating_sp: float
    cooling_sp: float
    outdoor_temp: float
    timestamp: str                            # "MM/DD HH:MM"
    time_of_day_hour: int                     # 0-23
    occupancy: float = 1.0                    # 0.0=unoccupied, 1.0=occupied
    hvac_mode: str = "auto"                   # "heating" | "cooling" | "auto" | "off"
    recent_energy_kwh: float = 0.0            # rolling last-hour energy estimate (kWh)
    comfort_pmv: Optional[float] = None       # PMV index if available (-3 to +3)
    historical_actions: List[str] = field(default_factory=list)   # last N action labels
    historical_speeds: List[float] = field(default_factory=list)  # last N coil speeds

    @property
    def cooling_deviation(self) -> float:
        return max(0.0, self.zone_temp - self.cooling_sp)

    @property
    def heating_deviation(self) -> float:
        return max(0.0, self.heating_sp - self.zone_temp)

    @property
    def in_comfort_band(self) -> bool:
        return self.cooling_deviation == 0.0 and self.heating_deviation == 0.0

    def to_prompt_block(self) -> str:
        """Render a dense, engineering-grade context block for the LLM prompt."""
        recent_actions_str = ", ".join(self.historical_actions[-6:]) if self.historical_actions else "none"
        avg_speed = (sum(self.historical_speeds[-6:]) / len(self.historical_speeds[-6:])
                     if self.historical_speeds else 0.0)
        pmv_str = f"{self.comfort_pmv:.2f}" if self.comfort_pmv is not None else "unavailable"

        comfort_status = "IN_BAND"
        if self.cooling_deviation > 0:
            comfort_status = f"ABOVE_COOLING by {self.cooling_deviation:.3f}C"
        elif self.heating_deviation > 0:
            comfort_status = f"BELOW_HEATING by {self.heating_deviation:.3f}C"

        return f"""=== HVAC ZONE OBSERVATION ===
Timestamp            : {self.timestamp}
Hour of Day          : {self.time_of_day_hour:02d}:00 (0=midnight, 12=noon)
Occupancy            : {"OCCUPIED" if self.occupancy >= 0.5 else "UNOCCUPIED"} ({self.occupancy:.1f})
HVAC Mode            : {self.hvac_mode.upper()}

=== THERMAL STATE ===
Zone Air Temperature : {self.zone_temp:.3f} C
Heating Setpoint     : {self.heating_sp:.3f} C
Cooling Setpoint     : {self.cooling_sp:.3f} C
Outdoor Drybulb      : {self.outdoor_temp:.3f} C
Comfort Status       : {comfort_status}
Cooling Deviation    : {self.cooling_deviation:.3f} C  (>0 = zone too hot)
Heating Deviation    : {self.heating_deviation:.3f} C  (>0 = zone too cold)
PMV Comfort Index    : {pmv_str}  (-3=cold, 0=neutral, +3=hot)

=== ENERGY & HISTORY ===
Recent Energy (1h)   : {self.recent_energy_kwh:.4f} kWh
Avg Coil Speed (6h)  : {avg_speed:.2f}  (scale 0.0=off, 1.9=boost)
Last 6 Actions       : [{recent_actions_str}]"""


@dataclass
class ReasoningDecision:
    """Structured output from the reasoning agent. All fields are required."""
    action: str
    coil_speed: float
    reasoning: str
    expected_energy_impact: str     # e.g. "Reduces compressor load by ~15%"
    expected_comfort_impact: str    # e.g. "Zone will reach setpoint in ~20 min"
    confidence_score: float         # 0.0 – 1.0
    rationale: str                  # short single-sentence summary for the log
    ok: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# System Prompt — Engineering Reasoning Persona
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an autonomous HVAC control agent embedded inside an EnergyPlus building simulation \
with real sensor feedback. You operate as a CLOSED-LOOP ENGINEERING CONTROLLER, not a chatbot.

Your task is to select the optimal coil speed action for a single-zone residential heat pump \
system based on the provided multi-sensor observation bundle.

=== AVAILABLE ACTIONS ===
| Action  | Coil Speed | Use When                                              |
|---------|------------|-------------------------------------------------------|
| off     | 0.0        | Zone is comfortably within both setpoints             |
| eco     | 1.0        | Small deviation (<0.5C), gentle correction needed     |
| normal  | 1.3        | Moderate deviation (0.5-1.5C), steady correction      |
| boost   | 1.9        | Large deviation (>1.5C), rapid correction needed      |

=== ENGINEERING REASONING REQUIREMENTS ===
Before selecting an action, you MUST reason through ALL of the following factors:

1. THERMAL LOAD ANALYSIS
   - Compute actual zone deviation from the active setpoint
   - Account for outdoor temperature as a proxy for thermal load pressure
   - Assess direction: Is zone rising or falling based on outdoor vs zone delta?

2. OCCUPANCY & TIME-OF-DAY CONTEXT
   - During occupied hours (07:00-22:00): prioritize comfort within ±0.5C of setpoint
   - During unoccupied hours (22:00-07:00): allow wider ±1.5C tolerance, prefer "off"
   - Pre-occupancy window (06:00-07:00): begin pre-conditioning to reach setpoint

3. ENERGY EFFICIENCY ANALYSIS
   - Review recent average coil speed from history
   - If zone was recently overshooting (oscillating hot/cold), reduce magnitude
   - Never escalate directly from "off" to "boost" in one step — prefer graduated response
   - If outdoor temp is within 2C of zone temp, passive conditioning may suffice → prefer "eco" or "off"

4. OSCILLATION RISK ASSESSMENT
   - Examine the last 6 actions: if alternating between boost/eco or eco/off repeatedly, reduce magnitude
   - Stable consecutive actions signal the controller is converging — maintain or reduce

5. CONFIDENCE CALIBRATION
   - High confidence (0.8-1.0): all signals agree, clear action indicated
   - Medium confidence (0.5-0.8): conflicting signals (e.g., occupancy says comfort, energy says conserve)
   - Low confidence (0.2-0.5): sensor data borderline, outdoor conditions ambiguous

=== CRITICAL ENGINEERING CONSTRAINTS ===
- Do NOT select "boost" unless deviation > 1.5C OR indoor temperature is moving away from setpoint
- Do NOT select "off" if zone is outside comfort band AND occupied
- Outdoor temp > zone temp + 3C during cooling mode → thermal infiltration risk → prefer "normal" or "boost"
- Never exceed confidence 0.9 when historical_actions is empty (insufficient data)
- PMV > +1.5 → strong cooling needed regardless of setpoint math
- PMV < -1.5 → strong heating needed regardless of setpoint math

=== OUTPUT FORMAT ===
Respond with ONLY a single JSON object. No prose before or after.
All fields are REQUIRED. Do not omit any field.

{
  "action": "<off|eco|normal|boost>",
  "reasoning": "<engineering chain-of-thought: 3-5 sentences covering thermal load, occupancy, energy history, oscillation risk>",
  "expected_energy_impact": "<quantified estimate: e.g. 'Compressor at 1.3/1.9 rated speed reduces kW draw by ~15% vs boost'>",
  "expected_comfort_impact": "<time-to-setpoint estimate: e.g. 'Zone at 25.2C will reach 26.6C cooling band in approximately 2 HVAC cycles'>",
  "confidence_score": <float 0.0 to 1.0>,
  "rationale": "<one concise sentence for the decision log>"
}"""


# ─────────────────────────────────────────────────────────────────────────────
# Core Reasoning Function
# ─────────────────────────────────────────────────────────────────────────────

def get_reasoning_decision(
    obs: ObservationContext,
    retries: int = 2
) -> ReasoningDecision:
    """
    Call the LLM reasoning agent with a full observation context.
    Returns a validated ReasoningDecision. Falls back to a rule-based
    safety decision if LLM is unreachable or returns malformed output.
    """
    user_prompt = (
        obs.to_prompt_block() + "\n\n"
        "Apply engineering reasoning to all factors above. "
        "Select the optimal HVAC action and return the required JSON."
    )

    payload = {
        "model": MODEL_NAME,
        "system": SYSTEM_PROMPT,
        "prompt": user_prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,      # Low temperature for deterministic engineering output
            "top_p": 0.9,
            "num_predict": 512,
        }
    }

    last_error = "Unknown error"

    for attempt in range(1 + retries):
        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                OLLAMA_URL,
                data=req_data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw_bytes = resp.read()

            resp_json = json.loads(raw_bytes.decode("utf-8"))
            raw_text = resp_json.get("response", "").strip()

            # Strip markdown fences if present
            if raw_text.startswith("```"):
                lines = raw_text.splitlines()
                lines = [l for l in lines if not l.strip().startswith("```")]
                raw_text = "\n".join(lines).strip()

            parsed = json.loads(raw_text)
            decision = _validate_and_build(parsed, obs)
            if decision is not None:
                return decision

            last_error = f"Invalid schema on attempt {attempt + 1}"

        except urllib.error.URLError as e:
            last_error = f"Network error: {e}"
        except json.JSONDecodeError as e:
            last_error = f"JSON parse error: {e}"
        except Exception as e:
            last_error = f"Unexpected error: {e}"

        if attempt < retries:
            time.sleep(0.5)

    # ── Safety fallback: deterministic rule-based decision ────────────────
    return _safety_fallback(obs, reason=last_error)


def _validate_and_build(parsed: dict, obs: ObservationContext) -> Optional[ReasoningDecision]:
    """Validate LLM-parsed JSON against the required schema."""
    required_keys = {"action", "reasoning", "expected_energy_impact",
                     "expected_comfort_impact", "confidence_score", "rationale"}
    if not required_keys.issubset(parsed.keys()):
        return None

    action = str(parsed.get("action", "")).strip().lower()
    if action not in VALID_ACTIONS:
        return None

    try:
        confidence = float(parsed["confidence_score"])
        confidence = max(0.0, min(1.0, confidence))
    except (ValueError, TypeError):
        return None

    # Sanity guard: override if action conflicts with obvious safety constraints
    action = _safety_override(action, obs)

    return ReasoningDecision(
        action=action,
        coil_speed=ACTION_TO_SPEED[action],
        reasoning=str(parsed.get("reasoning", ""))[:600],
        expected_energy_impact=str(parsed.get("expected_energy_impact", ""))[:200],
        expected_comfort_impact=str(parsed.get("expected_comfort_impact", ""))[:200],
        confidence_score=confidence,
        rationale=str(parsed.get("rationale", ""))[:200],
        ok=True,
    )


def _safety_override(action: str, obs: ObservationContext) -> str:
    """
    Hard engineering constraints applied AFTER LLM output.
    The LLM may reason incorrectly; these guards prevent unsafe actuator states.
    """
    # Guard 1: Never select "off" when zone is outside comfort band AND occupied
    if action == "off" and not obs.in_comfort_band and obs.occupancy >= 0.5:
        return "eco"  # Minimum correction during occupied hours

    # Guard 2: Never select "boost" for deviations < 0.5C (energy waste)
    active_dev = max(obs.cooling_deviation, obs.heating_deviation)
    if action == "boost" and active_dev < 0.5:
        return "eco"

    # Guard 3: PMV override
    if obs.comfort_pmv is not None:
        if obs.comfort_pmv > 1.5 and action in ("off", "eco"):
            return "normal"   # Thermal discomfort — must cool
        if obs.comfort_pmv < -1.5 and action in ("off", "eco"):
            return "normal"   # Thermal discomfort — must heat

    return action


def _safety_fallback(obs: ObservationContext, reason: str) -> ReasoningDecision:
    """
    Rule-based safety fallback when LLM is unavailable.
    Deterministic, guaranteed safe. Uses simple threshold logic as last resort.
    """
    active_dev = max(obs.cooling_deviation, obs.heating_deviation)

    if active_dev == 0.0:
        action = "off"
    elif active_dev < 0.5:
        action = "eco"
    elif active_dev < 1.5:
        action = "normal"
    else:
        action = "boost"

    direction = ("cooling" if obs.cooling_deviation > 0
                 else "heating" if obs.heating_deviation > 0
                 else "balanced")

    return ReasoningDecision(
        action=action,
        coil_speed=ACTION_TO_SPEED[action],
        reasoning=f"SAFETY FALLBACK (LLM unavailable: {reason}). Rule-based decision from deviation={active_dev:.3f}C.",
        expected_energy_impact="Unknown — LLM unavailable",
        expected_comfort_impact=f"Deterministic {direction} correction applied",
        confidence_score=0.4,
        rationale=f"LLM_FAILURE fallback: {action} applied for {direction} deviation={active_dev:.2f}C",
        ok=False,
    )
