"""
planning_agent.py — Multi-Candidate HVAC Action Planner.

Replaces single-shot action selection with explicit candidate evaluation.
The LLM evaluates ALL four candidate actions before choosing one,
producing structured output that includes:
  - Per-candidate feasibility, energy cost, comfort change, risk, evaluation
  - Chosen action with full chain-of-thought reasoning
  - Rejection reasoning for every unchosen candidate
  - Confidence score (anchored to ConfidenceEngine pre-computation)
  - Risk level, expected savings, expected comfort change

One LLM call per cycle. The planner uses the confidence breakdown
computed by ConfidenceEngine as a prior — the LLM must keep its
confidence_score within ±0.10 of the computed value.

Author: EcoLoop AI System
"""

import json
import os
import time
import requests
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict

from reasoning_agent import (
    ObservationContext,
    ACTION_TO_SPEED,
    VALID_ACTIONS,
    _safety_fallback,
    _safety_override,
)
from confidence_engine import ConfidenceBreakdown

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_NAME = os.getenv("ECOLOOP_MODEL", "qwen2.5:3b")

CANDIDATE_ACTIONS = ["off", "eco", "normal", "boost"]

# Energy cost relative to boost (1.9 speed = 100%)
CANDIDATE_ENERGY_FRACTIONS = {
    "off":    0.0,
    "eco":    1.0 / 1.9,   # ≈ 0.526
    "normal": 1.3 / 1.9,   # ≈ 0.684
    "boost":  1.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CandidateEvaluation:
    """LLM-evaluated assessment of one candidate action."""
    action:                   str
    coil_speed:               float
    feasible:                 bool
    expected_comfort_change_c: float   # °C change per HVAC cycle (+ = warming, - = cooling)
    energy_pct_of_boost:      float   # 0-100 relative to boost baseline
    risk_level:               str     # "low" | "medium" | "high"
    evaluation:               str     # reasoning for this specific candidate

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PlannerDecision:
    """
    Complete structured output from one planning cycle.
    Includes chosen action, all candidate evaluations, and rejection reasoning.
    """
    chosen_action:            str
    coil_speed:               float
    candidates:               List[CandidateEvaluation]
    rejection_reasoning:      Dict[str, str]   # action → why it was rejected
    reasoning:                str              # full chain-of-thought
    expected_energy_impact:   str             # human-readable energy description
    expected_comfort_impact:  str             # human-readable comfort description
    confidence_score:         float           # anchored to ConfidenceEngine value
    risk_level:               str             # "low" | "medium" | "high"
    expected_savings_pct:     float           # % energy saving vs boost baseline
    expected_comfort_change_c: float          # selected action's comfort change
    rationale:                str             # single-sentence log entry
    ok:                       bool            # False = safety fallback was used

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ─────────────────────────────────────────────────────────────────────────────
# System Prompt — Multi-Candidate Planner
# ─────────────────────────────────────────────────────────────────────────────

PLANNER_SYSTEM_PROMPT = """\
You are an autonomous HVAC planning agent. Unlike a simple classifier, you evaluate \
EVERY candidate action before committing to a choice. This produces more robust, \
explainable, and energy-efficient decisions than single-step selection.

=== CANDIDATE ACTIONS ===
| Action | Coil Speed | Energy (% of max) | Nominal Use Case                        |
|--------|------------|--------------------|-----------------------------------------|
| off    | 0.0        |  0%               | Zone fully comfortable, no load needed  |
| eco    | 1.0        | 53%               | Small deviation, gentle correction      |
| normal | 1.3        | 68%               | Moderate deviation, steady correction   |
| boost  | 1.9        | 100%              | Large deviation, rapid correction       |

=== EVALUATION FRAMEWORK ===
For EACH of the four candidates, assess these dimensions:

1. THERMAL FEASIBILITY
   - Will this action address the current comfort deviation within 1-2 HVAC cycles?
   - Set feasible=true only if the action is thermally adequate for the situation.

2. EXPECTED COMFORT CHANGE (°C per cycle)
   - Estimate the zone temperature change this action will produce in one ~10-min cycle.
   - Positive = zone warms, Negative = zone cools.
   - Base estimate on: coil capacity ratio, deviation magnitude, outdoor load pressure.

3. ENERGY COST (% of boost)
   - off=0%, eco=53%, normal=68%, boost=100%. These are fixed — do not change them.

4. RISK LEVEL
   - "low"    — action is thermally matched to the deviation, no oscillation risk
   - "medium" — action may overshoot or be slightly under-powered, minor risk
   - "high"   — action clearly mismatched: too aggressive or completely inadequate

5. REJECTION REASON (for unchosen candidates)
   - Explain precisely WHY this candidate is inferior to the chosen action.
   - Must be a specific engineering reason, not vague. Example:
     "boost: 1.9 speed is excessive for 0.3°C deviation — would overshoot and oscillate"

=== SELECTION CRITERIA (priority order) ===
1. SAFETY: Action must not allow comfort deviation to grow if zone is occupied
2. EFFICIENCY: Among feasible actions, prefer the lowest energy cost
3. STABILITY: Avoid actions that contradict the last 6 decisions (oscillation risk)
4. CONFIDENCE: If confidence is low (<0.55), prefer more conservative actions

=== CONFIDENCE ANCHORING ===
The system has pre-computed a mathematically grounded confidence score.
Your confidence_score MUST be within ±0.10 of the provided pre-computed value.
If the computed confidence is 0.72, your response must be between 0.62 and 0.82.

=== OUTPUT FORMAT ===
Respond with ONLY a single JSON object. No text before or after. All fields required.

{
  "candidates": [
    {
      "action": "off",
      "feasible": false,
      "expected_comfort_change_c": 0.3,
      "energy_pct_of_boost": 0,
      "risk_level": "high",
      "evaluation": "<specific engineering assessment>"
    },
    {
      "action": "eco",
      "feasible": true,
      "expected_comfort_change_c": -0.5,
      "energy_pct_of_boost": 53,
      "risk_level": "low",
      "evaluation": "<specific engineering assessment>"
    },
    {
      "action": "normal",
      "feasible": true,
      "expected_comfort_change_c": -0.8,
      "energy_pct_of_boost": 68,
      "risk_level": "low",
      "evaluation": "<assessment>"
    },
    {
      "action": "boost",
      "feasible": true,
      "expected_comfort_change_c": -1.5,
      "energy_pct_of_boost": 100,
      "risk_level": "medium",
      "evaluation": "<assessment>"
    }
  ],
  "chosen_action": "eco",
  "rejection_reasoning": {
    "off": "<why off was rejected>",
    "normal": "<why normal was not chosen over eco>",
    "boost": "<why boost was not chosen>"
  },
  "reasoning": "<3-5 sentence engineering chain-of-thought explaining the full decision>",
  "expected_energy_impact": "<quantified description, e.g. 'eco at 53% load saves 47% vs boost for this cycle'>",
  "expected_comfort_impact": "<time-to-setpoint estimate, e.g. 'zone at 27.2C will reach 26.6C band in ~2 cycles'>",
  "confidence_score": 0.78,
  "risk_level": "low",
  "expected_savings_pct": 47.0,
  "expected_comfort_change_c": -0.5,
  "rationale": "<one concise sentence for the decision log>"
}"""


# ─────────────────────────────────────────────────────────────────────────────
# Main Planner Function
# ─────────────────────────────────────────────────────────────────────────────

def get_planning_decision(
    obs: ObservationContext,
    confidence: ConfidenceBreakdown,
    retries: int = 2,
) -> PlannerDecision:
    """
    Call the LLM planner with full multi-candidate evaluation context.
    Returns a validated PlannerDecision. Falls back to deterministic safety
    planner if LLM is unreachable or returns malformed JSON.
    """
    user_prompt = _build_user_prompt(obs, confidence)

    payload = {
        "model": MODEL_NAME,
        "system": PLANNER_SYSTEM_PROMPT,
        "prompt": user_prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 800,
        }
    }

    last_error = "Unknown error"

    for attempt in range(1 + retries):
        try:
            t_ollama_0 = time.perf_counter()
            resp = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=(5, 35)  # 5s connect timeout, 35s read timeout
            )
            dt_ollama_ms = (time.perf_counter() - t_ollama_0) * 1000.0

            # Store on obs if tracer is active
            if hasattr(obs, "_tracer") and obs._tracer:
                obs._tracer.record_stage_ms("ollama_request", dt_ollama_ms)

            resp_json = resp.json()
            raw_text  = resp_json.get("response", "").strip()

            # Strip markdown fences
            if raw_text.startswith("```"):
                raw_text = "\n".join(
                    l for l in raw_text.splitlines()
                    if not l.strip().startswith("```")
                ).strip()

            parsed  = json.loads(raw_text)
            result  = _validate_and_build(parsed, obs, confidence)
            if result is not None:
                return result

            last_error = f"Schema validation failed on attempt {attempt + 1}"

        except requests.exceptions.Timeout as e:
            last_error = f"Network Timeout (connect/read): {e}"
        except requests.exceptions.RequestException as e:
            last_error = f"Network Request Error: {e}"
        except json.JSONDecodeError as e:
            last_error = f"JSON parse: {e}"
        except Exception as e:
            last_error = f"Unexpected: {e}"

        if attempt < retries:
            time.sleep(0.5)

    # ── Deterministic safety planner ─────────────────────────────────────────
    return _safety_planner(obs, confidence, reason=last_error)


# ─────────────────────────────────────────────────────────────────────────────
# Placeholder Detection & Deterministic Rejection Reasoning
# ─────────────────────────────────────────────────────────────────────────────

def _is_placeholder(text: str) -> bool:
    """Detect unfilled template strings returned by small LLMs."""
    if not text:
        return True
    t = text.lower().strip()
    return any([
        t.startswith("<why"),
        t.startswith("<reason"),
        t.startswith("<explain"),
        t == "n/a",
        t == "none",
        len(t) < 8,
        (t.startswith("why ") and len(t) < 25),
    ])


def _derive_rejection(rejected: str, chosen: str, obs: ObservationContext) -> str:
    """
    Deterministic, data-backed rejection reason when LLM returns a placeholder.
    Produces engineering rationale using actual observation values.
    """
    active_dev = max(obs.cooling_deviation, obs.heating_deviation)
    is_cooling = obs.cooling_deviation > 0
    direction  = "cooling" if is_cooling else ("heating" if obs.heating_deviation > 0 else "balanced")
    chosen_spd = ACTION_TO_SPEED.get(chosen, 1.0)
    rej_spd    = ACTION_TO_SPEED.get(rejected, 0.0)

    _TIER_ORDER = ["off", "eco", "normal", "boost"]
    chosen_idx  = _TIER_ORDER.index(chosen)  if chosen  in _TIER_ORDER else 1
    rej_idx     = _TIER_ORDER.index(rejected) if rejected in _TIER_ORDER else 0

    if rej_idx > chosen_idx:
        rej_pct = round(CANDIDATE_ENERGY_FRACTIONS.get(rejected, 1.0) * 100)
        cho_pct = round(CANDIDATE_ENERGY_FRACTIONS.get(chosen, 0.5) * 100)
        return (
            f"{rejected} (speed={rej_spd}) is more aggressive than required for "
            f"dev={active_dev:.2f}°C {direction}. Uses {rej_pct}% compressor load "
            f"vs {cho_pct}% for {chosen} — over-correction risk and energy waste."
        )
    elif rej_idx < chosen_idx:
        return (
            f"{rejected} (speed={rej_spd}) insufficient for dev={active_dev:.2f}°C "
            f"{direction}. {chosen} (speed={chosen_spd}) selected — deviation would "
            f"persist or grow with {rejected} given current thermal load."
        )
    else:
        return (
            f"{rejected}: equivalent tier to {chosen} but {chosen} chosen based on "
            f"energy-comfort optimisation for dev={active_dev:.2f}°C."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_user_prompt(obs: ObservationContext, confidence: ConfidenceBreakdown) -> str:
    """Combine observation context, memory summary, and confidence breakdown."""
    blocks = [obs.to_prompt_block()]

    # Inject memory summary if available
    memory_summary = getattr(obs, "_memory_summary", "")
    if memory_summary:
        blocks.append(memory_summary)

    energy_note = getattr(obs, "_energy_note", "")
    if energy_note:
        blocks.append(f"=== ENERGY OPTIMIZER ===\n{energy_note}")

    comfort_note = getattr(obs, "_comfort_note", "")
    if comfort_note:
        blocks.append(f"=== COMFORT OPTIMIZER ===\n{comfort_note}")

    blocks.append(confidence.to_prompt_block())
    blocks.append(
        "Now evaluate ALL FOUR candidate actions (off, eco, normal, boost) and return the required JSON."
    )

    return "\n\n".join(blocks)


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def _validate_and_build(
    parsed: dict,
    obs: ObservationContext,
    confidence: ConfidenceBreakdown,
) -> Optional[PlannerDecision]:
    """Validate LLM JSON against required schema. Returns None on any failure."""
    required = {
        "candidates", "chosen_action", "rejection_reasoning",
        "reasoning", "expected_energy_impact", "expected_comfort_impact",
        "confidence_score", "risk_level", "expected_savings_pct",
        "expected_comfort_change_c", "rationale",
    }
    if not required.issubset(parsed.keys()):
        return None

    chosen = str(parsed.get("chosen_action", "")).strip().lower()
    if chosen not in VALID_ACTIONS:
        return None

    # Parse candidates
    raw_candidates = parsed.get("candidates", [])
    if not isinstance(raw_candidates, list) or len(raw_candidates) < 2:
        return None

    candidates = []
    for rc in raw_candidates:
        action = str(rc.get("action", "")).strip().lower()
        if action not in VALID_ACTIONS:
            continue
        candidates.append(CandidateEvaluation(
            action=action,
            coil_speed=ACTION_TO_SPEED[action],
            feasible=bool(rc.get("feasible", False)),
            expected_comfort_change_c=float(rc.get("expected_comfort_change_c", 0.0)),
            energy_pct_of_boost=float(rc.get("energy_pct_of_boost",
                                              CANDIDATE_ENERGY_FRACTIONS[action] * 100)),
            risk_level=str(rc.get("risk_level", "medium")).lower(),
            evaluation=str(rc.get("evaluation", ""))[:300],
        ))

    if not candidates:
        return None

    # Confidence anchoring: clamp to ±0.10 of computed value
    raw_conf = float(parsed.get("confidence_score", confidence.total))
    anchored_conf = max(confidence.total - 0.10, min(confidence.total + 0.10, raw_conf))
    anchored_conf = max(0.0, min(1.0, anchored_conf))

    risk = str(parsed.get("risk_level", "medium")).lower()
    if risk not in ("low", "medium", "high"):
        risk = "medium"

    # Safety override
    chosen = _safety_override(chosen, obs)

    # Compute energy savings % for the chosen action vs boost
    chosen_energy_frac = CANDIDATE_ENERGY_FRACTIONS.get(chosen, 1.0)
    savings_pct = round((1.0 - chosen_energy_frac) * 100.0, 1)

    # Build rejection reasoning — fill any unfilled LLM placeholders deterministically
    raw_rejections = parsed.get("rejection_reasoning", {})
    rejections = {}
    for act in VALID_ACTIONS:
        if act == chosen:
            continue
        raw_reason = str(raw_rejections.get(act, "")).strip()
        if _is_placeholder(raw_reason):
            raw_reason = _derive_rejection(act, chosen, obs)
        rejections[act] = raw_reason[:200]

    return PlannerDecision(
        chosen_action=chosen,
        coil_speed=ACTION_TO_SPEED[chosen],
        candidates=candidates,
        rejection_reasoning=rejections,
        reasoning=str(parsed.get("reasoning", ""))[:600],
        expected_energy_impact=str(parsed.get("expected_energy_impact", ""))[:200],
        expected_comfort_impact=str(parsed.get("expected_comfort_impact", ""))[:200],
        confidence_score=anchored_conf,
        risk_level=risk,
        expected_savings_pct=savings_pct,
        expected_comfort_change_c=float(parsed.get("expected_comfort_change_c", 0.0)),
        rationale=str(parsed.get("rationale", ""))[:200],
        ok=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic Safety Planner
# ─────────────────────────────────────────────────────────────────────────────

def _safety_planner(
    obs: ObservationContext,
    confidence: ConfidenceBreakdown,
    reason: str,
) -> PlannerDecision:
    """
    Fully deterministic multi-candidate planner used when LLM is unavailable.
    Evaluates all candidates using pure threshold logic.
    """
    active_dev    = max(obs.cooling_deviation, obs.heating_deviation)
    is_cooling    = obs.cooling_deviation > 0
    outdoor_load  = obs.outdoor_temp - obs.zone_temp   # + = outdoor pushing heat in

    # Evaluate each candidate
    candidates = []
    for act in CANDIDATE_ACTIONS:
        spd    = ACTION_TO_SPEED[act]
        e_pct  = CANDIDATE_ENERGY_FRACTIONS[act] * 100.0

        # Comfort change per cycle estimate (rough physics proxy)
        # Each unit of speed provides ~0.8°C correction per cycle at 0 load
        # Reduced by thermal load from outdoors
        raw_correction = spd * 0.8 - max(0.0, outdoor_load * 0.05)
        comfort_change = -raw_correction if is_cooling else raw_correction

        # Feasibility
        feasible = (
            act != "off" or active_dev == 0.0
        )
        if active_dev > 1.5 and act in ("off", "eco"):
            feasible = False

        # Risk
        if active_dev == 0.0:
            risk = "high" if act in ("normal", "boost") else "low"
        elif active_dev < 0.5:
            risk = "high" if act == "boost" else ("medium" if act == "normal" else "low")
        elif active_dev < 1.5:
            risk = "high" if act == "off" else ("low" if act in ("eco", "normal") else "medium")
        else:
            risk = "high" if act in ("off", "eco") else "low"

        note = f"Deterministic rule: dev={active_dev:.2f}C, spd={spd:.1f}"
        candidates.append(CandidateEvaluation(
            action=act, coil_speed=spd, feasible=feasible,
            expected_comfort_change_c=round(comfort_change, 2),
            energy_pct_of_boost=round(e_pct, 1),
            risk_level=risk, evaluation=note,
        ))

    # Select best feasible + lowest energy
    feasible_cands = [c for c in candidates if c.feasible]
    if not feasible_cands:
        feasible_cands = candidates  # all infeasible — pick anyway

    # Sort: prefer lower risk first, then lower energy
    risk_order = {"low": 0, "medium": 1, "high": 2}
    chosen_cand = sorted(
        feasible_cands,
        key=lambda c: (risk_order.get(c.risk_level, 1), c.energy_pct_of_boost)
    )[0]
    chosen = chosen_cand.action

    rejections = {
        c.action: f"Safety fallback rejected {c.action}: risk={c.risk_level}, feasible={c.feasible}"
        for c in candidates if c.action != chosen
    }

    e_savings = round((1.0 - CANDIDATE_ENERGY_FRACTIONS[chosen]) * 100.0, 1)

    return PlannerDecision(
        chosen_action=chosen,
        coil_speed=ACTION_TO_SPEED[chosen],
        candidates=candidates,
        rejection_reasoning=rejections,
        reasoning=f"SAFETY FALLBACK ({reason}). Deterministic selection: dev={active_dev:.2f}C → {chosen}.",
        expected_energy_impact=f"Safety planner: {chosen} at {e_savings:.0f}% savings vs boost.",
        expected_comfort_impact=f"Estimated {abs(chosen_cand.expected_comfort_change_c):.1f}C correction per cycle.",
        confidence_score=confidence.total * 0.7,   # Discount: LLM unavailable
        risk_level=chosen_cand.risk_level,
        expected_savings_pct=e_savings,
        expected_comfort_change_c=chosen_cand.expected_comfort_change_c,
        rationale=f"LLM_FALLBACK: {chosen} for dev={active_dev:.2f}C (reason: {reason[:80]})",
        ok=False,
    )
