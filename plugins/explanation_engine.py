"""
explanation_engine.py — Decision Explanation Engine for EcoLoop HVAC Control.

Every AI decision produces:
  1. Human-readable explanation  — formatted text for logs and the dashboard
  2. Structured JSON explanation — complete machine-readable record

Both are stored in logs/explanations.jsonl (one JSON object per line).

The explanation covers:
  - Observed State       (sensor snapshot at decision time)
  - Reasoning Chain      (LLM chain-of-thought, verbatim)
  - Trade-off Evaluation (all 4 candidate actions with accept/reject reasoning)
  - Expected Outcomes    (energy savings %, comfort change, time-to-setpoint)
  - Confidence Breakdown (5-signal decomposition)
  - Risk Assessment      (level + primary risk factor)
  - Final Decision       (action, coil speed, outcome)

Author: EcoLoop AI System
"""

import json
import os
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict

# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CandidateSummary:
    """Compact per-candidate record for the explanation."""
    action:                   str
    coil_speed:               float
    chosen:                   bool
    feasible:                 bool
    energy_pct_of_boost:      float
    expected_comfort_change_c: float
    risk_level:               str
    verdict:                  str    # "CHOSEN" | "REJECTED"
    rejection_reason:         str    # empty if chosen


@dataclass
class DecisionExplanation:
    """Complete explanation record for one control cycle."""

    # Identification
    timestamp:    str
    cycle_number: int

    # Observed State
    zone_temp:         float
    heating_sp:        float
    cooling_sp:        float
    outdoor_temp:      float
    comfort_deviation: float
    cooling_deviation: float
    heating_deviation: float
    comfort_status:    str    # "IN_BAND" | "ABOVE_COOLING by X°C" | ...
    occupancy:         str    # "OCCUPIED" | "UNOCCUPIED"
    hour_of_day:       int
    hvac_mode:         str

    # Reasoning
    reasoning_chain:     str
    candidates:          List[CandidateSummary]
    rejection_reasoning: Dict[str, str]

    # Expected Outcomes
    expected_savings_pct:      float
    expected_comfort_change_c: float
    expected_energy_impact:    str
    expected_comfort_impact:   str

    # Confidence
    confidence_total:        float
    conf_historical_success: float
    conf_sensor_consistency: float
    conf_weather_certainty:  float
    conf_comfort_prediction: float
    conf_sim_stability:      float

    # Risk
    risk_level:     str
    primary_risk:   str   # Descriptive risk factor

    # Decision
    chosen_action: str
    coil_speed:    float
    outcome:       str    # "SUCCESS" | "CORRECTED" | "FALLBACK"
    llm_ok:        bool

    # Formatted outputs (generated, not stored in __init__)
    human_readable:   str = field(default="", repr=False)
    structured_json:  str = field(default="", repr=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        # Remove derived text fields from the structured JSON to avoid bloat
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

_cycle_counter = 0   # Module-level cycle counter (reset on process restart)


class ExplanationEngine:
    """
    Generates human-readable and structured JSON explanations from
    PlannerDecision, ConfidenceBreakdown, and ObservationContext objects.

    Usage (inside CoordinatorAgent.run_cycle):
        explanation = ExplanationEngine.generate(ctx, result, outcome)
        ExplanationEngine.store(explanation, log_path)
    """

    @staticmethod
    def generate(ctx, result, outcome: str) -> DecisionExplanation:
        """
        Build a complete DecisionExplanation from an AgentContext + ExecutionResult.

        Parameters
        ----------
        ctx     : AgentContext  (has obs, planner_decision, confidence, validation)
        result  : ExecutionResult
        outcome : str ("SUCCESS" | "CORRECTED" | "FALLBACK")
        """
        global _cycle_counter
        _cycle_counter += 1

        obs = ctx.obs
        pd  = ctx.planner_decision
        cb  = ctx.confidence

        # ── Comfort status string ─────────────────────────────────────────────
        if obs.cooling_deviation > 0:
            comfort_status = f"ABOVE COOLING SETPOINT by {obs.cooling_deviation:.3f}°C"
        elif obs.heating_deviation > 0:
            comfort_status = f"BELOW HEATING SETPOINT by {obs.heating_deviation:.3f}°C"
        else:
            comfort_status = "IN COMFORT BAND"

        active_dev = max(obs.cooling_deviation, obs.heating_deviation)

        # ── Candidate summaries ───────────────────────────────────────────────
        candidates: List[CandidateSummary] = []
        if pd and pd.candidates:
            for c in pd.candidates:
                chosen = (c.action == result.action)
                rej    = pd.rejection_reasoning.get(c.action, "") if not chosen else ""
                candidates.append(CandidateSummary(
                    action=c.action,
                    coil_speed=c.coil_speed,
                    chosen=chosen,
                    feasible=c.feasible,
                    energy_pct_of_boost=c.energy_pct_of_boost,
                    expected_comfort_change_c=c.expected_comfort_change_c,
                    risk_level=c.risk_level,
                    verdict="CHOSEN" if chosen else "REJECTED",
                    rejection_reason=rej,
                ))

        # ── Primary risk description ──────────────────────────────────────────
        primary_risk = ExplanationEngine._derive_primary_risk(
            active_dev, cb, ctx.validation, outcome
        )

        # ── Confidence values ─────────────────────────────────────────────────
        conf_total = cb.total  if cb else 0.0
        conf_hist  = cb.historical_success   if cb else 0.0
        conf_sens  = cb.sensor_consistency   if cb else 0.0
        conf_wth   = cb.weather_certainty    if cb else 0.0
        conf_comf  = cb.comfort_prediction   if cb else 0.0
        conf_stab  = cb.simulation_stability if cb else 0.0

        exp = DecisionExplanation(
            timestamp    = obs.timestamp,
            cycle_number = _cycle_counter,
            zone_temp    = obs.zone_temp,
            heating_sp   = obs.heating_sp,
            cooling_sp   = obs.cooling_sp,
            outdoor_temp = obs.outdoor_temp,
            comfort_deviation = active_dev,
            cooling_deviation = obs.cooling_deviation,
            heating_deviation = obs.heating_deviation,
            comfort_status    = comfort_status,
            occupancy         = "OCCUPIED" if obs.occupancy >= 0.5 else "UNOCCUPIED",
            hour_of_day       = obs.time_of_day_hour,
            hvac_mode         = obs.hvac_mode.upper(),
            reasoning_chain   = (pd.reasoning  if pd else "Unavailable — LLM fallback used"),
            candidates        = candidates,
            rejection_reasoning = (pd.rejection_reasoning if pd else {}),
            expected_savings_pct      = (pd.expected_savings_pct if pd else 0.0),
            expected_comfort_change_c = (pd.expected_comfort_change_c if pd else 0.0),
            expected_energy_impact    = (pd.expected_energy_impact if pd else "N/A"),
            expected_comfort_impact   = (pd.expected_comfort_impact if pd else "N/A"),
            confidence_total          = conf_total,
            conf_historical_success   = conf_hist,
            conf_sensor_consistency   = conf_sens,
            conf_weather_certainty    = conf_wth,
            conf_comfort_prediction   = conf_comf,
            conf_sim_stability        = conf_stab,
            risk_level    = (pd.risk_level if pd else "unknown"),
            primary_risk  = primary_risk,
            chosen_action = result.action,
            coil_speed    = result.clamped_speed,
            outcome       = outcome,
            llm_ok        = result.ok,
        )

        exp.human_readable  = ExplanationEngine._format_human(exp)
        exp.structured_json = json.dumps(exp.to_dict(), indent=2, ensure_ascii=False)

        return exp

    # ── Storage ───────────────────────────────────────────────────────────────

    @staticmethod
    def store(explanation: DecisionExplanation, log_dir: str) -> None:
        """
        Append explanation to logs/explanations.jsonl (one JSON record per line).
        Never raises — logging must not crash the EnergyPlus simulation.
        """
        try:
            os.makedirs(log_dir, exist_ok=True)
            path = os.path.join(log_dir, "explanations.jsonl")
            record = explanation.to_dict()
            record["human_readable"] = explanation.human_readable
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # ── Human-Readable Formatter ──────────────────────────────────────────────

    @staticmethod
    def _format_human(exp: DecisionExplanation) -> str:
        """
        Render a rich, multi-section human-readable explanation.
        Uses ASCII borders for readability in log files and dashboards.
        """
        bar_width = 10

        def conf_bar(val: float) -> str:
            filled = round(val * bar_width)
            return "[" + "#" * filled + "-" * (bar_width - filled) + "]"

        def candidate_row(c: CandidateSummary) -> str:
            tag    = ">>> CHOSEN  " if c.chosen else "    REJECTED"
            feasib = "YES" if c.feasible else "NO "
            return (
                f"  {tag} | {c.action:<6} | speed={c.coil_speed:.1f} | "
                f"energy={c.energy_pct_of_boost:>5.1f}% | "
                f"comfort={c.expected_comfort_change_c:+.2f}C | "
                f"risk={c.risk_level:<6} | feasible={feasib}"
            )

        # Rejection reasons block
        rej_lines = []
        for act, reason in exp.rejection_reasoning.items():
            if reason:
                rej_lines.append(f"  [{act.upper()}] {reason}")
        rej_block = "\n".join(rej_lines) if rej_lines else "  (none)"

        # Confidence signal section
        conf_risk = (
            "Simulation stability is LOW — oscillation risk elevated." if exp.conf_sim_stability < 0.4
            else "Sensor readings are inconsistent — readings may be stale." if exp.conf_sensor_consistency < 0.5
            else "LLM fallback was used — confidence penalised." if not exp.llm_ok
            else "All confidence signals within normal range."
        )

        lines = [
            "=" * 65,
            f"  HVAC DECISION EXPLANATION — Cycle #{exp.cycle_number} @ {exp.timestamp}",
            "=" * 65,
            "",
            "OBSERVED STATE",
            f"  Zone Temperature  : {exp.zone_temp:.3f}C",
            f"  Cooling Setpoint  : {exp.cooling_sp:.3f}C",
            f"  Heating Setpoint  : {exp.heating_sp:.3f}C",
            f"  Outdoor Temp      : {exp.outdoor_temp:.3f}C  (delta to zone: {exp.outdoor_temp - exp.zone_temp:+.2f}C)",
            f"  Comfort Status    : {exp.comfort_status}",
            f"  Active Deviation  : {exp.comfort_deviation:.3f}C",
            f"  Occupancy         : {exp.occupancy}  (Hour {exp.hour_of_day:02d}:00)",
            f"  HVAC Mode         : {exp.hvac_mode}",
            "",
            "REASONING CHAIN",
            *[f"  {line}" for line in exp.reasoning_chain.splitlines()],
            "",
            "CANDIDATE TRADE-OFF EVALUATION",
            *[candidate_row(c) for c in exp.candidates],
            "",
            "REJECTION RATIONALE",
            rej_block,
            "",
            "EXPECTED OUTCOMES",
            f"  Energy Savings vs Boost  : {exp.expected_savings_pct:+.1f}%",
            f"  Comfort Change (cycle)   : {exp.expected_comfort_change_c:+.2f}C",
            f"  Energy Impact            : {exp.expected_energy_impact}",
            f"  Comfort Impact           : {exp.expected_comfort_impact}",
            "",
            "CONFIDENCE BREAKDOWN",
            f"  Composite Score     : {exp.confidence_total:.3f}  {conf_bar(exp.confidence_total)}",
            f"  Historical Success  : {exp.conf_historical_success:.3f}  {conf_bar(exp.conf_historical_success)}  (weight=30%)",
            f"  Sensor Consistency  : {exp.conf_sensor_consistency:.3f}  {conf_bar(exp.conf_sensor_consistency)}  (weight=20%)",
            f"  Comfort Prediction  : {exp.conf_comfort_prediction:.3f}  {conf_bar(exp.conf_comfort_prediction)}  (weight=20%)",
            f"  Weather Certainty   : {exp.conf_weather_certainty:.3f}  {conf_bar(exp.conf_weather_certainty)}  (weight=15%)",
            f"  Sim. Stability      : {exp.conf_sim_stability:.3f}  {conf_bar(exp.conf_sim_stability)}  (weight=15%)",
            f"  Assessment          : {conf_risk}",
            "",
            "RISK ASSESSMENT",
            f"  Risk Level     : {exp.risk_level.upper()}",
            f"  Primary Factor : {exp.primary_risk}",
            "",
            "FINAL DECISION",
            f"  Action         : {exp.chosen_action.upper()}",
            f"  Coil Speed     : {exp.coil_speed:.2f}  (scale: 0.0=off, 1.9=boost)",
            f"  Confidence     : {exp.confidence_total:.3f}",
            f"  Outcome        : {exp.outcome}",
            f"  LLM Used       : {'YES' if exp.llm_ok else 'NO (safety fallback)'}",
            "=" * 65,
        ]
        return "\n".join(lines)

    # ── Internal Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _derive_primary_risk(
        active_dev: float,
        cb,
        validation,
        outcome: str,
    ) -> str:
        """Identify the single most salient risk factor for the explanation."""
        if outcome in ("CORRECTED", "ROLLED_BACK"):
            return "Validator intervention required — original LLM action violated constraints."
        if cb and cb.simulation_stability < 0.40:
            return f"Low simulation stability ({cb.simulation_stability:.2f}) — oscillation pattern detected in recent actions."
        if cb and cb.total < 0.50:
            return f"Low composite confidence ({cb.total:.2f}) — decision uncertainty is elevated."
        if active_dev > 1.5:
            return f"Large comfort deviation ({active_dev:.2f}°C) — zone is significantly outside comfort band."
        if cb and cb.sensor_consistency < 0.60:
            return "Sensor inconsistency detected — zone temperature readings may be unstable."
        if outcome == "FALLBACK":
            return "LLM unavailable — safety fallback used; decision reliability reduced."
        return "No significant risk factors detected — system operating normally."


# ─────────────────────────────────────────────────────────────────────────────
# Log Reader — used by dashboard
# ─────────────────────────────────────────────────────────────────────────────

def load_explanations(log_dir: str) -> list:
    """
    Read all explanations from logs/explanations.jsonl.
    Returns a list of dicts (one per decision cycle), newest last.
    Returns [] if file does not exist or is unreadable.
    """
    path = os.path.join(log_dir, "explanations.jsonl")
    if not os.path.exists(path):
        return []
    records = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except Exception:
        pass
    return records
