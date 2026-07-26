"""
agent_system.py — Multi-Agent Orchestration System for EcoLoop HVAC Control.

Phase 5 extensions:
  - ConfidenceEngine computes mathematically grounded confidence per cycle
  - PlannerAgent replaces single-shot LLM with multi-candidate evaluation
  - AgentContext carries ConfidenceBreakdown through the pipeline
  - LoggerAgent writes rejection_reasoning + full confidence breakdown to CSV

Seven collaborating, single-responsibility agents coordinated by a central
Coordinator. Agents communicate via typed dataclass messages.
Only one LLM call is made per cycle (inside CoordinatorAgent).

Agent pipeline per control cycle:
    CoordinatorAgent
        ├── EnergyOptimizerAgent    (pure Python, reads memory trends)
        ├── ComfortOptimizerAgent   (pure Python, reads comfort trends)
        ├── ReasoningAgent          (ONE LLM call, receives all context)
        ├── ValidatorAgent          (pure Python, enforces hard constraints)
        ├── PlannerAgent            (pure Python, synthesizes into final plan)
        ├── ActuatorExecutorAgent   (pure Python, clamps and formats output)
        └── LoggerAgent             (pure Python, writes memory + CSV)

Design principles:
    - Low latency: only ONE LLM call per full pipeline execution
    - No duplicated reasoning: each agent has a distinct, non-overlapping role
    - No threads / no async: sequential execution, EnergyPlus compatible
    - Typed messages: all inter-agent communication via dataclasses
    - Memory-aware: history injected into LLM prompt via ShortTermMemory

Author: EcoLoop AI System
"""

import csv
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from agent_memory import MemoryRecord, ShortTermMemory
from reasoning_agent import (
    ObservationContext,
    ReasoningDecision,
    ACTION_TO_SPEED,
    VALID_ACTIONS,
    get_reasoning_decision,
    _safety_fallback,
    _safety_override,
)
from confidence_engine import ConfidenceEngine, ConfidenceBreakdown
from economizer import EconomizerModule, EconomizerRecommendation
from demand_response import DemandResponseModule, DemandResponseRecommendation
from predictive_controller import PredictivePrecoolModule, PredictivePrecoolRecommendation
from planning_agent import (
    PlannerDecision,
    get_planning_decision,
    CandidateEvaluation,
)
from closed_loop_controller import (
    ViolationDetector,
    CorrectionEngine,
    ViolationType,
    CycleOutcome,
    COMFORT_VIOLATION_THRESH,
    COOLDOWN_CYCLES,
)
try:
    from explanation_engine import ExplanationEngine, load_explanations
    _EXPLANATION_AVAILABLE = True
except ImportError:
    ExplanationEngine = None
    _EXPLANATION_AVAILABLE = False

try:
    from resilience import ResilienceManager
    _RESILIENCE_AVAILABLE = True
except ImportError:
    ResilienceManager = None
    _RESILIENCE_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Inter-Agent Message Types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EnergyRecommendation:
    """Output from EnergyOptimizerAgent."""
    preferred_action:    str      # "off" | "eco" | "normal" | "boost"
    max_allowed_speed:   float    # Hard ceiling for coil speed this cycle
    trend:               str      # "rising" | "falling" | "stable"
    rationale:           str

@dataclass
class ComfortRecommendation:
    """Output from ComfortOptimizerAgent."""
    required_action:  str     # Minimum action required to satisfy comfort
    urgency:          float   # 0.0 (low) – 1.0 (critical)
    trend:            str     # "improving" | "degrading" | "stable"
    rationale:        str

@dataclass
class ValidationResult:
    """Output from ValidatorAgent."""
    approved:         bool
    violations:       List[str]
    override_action:  Optional[str]   # Set only when validator forces a change
    override_speed:   Optional[float]

@dataclass
class Plan:
    """Output from PlannerAgent — authoritative action for this cycle."""
    action:     str
    coil_speed: float
    rationale:  str
    source:     str   # "llm" | "validator_override" | "energy_constraint" | "fallback"

@dataclass
class ExecutionResult:
    """Output from ActuatorExecutorAgent."""
    action:        str
    coil_speed:    float    # Requested (pre-clamp)
    clamped_speed: float    # Applied to actuator (post-clamp)
    rationale:     str
    ok:            bool

@dataclass
class AgentContext:
    """
    Shared context bundle flowing through the pipeline.
    Created once per cycle by CoordinatorAgent.
    """
    obs:              ObservationContext
    memory:           ShortTermMemory
    energy_rec:       Optional[EnergyRecommendation]           = None
    comfort_rec:      Optional[ComfortRecommendation]          = None
    economizer_rec:   Optional[EconomizerRecommendation]       = None
    dr_rec:           Optional[DemandResponseRecommendation]   = None
    precool_rec:      Optional[PredictivePrecoolRecommendation] = None
    llm_decision:     Optional[ReasoningDecision]              = None
    planner_decision: Optional[PlannerDecision]       = None
    confidence:       Optional[ConfidenceBreakdown]   = None
    validation:       Optional[ValidationResult]      = None
    plan:             Optional[Plan]                  = None
    result:           Optional[ExecutionResult]       = None
    violations:       List[str]                       = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Agent 1: Energy Optimizer
# ─────────────────────────────────────────────────────────────────────────────

class EnergyOptimizerAgent:
    """
    Analyzes recent energy consumption trends from memory.
    Returns a recommended maximum action tier and trend direction.
    Pure Python — NO LLM call.
    """

    def process(self, ctx: AgentContext) -> EnergyRecommendation:
        recent = ctx.memory.get_recent(6)
        speeds = [r.coil_speed for r in recent]

        if not speeds:
            return EnergyRecommendation(
                preferred_action="eco",
                max_allowed_speed=1.9,
                trend="stable",
                rationale="No prior energy history. Conservative eco default."
            )

        avg_speed   = sum(speeds) / len(speeds)
        recent_kwh  = ctx.memory.recent_energy_total(6)
        success_rate = ctx.memory.recent_success_rate(6)

        # Trend analysis
        if len(speeds) >= 3:
            first_half  = sum(speeds[:len(speeds)//2]) / (len(speeds)//2)
            second_half = sum(speeds[len(speeds)//2:]) / (len(speeds) - len(speeds)//2)
            if second_half > first_half + 0.2:
                trend = "rising"
            elif second_half < first_half - 0.2:
                trend = "falling"
            else:
                trend = "stable"
        else:
            trend = "stable"

        # Energy ceiling: if energy has been high and comfort is OK, reduce ceiling
        obs = ctx.obs
        active_dev = max(obs.cooling_deviation, obs.heating_deviation)

        if trend == "rising" and active_dev < 0.5 and success_rate > 0.7:
            preferred = "eco"
            max_speed = 1.0
            note = f"Energy rising trend (avg_speed={avg_speed:.2f}), deviation low → cap at eco."
        elif avg_speed > 1.6 and recent_kwh > 0.01:
            preferred = "normal"
            max_speed = 1.3
            note = f"High avg coil speed ({avg_speed:.2f}) → cap at normal to conserve energy."
        else:
            preferred = "normal"
            max_speed = 1.9
            note = f"Energy trend={trend}, avg_speed={avg_speed:.2f} → no ceiling applied."

        return EnergyRecommendation(
            preferred_action=preferred,
            max_allowed_speed=max_speed,
            trend=trend,
            rationale=note
        )


# ─────────────────────────────────────────────────────────────────────────────
# Agent 2: Comfort Optimizer
# ─────────────────────────────────────────────────────────────────────────────

class ComfortOptimizerAgent:
    """
    Analyzes recent comfort deviation trends from memory.
    Returns the minimum action tier required to maintain comfort.
    Pure Python — NO LLM call.
    """

    def process(self, ctx: AgentContext) -> ComfortRecommendation:
        obs    = ctx.obs
        recent = ctx.memory.get_recent(6)

        active_dev  = max(obs.cooling_deviation, obs.heating_deviation)
        avg_dev     = ctx.memory.recent_avg_comfort_deviation(6)

        # Trend analysis
        if recent and len(recent) >= 2:
            prev_dev = recent[-1].comfort_deviation
            if active_dev > prev_dev + 0.15:
                trend = "degrading"
            elif active_dev < prev_dev - 0.15:
                trend = "improving"
            else:
                trend = "stable"
        else:
            trend = "stable"

        # Occupied hours demand tighter comfort
        hour = obs.time_of_day_hour
        occupied = obs.occupancy >= 0.5
        peak_comfort_hours = 7 <= hour <= 22

        # Urgency: 0.0 = comfortable, 1.0 = critical discomfort
        urgency = min(1.0, active_dev / 2.0)
        if occupied and peak_comfort_hours:
            urgency = min(1.0, urgency * 1.3)

        # Minimum required action
        if active_dev == 0.0:
            required = "off"
        elif active_dev < 0.5:
            required = "eco"
        elif active_dev < 1.5:
            required = "normal"
        else:
            required = "boost"

        # Pre-occupancy boost: ramp up 1 hour before occupied period
        if not occupied and hour == 6:
            required = "eco"    # Pre-condition gently
            urgency  = max(urgency, 0.3)

        note = (
            f"Current deviation={active_dev:.3f}°C (avg_6h={avg_dev:.3f}°C), "
            f"trend={trend}, urgency={urgency:.2f}, occupancy={'yes' if occupied else 'no'}. "
            f"Minimum required action: {required}."
        )

        return ComfortRecommendation(
            required_action=required,
            urgency=urgency,
            trend=trend,
            rationale=note
        )


# ─────────────────────────────────────────────────────────────────────────────
# Agent 3: Validator
# ─────────────────────────────────────────────────────────────────────────────

class ValidatorAgent:
    """
    Enforces hard engineering constraints on the LLM decision.
    Returns approval status and any forced overrides.
    Pure Python — NO LLM call. Deterministic.
    """

    _detector  = ViolationDetector()
    _corrector = CorrectionEngine()

    def process(self, ctx: AgentContext, cooldown_active: bool) -> ValidationResult:
        decision  = ctx.llm_decision
        obs       = ctx.obs
        history   = [
            type('CS', (), {
                'coil_speed': r.coil_speed,
                'action': r.action,
                'zone_temp': r.zone_temp,
                'timestamp': r.timestamp,
            })()
            for r in ctx.memory.get_recent(12)
        ]

        # Run violation detection from closed_loop_controller
        violation_types = self._detector.detect(decision, obs, history)

        if not violation_types:
            return ValidationResult(
                approved=True,
                violations=[],
                override_action=None,
                override_speed=None
            )

        # Run correction
        corrected, was_modified = self._corrector.correct(
            decision, violation_types, obs, cooldown_active
        )

        return ValidationResult(
            approved=not was_modified,
            violations=[v.name for v in violation_types],
            override_action=corrected.action if was_modified else None,
            override_speed=corrected.coil_speed if was_modified else None,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Agent 4: Planner
# ─────────────────────────────────────────────────────────────────────────────

class PlannerAgent:
    """
    Synthesizes EnergyRecommendation + ComfortRecommendation + LLM decision
    + ValidationResult into a single authoritative Plan.
    Resolution priority: safety/validator > comfort minimum > energy ceiling > LLM.
    Pure Python — NO LLM call.
    """

    # Action tier ordering for comparison
    _TIERS = {"off": 0, "eco": 1, "normal": 2, "boost": 3}

    def process(self, ctx: AgentContext) -> Plan:
        llm_action  = ctx.llm_decision.action
        llm_speed   = ctx.llm_decision.coil_speed
        energy_rec  = ctx.energy_rec
        comfort_rec = ctx.comfort_rec
        validation  = ctx.validation

        action = llm_action
        speed  = llm_speed
        source = "llm"

        # Rule 1: Validator override (highest priority — safety constraints)
        if validation and not validation.approved and validation.override_action:
            action = validation.override_action
            speed  = validation.override_speed or ACTION_TO_SPEED[action]
            source = "validator_override"

        # Rule 2: Comfort minimum floor
        if comfort_rec:
            comfort_tier = self._TIERS.get(comfort_rec.required_action, 0)
            current_tier = self._TIERS.get(action, 0)
            if comfort_tier > current_tier and comfort_rec.urgency > 0.5:
                action = comfort_rec.required_action
                speed  = ACTION_TO_SPEED[action]
                source = "comfort_floor"

        # Rule 3: Energy ceiling cap
        if energy_rec and speed > energy_rec.max_allowed_speed:
            speed = energy_rec.max_allowed_speed
            # Remap action to nearest tier at new speed
            action = self._speed_to_action(speed)
            source = "energy_ceiling" if source == "llm" else source

        rationale = (
            f"Plan[{source}]: action={action}, speed={speed:.2f}. "
            f"LLM={llm_action}, comfort_min={comfort_rec.required_action if comfort_rec else 'N/A'}, "
            f"energy_max={energy_rec.max_allowed_speed if energy_rec else 'N/A'}."
        )

        return Plan(action=action, coil_speed=speed, rationale=rationale, source=source)

    @staticmethod
    def _speed_to_action(speed: float) -> str:
        """Map a coil speed value back to the nearest named action."""
        for act, spd in sorted(ACTION_TO_SPEED.items(), key=lambda x: x[1], reverse=True):
            if speed >= spd:
                return act
        return "off"


# ─────────────────────────────────────────────────────────────────────────────
# Agent 5: Actuator Executor
# ─────────────────────────────────────────────────────────────────────────────

class ActuatorExecutorAgent:
    """
    Finalizes the coil speed value for writing to the EnergyPlus actuator.
    Applies hard clamping [0.0, 2.0], validates action label, returns ExecutionResult.
    Pure Python — NO LLM call.
    """

    @staticmethod
    def process(plan: Plan, llm_ok: bool) -> ExecutionResult:
        # Validate action label
        action = plan.action if plan.action in VALID_ACTIONS else "off"

        # Hard safety clamp
        raw_speed    = float(plan.coil_speed) if plan.coil_speed is not None else 0.0
        clamped      = max(0.0, min(2.0, raw_speed))

        return ExecutionResult(
            action=action,
            coil_speed=raw_speed,
            clamped_speed=clamped,
            rationale=plan.rationale,
            ok=llm_ok,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Agent 6: Logger
# ─────────────────────────────────────────────────────────────────────────────

class LoggerAgent:
    """
    Persists the completed cycle record to:
        1. ShortTermMemory ring buffer (in-memory + JSON)
        2. decision_log.csv (columnar, for dashboard)

    Pure Python — NO LLM call. Never raises — logging failures must not
    crash the EnergyPlus simulation.
    """

    def process(
        self,
        ctx:           AgentContext,
        result:        ExecutionResult,
        outcome:       str,
        violations:    List[str],
        via_mcp:       bool,
        log_path:      str,
    ) -> MemoryRecord:
        obs = ctx.obs

        # Compute comfort deviation for this cycle
        active_dev = max(obs.cooling_deviation, obs.heating_deviation)

        # Estimate energy proxy: coil_speed * nominal_kW * cycle_fraction
        # 1.9 (boost) ≈ 3600 W, operating ~10 min = 0.6 kWh max; scale linearly
        energy_kwh = result.clamped_speed * (3600.0 / 1.9) * (600.0 / 3_600_000.0)

        # Prefer planner_decision for richer fields; fall back to llm_decision
        pd = ctx.planner_decision
        ld = ctx.llm_decision
        reasoning_text = (
            pd.reasoning[:300] if pd else
            (ld.reasoning[:300] if ld else "N/A")
        )
        confidence_val = (
            pd.confidence_score if pd else
            (ld.confidence_score if ld else 0.0)
        )

        # Economizer 4-Stage Pipeline Telemetry Extraction
        e_rec = ctx.economizer_rec
        econ_recommended = e_rec.economizer_active if e_rec else False
        econ_mode        = e_rec.recommended_mode if e_rec else "NO_ACTION"
        temp_adv         = e_rec.temperature_advantage if e_rec else 0.0
        runtime_saved    = e_rec.estimated_runtime_saved_hours if e_rec else 0.0
        energy_saved     = e_rec.estimated_energy_saved_kwh if e_rec else 0.0
        econ_conf        = e_rec.confidence if e_rec else 0.0

        planner_act      = pd.chosen_action if pd else "boost"
        planner_accepted = econ_recommended and (planner_act in ("off", "eco"))

        val              = ctx.validation
        validator_ovr    = (val.approved is False or val.override_action is not None) if val else False

        final_act        = result.action if result else "boost"
        final_free_cool  = econ_recommended and (final_act in ("off", "eco")) and not validator_ovr

        # Demand Response Telemetry
        dr_rec           = ctx.dr_rec
        is_pk_win        = dr_rec.is_peak_window if dr_rec else False
        t_period         = dr_rec.current_tariff_period if dr_rec else "NORMAL"
        t_inr            = dr_rec.current_tariff_inr_kwh if dr_rec else 10.0
        dr_recommended   = dr_rec.dr_recommended if dr_rec else False
        dr_accepted      = dr_recommended and (planner_act in ("off", "eco"))
        dr_ovr           = validator_ovr if dr_recommended else False
        dr_final         = dr_recommended and (final_act in ("off", "eco")) and not validator_ovr
        dr_cost_saved    = dr_rec.estimated_cost_saved_inr if dr_rec else 0.0

        # Predictive Pre-Cooling Telemetry
        pc_rec           = ctx.precool_rec
        pc_recommended   = pc_rec.precool_recommended if pc_rec else False
        pc_peak_temp     = pc_rec.predicted_peak_outdoor_temp if pc_rec else 0.0
        pc_accepted      = pc_recommended and (planner_act in ("eco", "normal", "boost"))
        pc_ovr           = validator_ovr if pc_recommended else False
        pc_final         = pc_recommended and (final_act in ("eco", "normal", "boost")) and not validator_ovr

        record = MemoryRecord(
            timestamp                      = obs.timestamp,
            zone_temp                      = obs.zone_temp,
            heating_sp                     = obs.heating_sp,
            cooling_sp                     = obs.cooling_sp,
            outdoor_temp                   = obs.outdoor_temp,
            action                         = result.action,
            coil_speed                     = result.clamped_speed,
            reasoning                      = reasoning_text,
            confidence                     = confidence_val,
            energy_kwh                     = energy_kwh,
            comfort_deviation              = active_dev,
            outcome                        = outcome,
            success                        = (len(violations) == 0),
            violations                     = violations,
            economizer_recommended         = econ_recommended,
            economizer_mode                = econ_mode,
            temperature_advantage         = temp_adv,
            estimated_runtime_saved_hours  = runtime_saved,
            estimated_energy_saved_kwh    = energy_saved,
            planner_accepted               = planner_accepted,
            validator_overrode             = validator_ovr,
            final_free_cooling_used        = final_free_cool,
            economizer_confidence          = econ_conf,
            is_peak_window                 = is_pk_win,
            tariff_period                  = t_period,
            tariff_inr_kwh                 = t_inr,
            dr_recommended                 = dr_recommended,
            dr_planner_accepted            = dr_accepted,
            dr_validator_overrode          = dr_ovr,
            dr_final_used                  = dr_final,
            dr_cost_saved_inr              = dr_cost_saved,
            precool_recommended            = pc_recommended,
            predicted_peak_outdoor_temp   = pc_peak_temp,
            precool_planner_accepted       = pc_accepted,
            precool_validator_overrode     = pc_ovr,
            precool_final_used             = pc_final,
        )

        # Write to memory buffer
        ctx.memory.add(record)

        # Write to CSV log (pass full ctx for planner fields)
        self._write_csv(record, via_mcp, log_path, ctx=ctx)

        return record

    @staticmethod
    def _write_csv(record: MemoryRecord, via_mcp: bool, log_path: str,
                   ctx: "AgentContext | None" = None) -> None:
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            header_needed = not os.path.exists(log_path)

            # Extract planner-specific fields
            pd = ctx.planner_decision if ctx else None
            conf_bd = ctx.confidence if ctx else None
            rejection_json = ""
            candidates_json = ""
            risk_level = ""
            exp_savings = ""
            if pd:
                import json as _json
                rejection_json = _json.dumps(pd.rejection_reasoning)[:400]
                candidates_json = _json.dumps(
                    [{"action": c.action, "risk": c.risk_level,
                      "feasible": c.feasible, "energy_pct": c.energy_pct_of_boost}
                     for c in pd.candidates]
                )[:400]
                risk_level = pd.risk_level
                exp_savings = f"{pd.expected_savings_pct:.1f}"

            conf_s1 = f"{conf_bd.historical_success:.3f}" if conf_bd else ""
            conf_s2 = f"{conf_bd.sensor_consistency:.3f}" if conf_bd else ""
            conf_s3 = f"{conf_bd.weather_certainty:.3f}" if conf_bd else ""
            conf_s4 = f"{conf_bd.comfort_prediction:.3f}" if conf_bd else ""
            conf_s5 = f"{conf_bd.simulation_stability:.3f}" if conf_bd else ""

            with open(log_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if header_needed:
                    w.writerow([
                        "timestamp", "zone_temp", "heating_sp", "cooling_sp",
                        "outdoor_temp", "action", "coil_speed",
                        "confidence", "reasoning",
                        "energy_kwh", "comfort_deviation",
                        "outcome", "success", "violations", "via_mcp",
                        "risk_level", "expected_savings_pct",
                        "rejection_reasoning", "candidates",
                        "conf_historical", "conf_sensor", "conf_weather",
                        "conf_comfort", "conf_stability",
                        "economizer_recommended", "economizer_mode",
                        "temperature_advantage", "estimated_runtime_saved_hours",
                        "estimated_energy_saved_kwh", "planner_accepted",
                        "validator_overrode", "final_free_cooling_used",
                        "economizer_confidence",
                        "is_peak_window", "tariff_period", "tariff_inr_kwh",
                        "dr_recommended", "dr_planner_accepted",
                        "dr_validator_overrode", "dr_final_used", "dr_cost_saved_inr",
                        "precool_recommended", "predicted_peak_outdoor_temp",
                        "precool_planner_accepted", "precool_validator_overrode",
                        "precool_final_used",
                    ])
                w.writerow([
                    record.timestamp,
                    f"{record.zone_temp:.3f}",
                    f"{record.heating_sp:.3f}",
                    f"{record.cooling_sp:.3f}",
                    f"{record.outdoor_temp:.3f}",
                    record.action,
                    f"{record.coil_speed:.2f}",
                    f"{record.confidence:.3f}",
                    record.reasoning[:300],
                    f"{record.energy_kwh:.6f}",
                    f"{record.comfort_deviation:.3f}",
                    record.outcome,
                    record.success,
                    "|".join(record.violations),
                    via_mcp,
                    risk_level,
                    exp_savings,
                    rejection_json,
                    candidates_json,
                    conf_s1, conf_s2, conf_s3, conf_s4, conf_s5,
                    record.economizer_recommended,
                    record.economizer_mode,
                    f"{record.temperature_advantage:.2f}",
                    f"{record.estimated_runtime_saved_hours:.2f}",
                    f"{record.estimated_energy_saved_kwh:.3f}",
                    record.planner_accepted,
                    record.validator_overrode,
                    record.final_free_cooling_used,
                    f"{record.economizer_confidence:.2f}",
                    record.is_peak_window,
                    record.tariff_period,
                    f"{record.tariff_inr_kwh:.2f}",
                    record.dr_recommended,
                    record.dr_planner_accepted,
                    record.dr_validator_overrode,
                    record.dr_final_used,
                    f"{record.dr_cost_saved_inr:.2f}",
                    record.precool_recommended,
                    f"{record.predicted_peak_outdoor_temp:.2f}",
                    record.precool_planner_accepted,
                    record.precool_validator_overrode,
                    record.precool_final_used,
                ])
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Agent 7: Coordinator (Orchestrator)
# ─────────────────────────────────────────────────────────────────────────────

class CoordinatorAgent:
    """
    Top-level orchestrator. Owns the memory object and all specialist agents.
    Executes the full multi-agent pipeline in a single synchronous call.

    Pipeline per cycle:
        1. Build ObservationContext (inject memory history)
        2. EnergyOptimizerAgent  → EnergyRecommendation
        3. ComfortOptimizerAgent → ComfortRecommendation
        4. ReasoningAgent (LLM)  → ReasoningDecision   [SINGLE LLM CALL]
        5. ValidatorAgent        → ValidationResult
        6. PlannerAgent          → Plan
        7. ActuatorExecutorAgent → ExecutionResult
        8. LoggerAgent           → MemoryRecord + CSV

    Returns the ExecutionResult for the EnergyPlus plugin to apply.
    """

    def __init__(
        self,
        log_file_path:   str,
        memory_capacity: int = 12,
    ):
        # Resolve memory file path adjacent to the log directory
        log_dir = os.path.dirname(log_file_path)
        memory_path = os.path.join(log_dir, "agent_memory.json")

        self.memory   = ShortTermMemory(capacity=memory_capacity, persist_path=memory_path)
        self.log_path = log_file_path
        self.log_dir  = log_dir

        # Specialist agents (stateless — no shared mutable state between agents)
        self._energy_agent      = EnergyOptimizerAgent()
        self._comfort_agent     = ComfortOptimizerAgent()
        self._economizer        = EconomizerModule()
        self._demand_response   = DemandResponseModule()
        epw_p = os.path.join(log_dir, "..", "weather", "IND_KA_Bengaluru.432950_ISHRAE2014.epw")
        self._predictive        = PredictivePrecoolModule(epw_p)
        self._validator         = ValidatorAgent()
        self._planner           = PlannerAgent()
        self._executor          = ActuatorExecutorAgent()
        self._logger            = LoggerAgent()

        # Resilience manager (health monitor, circuit breakers, sensor guard)
        self._resilience: "ResilienceManager | None" = None
        if _RESILIENCE_AVAILABLE and ResilienceManager:
            try:
                self._resilience = ResilienceManager(log_dir)
            except Exception:
                self._resilience = None

        # Performance Tracer
        from performance_tracer import PerformanceTracer
        self.tracer = PerformanceTracer(log_dir)

        # Cooldown state (owned by coordinator, affects validator)
        self._cooldown_remaining = 0

    # ── Main Entry Point ──────────────────────────────────────────────────────

    def run_cycle(
        self,
        zone_temp:       float,
        heating_sp:      float,
        cooling_sp:      float,
        outdoor_temp:    float,
        timestamp:       str,
        hour:            int,
        occupancy:       float = 1.0,
        hvac_mode:       str   = "auto",
        comfort_pmv:     Optional[float] = None,
        via_mcp:         bool  = False,
    ) -> ExecutionResult:
        """
        Execute one complete multi-agent control cycle with full performance tracing.
        Returns ExecutionResult with clamped coil speed ready to apply.
        """
        self.tracer.start_cycle()

        # ── Step 1: Build Observation Context (memory-aware) ──────────────
        with self.tracer.stage("memory_loading"):
            recent_energy = self.memory.recent_energy_total(6)
            hist_actions  = self.memory.action_sequence()
            hist_speeds   = self.memory.speed_sequence()

        obs = ObservationContext(
            zone_temp           = zone_temp,
            heating_sp          = heating_sp,
            cooling_sp          = cooling_sp,
            outdoor_temp        = outdoor_temp,
            timestamp           = timestamp,
            time_of_day_hour    = hour,
            occupancy           = occupancy,
            hvac_mode           = hvac_mode,
            recent_energy_kwh   = recent_energy,
            comfort_pmv         = comfort_pmv,
            historical_actions  = hist_actions,
            historical_speeds   = hist_speeds,
        )
        obs._tracer = self.tracer

        ctx = AgentContext(obs=obs, memory=self.memory)

        # ── Step 2: Energy Optimizer ──────────────────────────────────────
        with self.tracer.stage("energy_optimizer"):
            ctx.energy_rec = self._energy_agent.process(ctx)

        # ── Step 3: Comfort Optimizer ─────────────────────────────────────
        with self.tracer.stage("comfort_optimizer"):
            ctx.comfort_rec = self._comfort_agent.process(ctx)

        # ── Step 3b: Economizer Advisory Agent ────────────────────────────
        with self.tracer.stage("economizer_agent"):
            ctx.economizer_rec = self._economizer.evaluate(obs)
            if ctx.economizer_rec:
                e_rec = ctx.economizer_rec
                obs._economizer_note = (
                    f"Recommended Mode   : {e_rec.recommended_mode}\n"
                    f"Economizer Active  : {e_rec.economizer_active}\n"
                    f"Temp Advantage     : +{e_rec.temperature_advantage:.2f}°C (Zone {obs.zone_temp:.2f}°C vs Outdoor {obs.outdoor_temp:.2f}°C)\n"
                    f"Confidence         : {e_rec.confidence:.2f}\n"
                    f"Est. Energy Saved  : {e_rec.estimated_energy_saved_kwh:.3f} kWh ({e_rec.estimated_runtime_saved_hours:.1f}h compressor runtime avoided)\n"
                    f"Reason             : {e_rec.reason}"
                )

        # ── Step 3c: Demand Response / Peak Tariff Advisory Agent ──────────
        with self.tracer.stage("demand_response_agent"):
            ctx.dr_rec = self._demand_response.evaluate(obs)
            if ctx.dr_rec:
                dr_r = ctx.dr_rec
                obs._dr_note = (
                    f"Peak Tariff Window : {dr_r.is_peak_window}\n"
                    f"Tariff Period      : {dr_r.current_tariff_period} (₹{dr_r.current_tariff_inr_kwh:.2f}/kWh)\n"
                    f"Action Bias        : {dr_r.recommended_action_bias}\n"
                    f"Est. Cost Saved    : ₹{dr_r.estimated_cost_saved_inr:.2f}\n"
                    f"Reason             : {dr_r.reason}"
                )

        # ── Step 3d: Predictive Pre-Cooling Advisory Agent ───────────────
        with self.tracer.stage("predictive_agent"):
            ctx.precool_rec = self._predictive.evaluate(obs)
            if ctx.precool_rec:
                pc_r = ctx.precool_rec
                obs._precool_note = (
                    f"Precool Recommended: {pc_r.precool_recommended}\n"
                    f"EPW 3h Peak Outdoor : {pc_r.predicted_peak_outdoor_temp:.1f}°C\n"
                    f"Target Precool SP   : {pc_r.target_precool_sp:.1f}°C\n"
                    f"Est Peak Reduc (W)  : {pc_r.estimated_peak_reduction_w:.1f} W\n"
                    f"Reason              : {pc_r.reason}"
                )

        # ── Step 4: Confidence Engine (pre-compute prior) ─────────────────
        with self.tracer.stage("confidence_engine"):
            ctx.confidence = ConfidenceEngine.compute(self.memory, obs)

        # ── Step 5: Planning Agent (ONE LLM call per cycle) ───────────────
        with self.tracer.stage("planning_agent"):
            enriched_obs = self._enrich_observation(obs, ctx)
            ctx.planner_decision = self._call_planner(enriched_obs, ctx.confidence)

        pd = ctx.planner_decision
        ctx.llm_decision = ReasoningDecision(
            action           = pd.chosen_action,
            coil_speed       = pd.coil_speed,
            reasoning        = pd.reasoning,
            expected_energy_impact  = pd.expected_energy_impact,
            expected_comfort_impact = pd.expected_comfort_impact,
            confidence_score = pd.confidence_score,
            rationale        = pd.rationale,
            ok               = pd.ok,
        )

        # ── Step 6: Validator ─────────────────────────────────────────────
        with self.tracer.stage("validator"):
            ctx.validation = self._validator.process(ctx, self._cooldown_remaining > 0)

        # ── Step 7: Planner Execution Strategy ───────────────────────────
        ctx.plan = self._planner.process(ctx)
        llm_ok  = ctx.llm_decision.ok if ctx.llm_decision else False
        result  = self._executor.process(ctx.plan, llm_ok)
        ctx.result = result

        # ── Step 8: Logger ────────────────────────────────────────────────
        violations = ctx.validation.violations if ctx.validation else []
        outcome    = self._resolve_outcome(ctx)
        with self.tracer.stage("decision_log_write"):
            self._logger.process(ctx, result, outcome, violations, via_mcp, self.log_path)

        # ── Step 9: Explanation Engine ───────────────────────────────────
        with self.tracer.stage("explanation_engine"):
            if _EXPLANATION_AVAILABLE and ExplanationEngine:
                try:
                    explanation = ExplanationEngine.generate(ctx, result, outcome)
                    ExplanationEngine.store(explanation, self.log_dir)
                    if self._resilience:
                        self._resilience.record_explanation_stored()
                except Exception:
                    pass

        # ── Step 10: Memory Save ──────────────────────────────────────────
        with self.tracer.stage("memory_save"):
            pass  # Memory append & disk save happens inside Coordinator/Logger

        # ── Step 11: Resilience health update ─────────────────────────────
        if self._resilience:
            try:
                self._resilience.record_actuator_set(result.clamped_speed)
            except Exception:
                pass

        # Finalize tracer for this cycle
        circuit_is_open = bool(self._resilience and not self._resilience.llm_available())
        conf_val = ctx.confidence.total if ctx.confidence else 0.0
        self.tracer.finish_cycle(
            timestamp=timestamp,
            confidence=conf_val,
            outcome=outcome,
            llm_ok=llm_ok,
            circuit_open=circuit_is_open,
        )

        # Update cooldown
        if violations and any("ROLLED_BACK" in outcome or v == "ENERGY_SURGE" for v in violations):
            self._cooldown_remaining = COOLDOWN_CYCLES
        elif self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        return result

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _enrich_observation(
        self, obs: ObservationContext, ctx: AgentContext
    ) -> ObservationContext:
        """
        Inject agent analysis results and memory summary into a variant of the
        observation that will be passed to the LLM. We do this by appending
        the summaries to the system context inside the reasoning agent call.
        We store them as attributes that to_prompt_block() can optionally use.
        """
        # Attach extra context as dynamic attributes (used by enriched prompt)
        import copy
        enriched = copy.copy(obs)

        # We pass memory summary and agent notes via the historical context fields.
        # The reasoning_agent's to_prompt_block() already renders historical_actions
        # and historical_speeds; we prepend the memory summary as additional context
        # by embedding it in a subclassed observation.
        enriched._memory_summary = self.memory.summarize_for_prompt(6)
        enriched._energy_note = (
            f"EnergyOptimizer: {ctx.energy_rec.rationale} "
            f"[max_speed={ctx.energy_rec.max_allowed_speed}, trend={ctx.energy_rec.trend}]"
            if ctx.energy_rec else ""
        )
        enriched._comfort_note = (
            f"ComfortOptimizer: {ctx.comfort_rec.rationale}"
            if ctx.comfort_rec else ""
        )
        return enriched

    def _call_planner(self, obs, confidence: ConfidenceBreakdown) -> PlannerDecision:
        """
        Call the planning agent with multi-candidate evaluation.
        Records LLM success/failure on the ResilienceManager circuit breaker.
        """
        # Check circuit breaker before attempting LLM call
        if self._resilience and not self._resilience.llm_available():
            from planning_agent import _safety_planner
            result = _safety_planner(obs, confidence, reason="Circuit breaker OPEN — Ollama service unavailable")
            return result

        try:
            result = get_planning_decision(
                EnrichedObservationContext(obs), confidence, retries=1
            )
            if self._resilience:
                if result.ok:
                    self._resilience.record_llm_success()
                else:
                    self._resilience.record_llm_failure("LLM returned safety fallback")
            return result
        except Exception as exc:
            if self._resilience:
                self._resilience.record_llm_failure(str(exc))
            from planning_agent import _safety_planner
            return _safety_planner(obs, confidence, reason=str(exc))

    @staticmethod
    def _resolve_outcome(ctx: AgentContext) -> str:
        if not ctx.llm_decision or not ctx.llm_decision.ok:
            return "FALLBACK"
        if ctx.validation and not ctx.validation.approved:
            return "CORRECTED"
        return "SUCCESS"


# ─────────────────────────────────────────────────────────────────────────────
# Enriched Observation Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class EnrichedObservationContext:
    """
    Thin wrapper around ObservationContext that overrides to_prompt_block()
    to prepend memory summary and agent analysis notes.
    All attribute access falls through to the wrapped object.
    """

    def __init__(self, obs: ObservationContext):
        self._obs = obs

    def __getattr__(self, name):
        return getattr(self._obs, name)

    def to_prompt_block(self) -> str:
        base = self._obs.to_prompt_block()
        sections = []

        memory_summary = getattr(self._obs, "_memory_summary", "")
        if memory_summary:
            sections.append(memory_summary)

        energy_note = getattr(self._obs, "_energy_note", "")
        if energy_note:
            sections.append(f"=== ENERGY OPTIMIZER ANALYSIS ===\n{energy_note}")

        comfort_note = getattr(self._obs, "_comfort_note", "")
        if comfort_note:
            sections.append(f"=== COMFORT OPTIMIZER ANALYSIS ===\n{comfort_note}")

        if sections:
            return "\n\n".join(sections) + "\n\n" + base
        return base
