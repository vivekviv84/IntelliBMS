# EnergyPlus, Copyright (c) 1996-present, The Board of Trustees of the
# University of Illinois, The Regents of the University of California, through
# Lawrence Berkeley National Laboratory (subject to receipt of any required
# approvals from the U.S. Dept. of Energy), Oak Ridge National Laboratory,
# managed by UT-Battelle, Alliance for Energy Innovation, LLC, and other
# contributors. All rights reserved.

"""
ecoloop_plugin.py — EnergyPlus Python Plugin (Phase 5: Multi-Agent + Memory)

Wires the CoordinatorAgent (7-agent pipeline + persistent short-term memory)
into the EnergyPlus HVAC iteration callback. One LLM call per simulated hour;
all sub-iterations reuse the cached ExecutionResult.

Control architecture priority:
    Path A: CoordinatorAgent  (full 7-agent pipeline, memory-aware reasoning)
    Path B: ClosedLoopController  (single-agent fallback, no memory)
    Path C: Hard rule-based safety  (last resort)
"""

import csv
import os
import sys
import time

# ─────────────────────────────────────────────────────────────────────────────
# sys.path bootstrap — must precede all local imports
# ─────────────────────────────────────────────────────────────────────────────

import site

_SITE_PACKAGES = site.getsitepackages() + [site.getusersitepackages()]
_PLUGIN_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_PLUGIN_DIR, ".."))
_REAL_PLUGINS = os.path.join(_PROJECT_ROOT, "plugins")
_MCP_DIR      = os.path.join(_PROJECT_ROOT, "mcp_server")
_MODELS_DIR   = os.path.join(_PROJECT_ROOT, "models")

for _p in _SITE_PACKAGES + [_REAL_PLUGINS, _PLUGIN_DIR, _MCP_DIR, _MODELS_DIR, _PROJECT_ROOT]:
    if _p and os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# ─────────────────────────────────────────────────────────────────────────────
# Import multi-agent coordinator (primary path)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from agent_system import CoordinatorAgent
    _COORDINATOR_AVAILABLE = True
except ImportError as _ce:
    CoordinatorAgent = None
    _COORDINATOR_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Import closed-loop controller (single-agent fallback)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from closed_loop_controller import ClosedLoopController
    _CLC_AVAILABLE = True
except ImportError:
    ClosedLoopController = None
    _CLC_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Action map for rule-based safety fallback
# ─────────────────────────────────────────────────────────────────────────────

try:
    from reasoning_agent import ACTION_TO_SPEED
except ImportError:
    ACTION_TO_SPEED = {"off": 0.0, "eco": 1.0, "normal": 1.3, "boost": 1.9}

# ─────────────────────────────────────────────────────────────────────────────
# Legacy MCP imports (kept for compatibility; used only if coordinator fails)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from state_bridge import write_zone_state, read_actuator_command
    from mcp_agent import get_mcp_llm_decision
    _MCP_AVAILABLE = True
except ImportError:
    write_zone_state = read_actuator_command = get_mcp_llm_decision = None
    _MCP_AVAILABLE = False

from pyenergyplus.plugin import EnergyPlusPlugin


def _clamp(value, lo: float = 0.0, hi: float = 2.0) -> float:
    if value is None:
        return 1.0
    return max(lo, min(hi, float(value)))


class CoilSpeedControl(EnergyPlusPlugin):
    """
    EnergyPlus plugin entry point.

    Instantiates CoordinatorAgent once on first use (lazy init to avoid
    crashing before api_data_fully_ready). Runs the 7-agent pipeline once
    per simulated hour; reuses the cached result on sub-iterations.
    """

    def __init__(self):
        super().__init__()
        self.need_to_get_handles = True

        # Variable/actuator handles
        self.zone_air_temp_handle            = None
        self.heating_setpoint_handle         = None
        self.cooling_setpoint_handle         = None
        self.outdoor_temp_handle             = None
        self.coil_speed_level_handle         = None
        self.coil_speed_override_report_handle = None

        # Hourly throttle state
        self.last_sim_hour_key = None
        self.last_result       = None          # ExecutionResult (or dict for legacy paths)

        # Log path
        self.log_file_path = os.environ.get(
            "DECISION_LOG_PATH",
            os.path.join(_PROJECT_ROOT, "logs", "decision_log.csv")
        )

        # Controllers — lazy-initialized after first successful handle acquisition
        self._coordinator: "CoordinatorAgent | None"          = None
        self._clc:         "ClosedLoopController | None"       = None

    # ─────────────────────────────────────────────────────────────────────────
    # EnergyPlus HVAC Callback
    # ─────────────────────────────────────────────────────────────────────────

    def on_inside_hvac_system_iteration_loop(self, state) -> int:
        try:
            if not self.api.exchange.api_data_fully_ready(state):
                return 0

            # Acquire handles once
            if self.need_to_get_handles:
                self._init_handles(state)
                self._init_controllers()
                self.need_to_get_handles = False

            t_ep_0 = time.perf_counter()

            # Read sensors
            t_sens_0 = time.perf_counter()
            zone_temp    = self.api.exchange.get_variable_value(state, self.zone_air_temp_handle)
            heating_sp   = self.api.exchange.get_variable_value(state, self.heating_setpoint_handle)
            cooling_sp   = self.api.exchange.get_variable_value(state, self.cooling_setpoint_handle)
            outdoor_temp = self._read_outdoor_temp(state)
            dt_sens_ms   = (time.perf_counter() - t_sens_0) * 1000.0

            month  = self.api.exchange.month(state)
            day    = self.api.exchange.day_of_month(state)
            hour   = self.api.exchange.hour(state)
            minute = self.api.exchange.minutes(state)
            print(f"[CALLBACK] month={month}, day={day}, hour={hour}, min={minute}, zone_temp={zone_temp}")

            timestamp        = f"{month:02d}/{day:02d} {hour:02d}:{minute:02d}"
            current_hour_key = (month, day, hour)

            # MCP Synchronization
            t_mcp_0 = time.perf_counter()
            if _MCP_AVAILABLE and write_zone_state:
                try:
                    write_zone_state(zone_temp, heating_sp, cooling_sp, outdoor_temp, timestamp, hour)
                except Exception:
                    pass
            dt_mcp_ms = (time.perf_counter() - t_mcp_0) * 1000.0

            # One decision per simulated hour (reuse on sub-iterations)
            if current_hour_key != self.last_sim_hour_key or self.last_result is None:
                self.last_result = self._dispatch(
                    zone_temp, heating_sp, cooling_sp, outdoor_temp, timestamp, hour
                )
                self.last_sim_hour_key = current_hour_key

                # Record ep_callback, sensor_collection, and mcp_sync on tracer if coordinator present
                if self._coordinator and hasattr(self._coordinator, "tracer"):
                    tr = self._coordinator.tracer
                    tr.record_stage_ms("ep_callback", (time.perf_counter() - t_ep_0) * 1000.0)
                    tr.record_stage_ms("sensor_collection", dt_sens_ms)
                    tr.record_stage_ms("mcp_sync", dt_mcp_ms)

            # Apply actuator
            t_act_0 = time.perf_counter()
            self._apply(state, self.last_result)
            dt_act_ms = (time.perf_counter() - t_act_0) * 1000.0

            if self._coordinator and hasattr(self._coordinator, "tracer"):
                self._coordinator.tracer.record_stage_ms("actuator_update", dt_act_ms)

            return 0

        except Exception as exc:
            print(f"[CALLBACK ERROR] Exception in iteration loop: {exc}")
            import traceback
            traceback.print_exc()
            self._safe_reset(state)
            return 0

    # ─────────────────────────────────────────────────────────────────────────
    # Decision Dispatch
    # ─────────────────────────────────────────────────────────────────────────

    def _dispatch(
        self,
        zone_temp: float, heating_sp: float, cooling_sp: float,
        outdoor_temp: float, timestamp: str, hour: int,
    ) -> dict:
        print(f"[DISPATCH] Hourly decision node @ {timestamp}")
        """
        Priority dispatch:
            A → CoordinatorAgent (7-agent pipeline + memory)
            B → ClosedLoopController (reasoning agent only)
            C → Rule-based safety fallback
        """

        # ── PATH A: Multi-Agent Coordinator ──────────────────────────────
        if self._coordinator is not None:
            try:
                result = self._coordinator.run_cycle(
                    zone_temp=zone_temp,
                    heating_sp=heating_sp,
                    cooling_sp=cooling_sp,
                    outdoor_temp=outdoor_temp,
                    timestamp=timestamp,
                    hour=hour,
                    occupancy=1.0,
                    hvac_mode="auto",
                    comfort_pmv=None,
                    via_mcp=False,
                )
                return {
                    "action":                  result.action,
                    "coil_speed":              result.clamped_speed,
                    "rationale":               result.rationale,
                    "ok":                      result.ok,
                }
            except Exception as exc:
                print(f"[PATH A ERROR] Coordinator.run_cycle failed @ {timestamp}: {exc}")
                import traceback
                traceback.print_exc()
                pass    # Fall through to Path B

        # ── PATH B: Closed-Loop Controller ────────────────────────────────
        if self._clc is not None:
            try:
                dec = self._clc.run_cycle(
                    zone_temp=zone_temp, heating_sp=heating_sp,
                    cooling_sp=cooling_sp, outdoor_temp=outdoor_temp,
                    timestamp=timestamp, hour=hour,
                )
                return {
                    "action":     dec.action,
                    "coil_speed": dec.coil_speed,
                    "rationale":  dec.rationale,
                    "ok":         dec.ok,
                }
            except Exception:
                pass

        # ── PATH C: Rule-based safety fallback ───────────────────────────
        return self._rule_fallback(zone_temp, heating_sp, cooling_sp, outdoor_temp, timestamp)

    # ─────────────────────────────────────────────────────────────────────────
    # Actuator Application
    # ─────────────────────────────────────────────────────────────────────────

    def _apply(self, state, result: dict) -> None:
        if result and result.get("ok"):
            speed = _clamp(result.get("coil_speed"))
            self.api.exchange.set_actuator_value(state, self.coil_speed_level_handle, speed)
            self.api.exchange.set_global_value(
                state, self.coil_speed_override_report_handle,
                1.0 if speed > 0.0 else 0.0
            )
        else:
            self.api.exchange.reset_actuator(state, self.coil_speed_level_handle)
            self.api.exchange.set_global_value(state, self.coil_speed_override_report_handle, 0.0)

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _init_handles(self, state) -> None:
        self.zone_air_temp_handle = self.api.exchange.get_variable_handle(
            state, "Zone Air Temperature", "LIVING ZONE"
        )
        self.heating_setpoint_handle = self.api.exchange.get_variable_handle(
            state, "Zone Thermostat Heating Setpoint Temperature", "LIVING ZONE"
        )
        self.cooling_setpoint_handle = self.api.exchange.get_variable_handle(
            state, "Zone Thermostat Cooling Setpoint Temperature", "LIVING ZONE"
        )
        for key in ["Environment", "ENVIRONMENT", "Site", ""]:
            h = self.api.exchange.get_variable_handle(
                state, "Site Outdoor Air Drybulb Temperature", key
            )
            if h >= 0:
                self.outdoor_temp_handle = h
                break
        self.coil_speed_level_handle = self.api.exchange.get_actuator_handle(
            state, "Coil Speed Control", "Unitary System DX Coil Speed Value", "TWOSPEED HEAT PUMP 1"
        )
        self.coil_speed_override_report_handle = self.api.exchange.get_global_handle(
            state, "CoilSpeedLevelOverrideReport"
        )
        print(f"[PLUGIN INIT] Handles acquired -- ZoneTemp: {self.zone_air_temp_handle}, HeatingSP: {self.heating_setpoint_handle}, CoolingSP: {self.cooling_setpoint_handle}, CoilSpeed: {self.coil_speed_level_handle}")

    def _init_controllers(self) -> None:
        """Lazy initialization after handles are confirmed valid."""
        if _COORDINATOR_AVAILABLE and self._coordinator is None:
            try:
                self._coordinator = CoordinatorAgent(log_file_path=self.log_file_path)
            except Exception as exc:
                print("Failed to initialize CoordinatorAgent:", exc)
                import traceback
                traceback.print_exc()
                self._coordinator = None

        print(f"[CONTROLLER INIT] Coordinator status: {self._coordinator is not None}, CLC status: {self._clc is not None}")

        if _CLC_AVAILABLE and self._clc is None and self._coordinator is None:
            try:
                self._clc = ClosedLoopController(log_file_path=self.log_file_path)
            except Exception:
                self._clc = None

    def _read_outdoor_temp(self, state) -> float:
        if self.outdoor_temp_handle is not None and self.outdoor_temp_handle >= 0:
            return self.api.exchange.get_variable_value(state, self.outdoor_temp_handle)
        try:
            hr = self.api.exchange.hour(state)
            return self.api.exchange.today_weather_outdoor_dry_bulb_at_time(state, hr, 1)
        except Exception:
            return 20.0

    def _rule_fallback(
        self, zone_temp, heating_sp, cooling_sp, outdoor_temp, timestamp
    ) -> dict:
        cooling_dev = max(0.0, zone_temp - cooling_sp)
        heating_dev = max(0.0, heating_sp - zone_temp)
        dev = max(cooling_dev, heating_dev)
        action = "off" if dev == 0.0 else "eco" if dev < 0.5 else "normal" if dev < 1.5 else "boost"
        result = {
            "action": action,
            "coil_speed": ACTION_TO_SPEED[action],
            "rationale": f"RULE_FALLBACK: {action} for dev={dev:.2f}C at {timestamp}",
            "ok": False,
        }
        self._write_fallback_log(timestamp, zone_temp, heating_sp, cooling_sp, outdoor_temp, result)
        return result

    def _write_fallback_log(self, ts, zt, hsp, csp, oat, result) -> None:
        try:
            os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
            header_needed = not os.path.exists(self.log_file_path)
            with open(self.log_file_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if header_needed:
                    w.writerow([
                        "timestamp", "zone_temp", "heating_sp", "cooling_sp",
                        "outdoor_temp", "action", "coil_speed", "confidence",
                        "reasoning", "energy_kwh", "comfort_deviation",
                        "outcome", "success", "violations", "via_mcp"
                    ])
                dev = max(max(0.0, zt - csp), max(0.0, hsp - zt))
                w.writerow([
                    ts, f"{zt:.3f}", f"{hsp:.3f}", f"{csp:.3f}", f"{oat:.3f}",
                    result["action"], f"{result['coil_speed']:.2f}",
                    "0.400", result["rationale"],
                    "0.000000", f"{dev:.3f}",
                    "FALLBACK", False, "", False,
                ])
        except Exception:
            pass

    def _safe_reset(self, state) -> None:
        try:
            if self.coil_speed_level_handle is not None:
                self.api.exchange.reset_actuator(state, self.coil_speed_level_handle)
            if self.coil_speed_override_report_handle is not None:
                self.api.exchange.set_global_value(state, self.coil_speed_override_report_handle, 0.0)
        except Exception:
            pass
