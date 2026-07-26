# EnergyPlus, Copyright (c) 1996-present, The Board of Trustees of the
# University of Illinois, The Regents of the University of California, through
# Lawrence Berkeley National Laboratory (subject to receipt of any required
# approvals from the U.S. Dept. of Energy), Oak Ridge National Laboratory,
# managed by UT-Battelle, Alliance for Energy Innovation, LLC, and other
# contributors. All rights reserved.

import csv
import os
import sys

from pyenergyplus.plugin import EnergyPlusPlugin

plugin_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(plugin_dir, ".."))


class CoilSpeedControl(EnergyPlusPlugin):

    def __init__(self):
        super().__init__()
        self.need_to_get_handles = True
        self.zone_air_temp_handle = None
        self.heating_setpoint_handle = None
        self.cooling_setpoint_handle = None
        self.outdoor_temp_handle = None
        self.coil_speed_level_handle = None
        self.coil_speed_override_report_handle = None

        self.last_sim_hour_key = None
        self.log_file_path = os.path.join(project_root, "logs", "baseline_log.csv")

    def _ensure_log_file(self):
        try:
            log_dir = os.path.dirname(self.log_file_path)
            os.makedirs(log_dir, exist_ok=True)
            if not os.path.exists(self.log_file_path):
                with open(self.log_file_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "timestamp", "zone_temp", "heating_sp", "cooling_sp",
                        "outdoor_temp", "action", "coil_speed"
                    ])
        except Exception:
            pass

    def _log_baseline(self, timestamp_str, zone_temp, heating_sp, cooling_sp, outdoor_temp, action, coil_speed):
        try:
            self._ensure_log_file()
            with open(self.log_file_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp_str,
                    f"{zone_temp:.2f}",
                    f"{heating_sp:.2f}",
                    f"{cooling_sp:.2f}",
                    f"{outdoor_temp:.2f}",
                    action,
                    coil_speed
                ])
        except Exception:
            pass

    def on_inside_hvac_system_iteration_loop(self, state) -> int:
        if self.api.exchange.api_data_fully_ready(state):
            if self.need_to_get_handles:
                self.zone_air_temp_handle = self.api.exchange.get_variable_handle(
                    state, "Zone Air Temperature", "LIVING ZONE"
                )

                self.heating_setpoint_handle = self.api.exchange.get_variable_handle(
                    state, "Zone Thermostat Heating Setpoint Temperature", "LIVING ZONE"
                )

                self.cooling_setpoint_handle = self.api.exchange.get_variable_handle(
                    state, "Zone Thermostat Cooling Setpoint Temperature", "LIVING ZONE"
                )

                # Try finding outdoor temp handle safely
                for key in ["Environment", "ENVIRONMENT", "Site", ""]:
                    h = self.api.exchange.get_variable_handle(state, "Site Outdoor Air Drybulb Temperature", key)
                    if h >= 0:
                        self.outdoor_temp_handle = h
                        break

                self.coil_speed_level_handle = self.api.exchange.get_actuator_handle(
                    state, "Coil Speed Control", "Unitary System DX Coil Speed Value", "TWOSPEED HEAT PUMP 1"
                )

                self.coil_speed_override_report_handle = self.api.exchange.get_global_handle(
                    state, "CoilSpeedLevelOverrideReport"
                )

                self.need_to_get_handles = False

            # Read variables safely
            zone_air_temp = self.api.exchange.get_variable_value(state, self.zone_air_temp_handle)
            heating_setpoint = self.api.exchange.get_variable_value(state, self.heating_setpoint_handle)
            cooling_setpoint = self.api.exchange.get_variable_value(state, self.cooling_setpoint_handle)

            if self.outdoor_temp_handle is not None and self.outdoor_temp_handle >= 0:
                outdoor_temp = self.api.exchange.get_variable_value(state, self.outdoor_temp_handle)
            else:
                try:
                    hr = self.api.exchange.hour(state)
                    outdoor_temp = self.api.exchange.today_weather_outdoor_dry_bulb_at_time(state, hr, 1)
                except Exception:
                    outdoor_temp = 20.0

            # Control logic (unmodified rule-based baseline)
            if zone_air_temp < heating_setpoint:
                coil_speed = 1.9
                action = "boost_heating"
                self.api.exchange.set_actuator_value(state, self.coil_speed_level_handle, 1.9)
                self.api.exchange.set_global_value(state, self.coil_speed_override_report_handle, 1.0)
            elif zone_air_temp < heating_setpoint + 0.5:
                coil_speed = 1.0
                action = "eco_heating"
                self.api.exchange.set_actuator_value(state, self.coil_speed_level_handle, 1.0)
                self.api.exchange.set_global_value(state, self.coil_speed_override_report_handle, 1.0)
            elif zone_air_temp > cooling_setpoint + 1:
                coil_speed = 1.95
                action = "boost_cooling"
                self.api.exchange.set_actuator_value(state, self.coil_speed_level_handle, 1.95)
                self.api.exchange.set_global_value(state, self.coil_speed_override_report_handle, 1.0)
            elif zone_air_temp > cooling_setpoint:
                coil_speed = 1.0
                action = "eco_cooling"
                self.api.exchange.set_actuator_value(state, self.coil_speed_level_handle, 1.0)
                self.api.exchange.set_global_value(state, self.coil_speed_override_report_handle, 1.0)
            else:
                coil_speed = 0.0
                action = "off"
                self.api.exchange.reset_actuator(state, self.coil_speed_level_handle)
                self.api.exchange.set_global_value(state, self.coil_speed_override_report_handle, 0.0)

            # Log once per simulated hour
            month = self.api.exchange.month(state)
            day = self.api.exchange.day_of_month(state)
            hour = self.api.exchange.hour(state)
            minute = self.api.exchange.minutes(state)

            current_sim_hour_key = (month, day, hour)
            if current_sim_hour_key != self.last_sim_hour_key:
                timestamp_str = f"{month:02d}/{day:02d} {hour:02d}:{minute:02d}"
                self._log_baseline(timestamp_str, zone_air_temp, heating_setpoint, cooling_setpoint, outdoor_temp, action, coil_speed)
                self.last_sim_hour_key = current_sim_hour_key

            return 0

        else:
            return 0
