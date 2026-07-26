import json
import os
import time

BRIDGE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BRIDGE_DIR, "state.json")
COMMAND_FILE = os.path.join(BRIDGE_DIR, "command.json")


def _write_json_file(file_path: str, data: dict, max_retries: int = 5):
    for attempt in range(max_retries):
        try:
            temp_file = file_path + f".tmp.{os.getpid()}"
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(temp_file, file_path)
            return True
        except Exception:
            time.sleep(0.02)
    return False


def _read_json_file(file_path: str, max_retries: int = 5) -> dict:
    for attempt in range(max_retries):
        try:
            if not os.path.exists(file_path):
                return {}
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            time.sleep(0.02)
    return {}


def write_zone_state(data: dict) -> bool:
    """Write current zone state (zone_temp, heating_sp, cooling_sp, outdoor_temp) to state.json."""
    return _write_json_file(STATE_FILE, data)


def read_zone_state() -> dict:
    """Read current zone state from state.json."""
    return _read_json_file(STATE_FILE)


def write_actuator_command(data: dict) -> bool:
    """Write actuator command (action, coil_speed, rationale, ok) to command.json."""
    return _write_json_file(COMMAND_FILE, data)


def read_actuator_command() -> dict:
    """Read current actuator command decision from command.json."""
    return _read_json_file(COMMAND_FILE)
