import json
import os
import time
import urllib.request
import urllib.error

ACTION_TO_SPEED = {
    "off": 0.0,
    "eco": 1.0,
    "normal": 1.3,
    "boost": 1.9
}


def build_rationale(action: str, dev_from_cooling: float, dev_from_heating: float) -> str:
    if action == "off":
        return f"Zone temp is within comfort band (no active correction needed)."
    if dev_from_cooling > 0:
        return f"Zone is {dev_from_cooling:.2f}C above cooling setpoint, {action} cooling applied."
    if dev_from_heating > 0:
        return f"Zone is {dev_from_heating:.2f}C below heating setpoint, {action} heating applied."
    return f"{action} action applied."


SYSTEM_PROMPT = """You are an HVAC control agent for a building energy optimization system.
Given current zone conditions and deviation numbers, classify the correct control action to minimize energy use while keeping the zone within comfort bounds.

Definitions:
- cooling_deviation: degrees above cooling setpoint (max(0, zone_temp - cooling_sp)). >0 means zone is too hot.
- heating_deviation: degrees below heating setpoint (max(0, heating_sp - zone_temp)). >0 means zone is too cold.

Actions and meaning:
- "off": zone temp is safely within setpoints, no heating/cooling load needed
- "eco": zone temp is near the edge of the comfort band (0 to 0.5C deviation), use minimal correction
- "normal": zone temp is moderately outside setpoint (0.5C to 1.5C deviation), standard correction needed
- "boost": zone temp is significantly outside setpoint (>1.5C beyond it), maximum correction needed

Rules (apply exactly, based on the deviation numbers given):
- If both deviations are <= 0 (zone temp is between setpoints): action = "off"
- If a positive deviation is between 0 and 1.5: action = "eco" if <0.5, else "normal"
- If a positive deviation is > 1.5: action = "boost"
- Only one deviation can be positive at a time, or neither.

Examples:
Input: cooling_deviation=4.00, heating_deviation=0.00 -> {"action": "boost", "rationale": "Zone is 4.00C above cooling setpoint, boost cooling applied."}
Input: cooling_deviation=0.00, heating_deviation=4.00 -> {"action": "boost", "rationale": "Zone is 4.00C below heating setpoint, boost heating applied."}
Input: cooling_deviation=0.30, heating_deviation=0.00 -> {"action": "eco", "rationale": "Zone is 0.30C above cooling setpoint, eco cooling applied."}
Input: cooling_deviation=0.00, heating_deviation=0.00 -> {"action": "off", "rationale": "Zone temp is within comfort band (no active correction needed)."}

Respond with ONLY valid JSON, nothing else, in this exact format:
{"action": "off" | "eco" | "normal" | "boost", "rationale": "<one short sentence>"}"""


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

def get_llm_decision(zone_temp: float, heating_sp: float, cooling_sp: float,
                      outdoor_temp: float, retries: int = 1) -> dict:
    cooling_deviation = max(0.0, zone_temp - cooling_sp)
    heating_deviation = max(0.0, heating_sp - zone_temp)
    dev_from_cooling = zone_temp - cooling_sp
    dev_from_heating = heating_sp - zone_temp

    user_prompt = (
        f"Zone temp: {zone_temp:.2f}C, Heating setpoint: {heating_sp:.2f}C, "
        f"Cooling setpoint: {cooling_sp:.2f}C, Outdoor temp: {outdoor_temp:.2f}C. "
        f"cooling_deviation={cooling_deviation:.2f}, heating_deviation={heating_deviation:.2f}. "
        "Classify the action."
    )

    payload = {
        "model": "qwen2.5:3b",
        "system": SYSTEM_PROMPT,
        "prompt": user_prompt,
        "stream": False,
        "format": "json"
    }

    last_reason = "Unknown error"

    for attempt in range(1 + retries):
        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                OLLAMA_URL,
                data=req_data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_bytes = resp.read()

            resp_json = json.loads(resp_bytes.decode("utf-8"))
            raw_response = resp_json.get("response", "").strip()

            if raw_response.startswith("```"):
                lines = raw_response.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw_response = "\n".join(lines).strip()

            model_data = json.loads(raw_response)
            action = model_data.get("action")
            rationale = model_data.get("rationale")

            if action in ACTION_TO_SPEED:
                if not rationale:
                    rationale = build_rationale(action, dev_from_cooling, dev_from_heating)
                return {
                    "action": action,
                    "coil_speed": ACTION_TO_SPEED[action],
                    "rationale": rationale,
                    "ok": True
                }
            else:
                last_reason = f"Invalid action '{action}' returned by LLM"
        except Exception as e:
            last_reason = str(e)

    return {
        "action": None,
        "coil_speed": None,
        "rationale": f"LLM_FAILURE: {last_reason}",
        "ok": False
    }
