import json
import os
import sys
import time
import urllib.request
import urllib.error

# Ensure mcp_server directory is in sys.path
server_dir = os.path.dirname(os.path.abspath(__file__))
if server_dir not in sys.path:
    sys.path.insert(0, server_dir)

from state_bridge import read_zone_state, write_actuator_command, read_actuator_command
from tools import get_mcp_tool_definitions, handle_mcp_tool_call

OLLAMA_CHAT_URL = os.getenv("OLLAMA_CHAT_URL", "http://localhost:11434/api/chat")

MCP_SYSTEM_PROMPT = """You are an HVAC control agent operating via Model Context Protocol (MCP) tools for a building energy optimization system.

Your required workflow:
1. Always call `get_zone_state()` first to inspect current zone conditions and setpoints.
2. Based on the state:
   - cooling_deviation = zone_temp - cooling_sp
   - heating_deviation = heating_sp - zone_temp
   Classify action:
   - "off" if both deviations <= 0
   - "eco" if positive deviation between 0 and 0.5
   - "normal" if positive deviation between 0.5 and 1.5
   - "boost" if positive deviation > 1.5
3. Call `set_actuator(action, rationale)` with the classified action and short rationale."""


def get_mcp_llm_decision(retries: int = 1) -> dict:
    """Execute a genuine multi-step MCP tool-calling loop using Ollama local API."""
    last_reason = "Unknown MCP error"

    for attempt in range(1 + retries):
        try:
            tools_def = get_mcp_tool_definitions()
            messages = [
                {"role": "system", "content": MCP_SYSTEM_PROMPT},
                {"role": "user", "content": "Retrieve the current zone state using the get_zone_state tool, evaluate conditions, and set the actuator action using set_actuator."}
            ]

            # Multi-turn tool execution loop (max 4 turns)
            for turn in range(4):
                payload = {
                    "model": "qwen2.5:3b",
                    "messages": messages,
                    "tools": tools_def,
                    "stream": False
                }

                req_data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    OLLAMA_CHAT_URL,
                    data=req_data,
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    resp_bytes = resp.read()

                resp_json = json.loads(resp_bytes.decode("utf-8"))
                assistant_msg = resp_json.get("message", {})
                messages.append(assistant_msg)

                tool_calls = assistant_msg.get("tool_calls")
                if tool_calls:
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        t_name = fn.get("name")
                        t_args = fn.get("arguments", {})

                        t_res = handle_mcp_tool_call(t_name, t_args)

                        messages.append({
                            "role": "tool",
                            "name": t_name,
                            "content": json.dumps(t_res)
                        })

                        if t_name == "set_actuator" and t_res.get("ok"):
                            return t_res
                else:
                    break

            cmd_res = read_actuator_command()
            if cmd_res and cmd_res.get("ok"):
                return cmd_res

            last_reason = "MCP tool loop finished without set_actuator confirmation"
        except Exception as e:
            last_reason = str(e)

    return {
        "action": None,
        "coil_speed": None,
        "rationale": f"LLM_FAILURE: {last_reason}",
        "ok": False
    }


if __name__ == "__main__":
    print("=== Testing Standalone MCP Agent Tool Loop ===")
    from state_bridge import write_zone_state
    write_zone_state({
        "zone_temp": 25.2,
        "heating_sp": 20.0,
        "cooling_sp": 24.0,
        "outdoor_temp": 30.0,
        "timestamp": "07/01 12:00"
    })
    res = get_mcp_llm_decision()
    print("MCP Agent Decision Result:", res)
