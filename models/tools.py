import os
import sys

# Ensure mcp_server directory is in sys.path
server_dir = os.path.dirname(os.path.abspath(__file__))
if server_dir not in sys.path:
    sys.path.insert(0, server_dir)

from state_bridge import read_zone_state, write_actuator_command

ACTION_TO_SPEED = {
    "off": 0.0,
    "eco": 1.0,
    "normal": 1.3,
    "boost": 1.9
}


def get_zone_state() -> dict:
    """Read and return current zone thermal state including zone_temp, heating_sp, cooling_sp, outdoor_temp."""
    return read_zone_state()


def set_actuator(action: str, rationale: str = "") -> dict:
    """Validate and set HVAC coil speed control action ('off', 'eco', 'normal', 'boost') with short rationale."""
    action = str(action).strip().lower()
    if action not in ACTION_TO_SPEED:
        err_res = {
            "status": "error",
            "action": None,
            "coil_speed": None,
            "rationale": f"LLM_FAILURE: Invalid action '{action}'",
            "ok": False
        }
        write_actuator_command(err_res)
        return err_res

    speed = ACTION_TO_SPEED[action]
    success_res = {
        "status": "success",
        "action": action,
        "coil_speed": speed,
        "rationale": rationale if rationale else f"Action '{action}' selected via MCP tool call.",
        "ok": True
    }
    write_actuator_command(success_res)
    return success_res


# Decorate with FastMCP if mcp package is present
try:
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("EcoLoopHVACServer")
    mcp.tool()(get_zone_state)
    mcp.tool()(set_actuator)
except Exception:
    mcp = None


def get_mcp_tool_definitions() -> list:
    """Return MCP function tool definitions formatted for Ollama / LLM API tool calls."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_zone_state",
                "description": "Read the current zone temperature, heating setpoint, cooling setpoint, and outdoor temperature.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "set_actuator",
                "description": "Set the HVAC control action to manage coil speed. Allowed actions: 'off', 'eco', 'normal', 'boost'.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["off", "eco", "normal", "boost"],
                            "description": "The classified HVAC action."
                        },
                        "rationale": {
                            "type": "string",
                            "description": "A concise one-sentence rationale for the action."
                        }
                    },
                    "required": ["action"]
                }
            }
        }
    ]


def handle_mcp_tool_call(name: str, args: dict) -> dict:
    """Route tool call execution to the corresponding MCP tool function."""
    if name == "get_zone_state":
        return get_zone_state()
    elif name == "set_actuator":
        return set_actuator(
            action=args.get("action", ""),
            rationale=args.get("rationale", "")
        )
    else:
        return {"status": "error", "message": f"Unknown tool: {name}"}


if __name__ == "__main__":
    print("MCP Tools defined:")
    for t in get_mcp_tool_definitions():
        print(" -", t["function"]["name"])
