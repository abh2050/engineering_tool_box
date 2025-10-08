#!/usr/bin/env python3
"""Minimal MCP client that spawns the local server and exercises a tool call."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from mcp.client import stdio
from mcp.client.session import ClientSession

REPO_ROOT = Path(__file__).resolve().parents[1]

SAMPLE_TOOL_ARGUMENTS: Dict[str, Dict[str, Any]] = {
    "pipe_pressure_drop": {
        "volumetric_flow_rate": {"value": 0.012, "units": "m^3/s"},
        "diameter": {"value": 0.1, "units": "m"},
        "length": {"value": 50, "units": "m"},
        "roughness": {"value": 0.000045, "units": "m"},
        "density": {"value": 998, "units": "kg/m^3"},
        "dynamic_viscosity": {"value": 1e-3, "units": "Pa*s"},
    },
    "beam_deflection": {
        "load_case": "point_load_center",
        "length": {"value": 4.0, "units": "m"},
        "elastic_modulus": {"value": 200e9, "units": "Pa"},
        "moment_of_inertia": {"value": 8e-6, "units": "m^4"},
        "point_load": {"value": 10e3, "units": "N"},
    },
}


async def run_client(tool: str | None, arguments_json: str | None) -> None:
    """Spawn the MCP server over stdio and optionally call a tool."""

    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
    else:
        load_dotenv(override=False)

    server_env = os.environ.copy()
    server_env.setdefault("PYTHONPATH", str(REPO_ROOT))

    params = stdio.StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server"],
        cwd=str(REPO_ROOT),
        env=server_env,
    )

    async with stdio.stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            init_result = await session.initialize()
            print(f"✅ Initialized MCP session (protocol {init_result.protocolVersion})")

            await session.send_ping()
            print("✅ Server responded to ping")

            tools_result = await session.list_tools()
            tool_names = ", ".join(tool.name for tool in tools_result.tools) or "<none>"
            print(f"🧰 Tools available: {tool_names}")

            if tool:
                arguments: Dict[str, Any]
                if arguments_json:
                    arguments = json.loads(arguments_json)
                else:
                    sample = SAMPLE_TOOL_ARGUMENTS.get(tool)
                    if sample is None:
                        raise SystemExit(
                            f"No sample arguments for tool '{tool}'. Provide --arguments with JSON payload."
                        )
                    arguments = sample

                print(f"⚙️ Calling tool '{tool}' with arguments:\n{json.dumps(arguments, indent=2)}")
                call_result = await session.call_tool(tool, arguments=arguments)

                if call_result.isError:
                    print("❌ Tool returned an error:")
                    print(json.dumps(call_result.structuredContent or {}, indent=2))
                else:
                    print("✅ Tool call succeeded. Structured content:")
                    print(json.dumps(call_result.structuredContent, indent=2, default=str))
                    if call_result.content:
                        print("📄 Text content:")
                        for chunk in call_result.content:
                            if chunk.text:
                                print(chunk.text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tool",
        type=str,
        default=None,
        help="Optional tool to call after initialization (e.g. pipe_pressure_drop)",
    )
    parser.add_argument(
        "--arguments",
        type=str,
        default=None,
        help="JSON payload with arguments for the tool. If omitted, sample data is used when available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run_client(args.tool, args.arguments))
    except KeyboardInterrupt:
        print("Interrupted by user")


if __name__ == "__main__":
    main()
