"""Model Context Protocol server exposing engineering tools."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from mcp.server import Server, stdio

from core.adapters import WorkbenchMCPAdapter
from core.graph import GraphConfig


logger = logging.getLogger(__name__)


def build_server(config: GraphConfig | None = None) -> tuple[Server, WorkbenchMCPAdapter]:
    """Create the MCP server and register tool handlers."""

    adapter = WorkbenchMCPAdapter.create(config=config)
    server = Server("engineering-workbench")

    @server.list_tools()
    async def _list_tools() -> Any:
        logger.debug("Listing tools")
        result = adapter.list_tools()
        return list(result.tools)

    @server.call_tool()
    async def _call_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
        logger.debug("Calling tool %s", tool_name)
        logger.debug("Raw arguments type=%s value=%s", type(arguments), arguments)
        arguments = arguments or {}
        user_query = None
        if isinstance(arguments, dict):
            user_query = arguments.pop("user_query", None)
        call_result = adapter.call_tool(
            tool_name,
            arguments=arguments,
            user_query=user_query,
        )
        logger.debug(
            "Call result ok=%s keys=%s",
            not call_result.isError,
            list((call_result.structuredContent or {}).keys()),
        )
        if call_result.isError:
            details = call_result.structuredContent or {"message": "Tool call failed"}
            raise RuntimeError(details)

        structured_payload = call_result.structuredContent or {}
        structured = structured_payload.get("results") if isinstance(structured_payload, dict) else None
        if not isinstance(structured, dict):
            structured = structured_payload if isinstance(structured_payload, dict) else {}
        if not structured:
            logger.warning("Structured content missing result fields; returning empty payload")
        unstructured = call_result.content or []
        return unstructured, structured

    return server, adapter


async def serve(config: GraphConfig | None = None) -> None:
    """Run the MCP server using stdio transport."""

    server, _ = build_server(config=config)
    init_options = server.create_initialization_options()

    async with stdio.stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, init_options)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve())
