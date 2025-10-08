"""Adapters connecting the LangGraph workbench to MCP-compatible tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

from .graph import GraphConfig, GraphInput, WorkbenchGraph, build_graph
from .routing import ToolMetadata, load_tool_metadata


class UnknownToolError(ValueError):
    """Raised when a requested tool is not available in the registry."""


def _metadata_to_tool(meta: ToolMetadata) -> Tool:
    return Tool(
        name=meta.name,
        title=meta.name.replace("_", " ").title(),
        description=meta.description,
        inputSchema=meta.schema.get("input_schema", {}),
        outputSchema=meta.schema.get("output_schema"),
        _meta={"keywords": list(meta.keywords)},
    )


def _build_markdown(response: Dict[str, Any]) -> str:
    if response.get("markdown"):
        return str(response["markdown"])

    lines: List[str] = []
    tool_name = response.get("tool_name")
    if tool_name:
        lines.append(f"**Tool:** `{tool_name}`")

    results = response.get("results") or {}
    units = response.get("units") or {}
    if results:
        lines.append("")
        lines.append("**Results:**")
        for key, value in results.items():
            unit = units.get(key)
            if unit:
                lines.append(f"- {key}: {value} {unit}")
            else:
                lines.append(f"- {key}: {value}")

    warnings = response.get("warnings") or []
    if warnings:
        lines.append("")
        lines.append("**Warnings:**")
        for warning in warnings:
            lines.append(f"- {warning}")

    error = response.get("error")
    if error:
        lines.append("")
        message = error.get("message") if isinstance(error, dict) else str(error)
        lines.append(f"**Error:** {message}")

    text = "\n".join(lines).strip()
    return text or "No output produced."


@dataclass(slots=True)
class WorkbenchMCPAdapter:
    """Expose workbench tools over the Model Context Protocol interface."""

    graph: WorkbenchGraph
    metadata: List[ToolMetadata] = field(default_factory=list)
    _metadata_by_name: Dict[str, ToolMetadata] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._metadata_by_name = {meta.name: meta for meta in self.metadata}

    @classmethod
    def create(
        cls,
        *,
        config: GraphConfig | None = None,
        tool_metadata: Iterable[ToolMetadata] | None = None,
    ) -> "WorkbenchMCPAdapter":
        metadata_list = list(tool_metadata or load_tool_metadata())
        graph = build_graph(config)
        return cls(graph=graph, metadata=metadata_list)

    # ------------------------------------------------------------------
    # MCP surface area
    # ------------------------------------------------------------------

    def list_tools(self) -> ListToolsResult:
        """Return tool descriptors compatible with MCP `tools/list`."""

        return ListToolsResult(tools=[_metadata_to_tool(meta) for meta in self.metadata])

    def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any] | None = None,
        *,
        user_query: str | None = None,
        history_id: int | None = None,
    ) -> CallToolResult:
        """Execute a tool via the LangGraph workflow and wrap the response."""

        if tool_name not in self._metadata_by_name:
            raise UnknownToolError(f"Tool '{tool_name}' is not registered")

        payload: GraphInput = {
            "user_query": user_query or f"MCP invocation for {tool_name}",
            "selected_tool": tool_name,
            "auto_route": False,
            "tool_inputs": dict(arguments or {}),
            "history_id": history_id,
            "save_history": True,
        }

        response = self.graph.invoke(payload)

        text = _build_markdown(response)
        is_error = not response.get("ok", False)

        return CallToolResult(
            content=[TextContent(type="text", text=text)],
            structuredContent=response,
            isError=is_error,
        )


def register_tools(config: GraphConfig | None = None) -> WorkbenchMCPAdapter:
    """Factory returning an MCP adapter instance with registered tools."""

    return WorkbenchMCPAdapter.create(config=config)


__all__ = ["WorkbenchMCPAdapter", "register_tools", "UnknownToolError"]
