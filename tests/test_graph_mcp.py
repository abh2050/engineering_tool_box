"""Tests covering LangGraph orchestration and MCP adapter behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.adapters import UnknownToolError, WorkbenchMCPAdapter
from core.graph import GraphConfig, build_graph
from core.history import HistoryManager


@pytest.fixture()
def temp_history(tmp_path: Path) -> HistoryManager:
    """History manager writing to an isolated on-disk database for tests."""

    db_path = tmp_path / "history.db"
    return HistoryManager(database_path=db_path)


@pytest.fixture()
def graph(temp_history: HistoryManager):
    config = GraphConfig(history=temp_history, enable_search=False)
    return build_graph(config)


@pytest.fixture()
def adapter(temp_history: HistoryManager):
    config = GraphConfig(history=temp_history, enable_search=False)
    return WorkbenchMCPAdapter.create(config=config)


def test_graph_invocation_runs_tool(graph):
    payload = {
        "user_query": "Compute pipe pressure drop",
        "selected_tool": "pipe_pressure_drop",
        "auto_route": False,
        "tool_inputs": {
            "volumetric_flow_rate": {"value": 0.012, "units": "m^3/s"},
            "diameter": {"value": 0.1, "units": "m"},
            "length": {"value": 50, "units": "m"},
            "roughness": {"value": 0.000045, "units": "m"},
            "density": {"value": 998, "units": "kg/m^3"},
            "dynamic_viscosity": {"value": 1e-3, "units": "Pa*s"},
        },
    }

    response = graph.invoke(payload)

    assert response["ok"] is True
    assert response["tool_name"] == "pipe_pressure_drop"
    assert "delta_p" in response["results"]
    assert response["history_id"] is not None
    assert response["metadata"]["tool_runtime_ms"] >= 0
    assert response["route"]["source"] == "user-selection"


def test_mcp_adapter_call_tool(adapter):
    result = adapter.call_tool(
        "beam_deflection",
        arguments={
            "load_case": "point_load_center",
            "length": {"value": 4.0, "units": "m"},
            "elastic_modulus": {"value": 200e9, "units": "Pa"},
            "moment_of_inertia": {"value": 8e-6, "units": "m^4"},
            "point_load": {"value": 10e3, "units": "N"},
        },
        user_query="Calculate beam deflection",
    )

    assert result.isError is False
    assert result.structuredContent["ok"] is True
    assert result.structuredContent["tool_name"] == "beam_deflection"
    assert result.structuredContent["history_id"] is not None
    assert any("Results" in chunk.text for chunk in result.content)


def test_mcp_adapter_unknown_tool(adapter):
    with pytest.raises(UnknownToolError):
        adapter.call_tool("nonexistent_tool", arguments={})


def test_mcp_adapter_lists_all_tools(adapter):
    listed = adapter.list_tools()
    tool_names = {tool.name for tool in listed.tools}
    assert {
        "pipe_pressure_drop",
        "beam_deflection",
        "pump_power_npsh",
        "hx_lmtd",
        "bolt_preload_torque",
    }.issubset(tool_names)