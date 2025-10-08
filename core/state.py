"""Shared state definitions for the LangGraph-powered workbench."""

from __future__ import annotations

from dataclasses import asdict, field
from datetime import UTC, datetime


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(UTC)
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass


@dataclass
class Substitution:
    """Representation of a symbol substitution inside a derivation step."""

    symbol: str
    value: float | None = None
    units: str | None = None
    expression: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Step:
    """Standardized computational step captured during tool execution."""

    index: int
    description: str
    equation_tex: str | None = None
    substitutions: List[Substitution] = field(default_factory=list)
    result_value: float | None = None
    result_units: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["substitutions"] = [sub.to_dict() for sub in self.substitutions]
        return data


class ToolInvocation(BaseModel):
    """Record of a single tool execution inside the graph."""

    tool_name: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    steps: List[Step] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None

    model_config = {"arbitrary_types_allowed": True}

    def to_serializable(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "steps": [step.to_dict() for step in self.steps],
            "warnings": self.warnings,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class WorkbenchState(BaseModel):
    """LangGraph state shared across nodes."""

    user_query: str | None = None
    selected_tool: str | None = None
    auto_route: bool = True
    inputs_normalized: Dict[str, Any] = Field(default_factory=dict)
    steps: List[Step] = Field(default_factory=list)
    results: Dict[str, Any] = Field(default_factory=dict)
    units: Dict[str, str] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    history_id: int | None = None
    tool_invocation: ToolInvocation | None = None

    model_config = {"arbitrary_types_allowed": True}

    def reset(self) -> None:
        """Clear transient state while preserving user intent selections."""

        self.inputs_normalized.clear()
        self.steps.clear()
        self.results.clear()
        self.units.clear()
        self.warnings.clear()
        self.tool_invocation = None

    def add_step(self, step: Step) -> None:
        """Append a step ensuring increasing index order."""

        if self.steps and step.index <= self.steps[-1].index:
            step.index = self.steps[-1].index + 1
        self.steps.append(step)

    def to_history_payload(self) -> Dict[str, Any]:
        """Prepare a JSON-serializable payload for persistence."""

        return {
            "user_query": self.user_query,
            "selected_tool": self.selected_tool,
            "auto_route": self.auto_route,
            "inputs_normalized": self.inputs_normalized,
            "steps": [step.to_dict() for step in self.steps],
            "results": self.results,
            "units": self.units,
            "warnings": self.warnings,
        }


def init_state() -> WorkbenchState:
    """Convenience factory used by the LangGraph graph builder."""

    return WorkbenchState()

