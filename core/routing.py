"""Router utilities choosing the appropriate engineering tool."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from loguru import logger

try:  # pragma: no cover - optional dependency shim for environments missing python-dotenv
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - defensive guard
    load_dotenv = lambda *args, **kwargs: None  # type: ignore[assignment]

from .state import WorkbenchState


TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"


@dataclass(slots=True)
class ToolMetadata:
    """Metadata describing a calculational tool."""

    name: str
    description: str
    keywords: tuple[str, ...]
    schema: Dict[str, Any]

    @property
    def schema_for_openai(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.schema.get("input_schema", {}),
        }


@dataclass(slots=True)
class RouteDecision:
    """Outcome of the routing process."""

    tool_name: str
    confidence: float
    reason: str
    source: str


def load_tool_metadata() -> List[ToolMetadata]:
    """Load the five engineering tool schemas declared in ``tools/schemas``."""

    schemas_path = TOOLS_DIR / "schemas"
    registry: List[ToolMetadata] = []
    for schema_path in sorted(schemas_path.glob("*.json")):
        with schema_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        keywords = tuple(data.get("keywords", []))
        registry.append(
            ToolMetadata(
                name=data.get("name", schema_path.stem),
                description=data.get("description", ""),
                keywords=keywords,
                schema=data,
            )
        )
    return registry


def _get_openai_client():  # pragma: no cover - lazy import avoids optional dependency errors
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        logger.debug("OpenAI SDK not installed: %s", exc)
        return None

    project_env = Path(__file__).resolve().parents[2] / ".env"
    if project_env.exists():
        load_dotenv(dotenv_path=project_env, override=False)
    else:
        load_dotenv(override=False)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.debug("OPENAI_API_KEY not set; falling back to heuristic routing")
        return None

    logger.debug("OpenAI routing enabled via key from %s", project_env if project_env.exists() else "environment")
    return OpenAI(api_key=api_key)


class Router:
    """Selects the most appropriate tool via LLM-assisted or heuristic routing."""

    def __init__(
        self,
        tool_registry: Optional[Iterable[ToolMetadata]] = None,
        model: Optional[str] = None,
        client: Any | None = None,
    ) -> None:
        self.tool_registry = list(tool_registry or load_tool_metadata())
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.client = client or _get_openai_client()

    def route(self, state: WorkbenchState) -> RouteDecision:
        """Determine which tool should handle the current request."""

        if state.selected_tool:
            logger.debug("User selected tool %s; bypassing routing", state.selected_tool)
            return RouteDecision(
                tool_name=state.selected_tool,
                confidence=1.0,
                reason="User selection",
                source="user-selection",
            )

        if not state.user_query:
            raise ValueError("Routing requires a non-empty user query.")

        if self.client:
            decision = self._route_with_openai(state.user_query)
            if decision:
                return decision

        return self._route_with_keywords(state.user_query)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _route_with_openai(self, prompt: str) -> RouteDecision | None:
        """Invoke the OpenAI Chat Completions API with function calling."""

        try:
            # Convert tool schemas to OpenAI function format
            functions = []
            for meta in self.tool_registry:
                # Get the input schema and create a simplified version for OpenAI
                input_schema = meta.schema.get("input_schema", {})
                
                # Create a simplified schema that OpenAI can handle
                simplified_schema = {
                    "type": "object",
                    "properties": input_schema.get("properties", {}),
                    "required": input_schema.get("required", [])
                }
                
                # Remove unsupported features like allOf, anyOf, oneOf, etc.
                # These are used for complex validation but not needed for routing
                
                functions.append({
                    "type": "function",
                    "function": {
                        "name": meta.name,
                        "description": meta.description,
                        "parameters": simplified_schema
                    }
                })

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a tool router that MUST call exactly one function based on the user's query. Do not provide explanations or ask for more information - just call the most appropriate function. For pipe/pressure/flow questions, call pipe_pressure_drop. For beam/deflection/structural questions, call beam_deflection. For pump questions, call pump_power_npsh. For heat exchanger questions, call hx_lmtd. For bolt/torque questions, call bolt_preload_torque."},
                    {"role": "user", "content": prompt}
                ],
                tools=functions,
                tool_choice="required",  # Force function calling
                temperature=0.1
            )
            
            # Check if a function was called
            message = response.choices[0].message
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                return RouteDecision(
                    tool_name=tool_call.function.name,
                    confidence=0.9,
                    reason="OpenAI function calling",
                    source="openai",
                )
            
            return None
            
        except Exception as exc:  # pragma: no cover - external dependency
            logger.warning("OpenAI routing failed (%s); falling back to heuristics", exc)
            return None

    def _route_with_keywords(self, prompt: str) -> RouteDecision:
        """Simple keyword-based matching fallback."""

        prompt_lower = prompt.lower()
        best_score = -1
        best_metadata: ToolMetadata | None = None
        for metadata in self.tool_registry:
            score = 0
            for keyword in metadata.keywords:
                if keyword in prompt_lower:
                    score += 1
            if score > best_score:
                best_score = score
                best_metadata = metadata

        if best_metadata is None:
            raise RuntimeError("No tools registered; cannot route request.")

        confidence = max(0.2, min(1.0, 0.3 + 0.1 * best_score))
        return RouteDecision(
            tool_name=best_metadata.name,
            confidence=confidence,
            reason="Keyword heuristic match",
            source="heuristic",
        )


def route_request(user_query: str, selected_tool: str | None = None) -> RouteDecision:
    """Convenience wrapper for ad-hoc routing without a pre-built state."""

    state = WorkbenchState(user_query=user_query, selected_tool=selected_tool)
    router = Router()
    return router.route(state)

