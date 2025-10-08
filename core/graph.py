"""LangGraph orchestration utilities for the engineering workbench."""

from __future__ import annotations

import traceback
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib import import_module
from time import perf_counter, sleep
from typing import Any, Callable, Dict, List, TypedDict

from langgraph.graph import END, StateGraph
from loguru import logger

from .explainer import format_final_results, steps_to_markdown, summarize_warnings
from .history import HistoryManager
from .routing import RouteDecision, Router, load_tool_metadata
from .search import SearchIndex, SearchResult
from .state import ToolInvocation, WorkbenchState, init_state


ToolCallable = Callable[[Dict[str, Any]], Dict[str, Any]]


class GraphInput(TypedDict, total=False):
    """Payload accepted by the LangGraph workflow entrypoint."""

    user_query: str
    selected_tool: str | None
    auto_route: bool
    tool_inputs: Dict[str, Any]
    history_id: int | None
    save_history: bool


class GraphError(TypedDict, total=False):
    """Structured error information propagated through the graph state."""

    message: str
    details: str | None
    tool_name: str | None
    node: str | None


class GraphState(TypedDict, total=False):
    """Mutable state carried between LangGraph nodes."""

    workbench: WorkbenchState
    raw_inputs: Dict[str, Any]
    payload: GraphInput | None
    route_decision: RouteDecision | None
    response_markdown: str | None
    search_results: List[SearchResult]
    error: GraphError | None
    response: Dict[str, Any] | None
    metadata: Dict[str, Any] | None
    save_history: bool
    parameters_needed: Dict[str, Any] | None


@dataclass(slots=True)
class ToolRunnerRegistry:
    """Lazy loader for tool ``run`` callables within the ``tools`` package."""

    package: str = "tools"
    attribute: str = "run"
    _cache: Dict[str, ToolCallable] = field(default_factory=dict)

    def resolve(self, tool_name: str) -> ToolCallable:
        """Return the callable implementing the requested tool."""

        if tool_name in self._cache:
            return self._cache[tool_name]

        module_name = f"{self.package}.{tool_name}"
        try:
            module = import_module(module_name)
        except ModuleNotFoundError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Tool module '{module_name}' could not be imported") from exc

        candidate = getattr(module, self.attribute, None)
        if candidate is None:
            raise AttributeError(f"Tool module '{module_name}' lacks attribute '{self.attribute}'")
        if not callable(candidate):
            raise TypeError(
                f"Tool '{module_name}.{self.attribute}' must be callable, got {type(candidate).__name__}"
            )

        self._cache[tool_name] = candidate
        return candidate

    def registered_tools(self) -> List[str]:
        """Return tool names that have been resolved in this registry."""

        return sorted(self._cache.keys())


@dataclass(slots=True)
class RetryPolicy:
    """Simple retry configuration for tool execution."""

    max_attempts: int = 2
    backoff_seconds: float = 0.0
    retry_exceptions: tuple[type[BaseException], ...] = (Exception,)

    def should_retry(self, attempt: int, exc: BaseException) -> bool:
        return attempt < self.max_attempts and isinstance(exc, self.retry_exceptions)


@dataclass(slots=True)
class GraphConfig:
    """Dependency container used when constructing the LangGraph workflow."""

    router: Router = field(default_factory=Router)
    history: HistoryManager = field(default_factory=HistoryManager)
    search_index: SearchIndex | None = None
    tool_registry: ToolRunnerRegistry = field(default_factory=ToolRunnerRegistry)
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    enable_search: bool = True
    debug: bool = False

    def __post_init__(self) -> None:
        if self.search_index is None and self.enable_search:
            tool_metadata = load_tool_metadata()
            self.search_index = SearchIndex(tool_registry=tool_metadata, history_manager=self.history)
        elif self.search_index is not None and self.search_index.history_manager is not self.history:
            # Ensure the search index shares the same history manager so new runs surface in search results.
            self.search_index.history_manager = self.history


def default_state() -> GraphState:
    """Initialize the base graph state with an empty workbench container."""

    return GraphState(
        workbench=init_state(),
        raw_inputs={},
        payload=None,
        route_decision=None,
        response_markdown=None,
        search_results=[],
        error=None,
        response=None,
        metadata=None,
        save_history=True,
    )


def _has_error(state: GraphState) -> bool:
    return bool(state.get("error"))


def _format_error(config: GraphConfig, *, message: str, exc: Exception, node: str, tool_name: str | None = None) -> GraphError:
    details = str(exc)
    if config.debug:
        details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return {
        "message": message,
        "details": details,
        "node": node,
        "tool_name": tool_name,
    }


def _simple_error(*, message: str, node: str, tool_name: str | None = None, details: str | None = None) -> GraphError:
    return {
        "message": message,
        "details": details,
        "node": node,
        "tool_name": tool_name,
    }


@dataclass(slots=True)
class PrepareAgent:
    """Initialize or reset the working state for a new calculation run."""

    def __call__(self, state: GraphState) -> GraphState:
        payload = state.get("payload") or {}
        workbench = state["workbench"]
        workbench.reset()

        user_query = payload.get("user_query") if isinstance(payload, dict) else None
        selected_tool = payload.get("selected_tool") if isinstance(payload, dict) else None
        auto_route = payload.get("auto_route", True) if isinstance(payload, dict) else True
        history_id = payload.get("history_id") if isinstance(payload, dict) else None
        tool_inputs = payload.get("tool_inputs") if isinstance(payload, dict) else None
        save_history = payload.get("save_history", True) if isinstance(payload, dict) else True

        workbench.user_query = user_query
        workbench.selected_tool = selected_tool
        workbench.auto_route = bool(auto_route)
        workbench.history_id = history_id

        raw_inputs = dict(tool_inputs or {})

        return {
            "workbench": workbench,
            "raw_inputs": raw_inputs,
            "payload": None,
            "error": None,
            "route_decision": None,
            "response_markdown": None,
            "search_results": [],
            "metadata": None,
            "save_history": bool(save_history),
        }


@dataclass(slots=True)
class RouterAgent:
    config: GraphConfig

    def __call__(self, state: GraphState) -> GraphState:
        if _has_error(state):
            return {}

        workbench = state["workbench"]

        if not workbench.auto_route and not workbench.selected_tool:
            return {
                "error": _simple_error(
                    message="Select a tool or enable auto-route before running the calculation.",
                    node="router_agent",
                )
            }

        try:
            decision = self.config.router.route(workbench)
        except Exception as exc:  # pragma: no cover - depends on external API/network
            logger.exception("Routing failed")
            return {"error": _format_error(self.config, message="Routing failed", exc=exc, node="router_agent")}

        workbench.selected_tool = decision.tool_name
        return {
            "route_decision": decision,
            "workbench": workbench,
        }


@dataclass(slots=True)
class ParameterValidationAgent:
    config: GraphConfig

    def __call__(self, state: GraphState) -> GraphState:
        if _has_error(state):
            return {}

        workbench = state["workbench"]
        tool_name = workbench.selected_tool
        if not tool_name:
            return {
                "error": _simple_error(
                    message="No tool selected for parameter validation.",
                    node="parameter_validation_agent",
                )
            }

        raw_inputs = dict(state.get("raw_inputs", {}))

        # Get tool metadata to check required parameters
        from .routing import load_tool_metadata
        tool_metadata_list = load_tool_metadata()
        tool_metadata = None
        for meta in tool_metadata_list:
            if meta.name == tool_name:
                tool_metadata = meta
                break

        if not tool_metadata:
            return {
                "error": _simple_error(
                    message=f"Tool metadata not found for '{tool_name}'.",
                    node="parameter_validation_agent",
                )
            }

        # Extract schema information
        schema = tool_metadata.schema.get("input_schema", {})
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        
        # Handle conditional requirements (like beam_deflection's allOf)
        all_of = schema.get("allOf", [])
        for condition in all_of:
            if_clause = condition.get("if", {})
            then_clause = condition.get("then", {})
            
            # Check if condition is met
            if_properties = if_clause.get("properties", {})
            condition_met = True
            for prop_name, prop_condition in if_properties.items():
                prop_value = raw_inputs.get(prop_name)
                if prop_condition.get("const") and prop_value != prop_condition["const"]:
                    condition_met = False
                    break
            
            # If condition is met, add required fields from then clause
            if condition_met:
                then_required = then_clause.get("required", [])
                required.update(then_required)

        # Check for missing required parameters
        missing_params = []
        for param_name in required:
            if param_name not in raw_inputs or raw_inputs[param_name] is None:
                missing_params.append(param_name)

        # If parameters are missing, return a structured response for the UI
        if missing_params:
            missing_info = []
            for param_name in missing_params:
                param_schema = properties.get(param_name, {})
                param_info = {
                    "name": param_name,
                    "description": param_schema.get("description", ""),
                    "type": param_schema.get("type", ""),
                    "examples": param_schema.get("examples", []),
                    "properties": param_schema.get("properties", {}),
                }
                
                # Include enum values if present
                if "enum" in param_schema:
                    param_info["enum"] = param_schema["enum"]
                
                missing_info.append(param_info)

            return {
                "parameters_needed": {
                    "tool_name": tool_name,
                    "missing_parameters": missing_info,
                    "provided_inputs": raw_inputs,
                }
            }

        # All required parameters are present, proceed to tool execution
        return {"workbench": workbench, "raw_inputs": raw_inputs}


@dataclass(slots=True)
class ToolAgent:
    config: GraphConfig

    def __call__(self, state: GraphState) -> GraphState:
        if _has_error(state):
            return {}

        # Check if parameters are needed
        if state.get("parameters_needed"):
            return {}

        workbench = state["workbench"]
        tool_name = workbench.selected_tool
        if not tool_name:
            return {
                "error": _simple_error(
                    message="No tool selected for execution.",
                    node="tool_agent",
                )
            }

        raw_inputs = dict(state.get("raw_inputs", {}))

        try:
            runner = self.config.tool_registry.resolve(tool_name)
        except Exception as exc:
            logger.exception("Failed to load tool '%s'", tool_name)
            return {
                "error": _format_error(
                    self.config,
                    message=f"Failed to load tool '{tool_name}'",
                    exc=exc,
                    node="tool_agent",
                    tool_name=tool_name,
                )
            }

        attempts = 0
        outcome: Dict[str, Any] | None = None
        last_exc: Exception | None = None
        started_at = datetime.now(UTC)
        completed_at = started_at
        elapsed_ms = 0.0

        while attempts < self.config.retry_policy.max_attempts:
            attempts += 1
            started_at = datetime.now(UTC)
            t0 = perf_counter()
            try:
                outcome = runner(raw_inputs)
                completed_at = datetime.now(UTC)
                elapsed_ms = (perf_counter() - t0) * 1000.0
                break
            except Exception as exc:  # pragma: no cover - tool runtime errors depend on inputs
                last_exc = exc
                if self.config.retry_policy.should_retry(attempts, exc):
                    logger.warning(
                        "Tool '%s' attempt %s failed (%s); retrying.",
                        tool_name,
                        attempts,
                        exc,
                    )
                    if self.config.retry_policy.backoff_seconds > 0:
                        sleep(self.config.retry_policy.backoff_seconds)
                    continue
                logger.exception("Tool '%s' execution failed", tool_name)
                friendly_message = str(exc) if isinstance(exc, ValueError) else f"Tool '{tool_name}' execution failed"
                return {
                    "error": _format_error(
                        self.config,
                        message=friendly_message,
                        exc=exc,
                        node="tool_agent",
                        tool_name=tool_name,
                    )
                }

        if outcome is None:
            exc = last_exc or RuntimeError("Unknown tool failure")
            friendly_message = str(exc) if isinstance(exc, ValueError) else f"Tool '{tool_name}' execution failed"
            return {
                "error": _format_error(
                    self.config,
                    message=friendly_message,
                    exc=exc,
                    node="tool_agent",
                    tool_name=tool_name,
                )
            }

        results = dict(outcome.get("results", {}))
        units = dict(outcome.get("units", {}))
        steps = list(outcome.get("steps", []))
        warnings = list(outcome.get("warnings", []))
        metadata = dict(outcome.get("metadata", {}))

        workbench.inputs_normalized = raw_inputs
        workbench.results = results
        workbench.units = units
        workbench.warnings = warnings
        workbench.steps = steps

        invocation = ToolInvocation(
            tool_name=tool_name,
            inputs=raw_inputs,
            outputs=results,
            steps=steps,
            warnings=warnings,
            started_at=started_at,
            completed_at=completed_at,
        )
        workbench.tool_invocation = invocation

        metadata.setdefault("tool_runtime_ms", elapsed_ms)
        metadata.setdefault("retry_attempts", attempts)

        return {
            "workbench": workbench,
            "metadata": metadata,
        }


@dataclass(slots=True)
class ExplainerAgent:
    def __call__(self, state: GraphState) -> GraphState:
        if _has_error(state):
            return {}

        workbench = state["workbench"]

        sections: List[str] = []
        if workbench.steps:
            sections.append(steps_to_markdown(workbench.steps))
        if workbench.results:
            sections.append("## Final Results\n\n" + format_final_results(workbench.results, workbench.units))

        warnings_md = summarize_warnings(workbench.warnings)
        if warnings_md:
            sections.append("## Warnings\n" + warnings_md)

        markdown = "\n\n".join(section.strip() for section in sections if section)

        return {
            "response_markdown": markdown or None,
        }


@dataclass(slots=True)
class SearchAgent:
    config: GraphConfig

    def __call__(self, state: GraphState) -> GraphState:
        if _has_error(state):
            return {}

        workbench = state["workbench"]

        if not (self.config.enable_search and self.config.search_index and workbench.user_query):
            return {"search_results": []}

        try:
            results = self.config.search_index.search(workbench.user_query, limit=5)
        except Exception as exc:  # pragma: no cover - search index may hit IO errors
            logger.warning("Search enrichment failed: %s", exc)
            return {"search_results": []}

        return {"search_results": results}


@dataclass(slots=True)
class HistoryAgent:
    config: GraphConfig

    def __call__(self, state: GraphState) -> GraphState:
        if _has_error(state):
            return {}

        workbench = state["workbench"]

        if not state.get("save_history", True):
            return {"workbench": workbench}

        try:
            history_id = self.config.history.log_run(
                tool_name=workbench.selected_tool or "",
                user_query=workbench.user_query or "",
                auto_route=workbench.auto_route,
                inputs=workbench.inputs_normalized,
                outputs=workbench.results,
                steps=workbench.steps,
                units=workbench.units,
                warnings=workbench.warnings,
            )
            workbench.history_id = history_id
        except Exception as exc:  # pragma: no cover - database issues
            logger.warning("Failed to log history: %s", exc)

        return {"workbench": workbench}


@dataclass(slots=True)
class FinalizerAgent:
    config: GraphConfig

    def __call__(self, state: GraphState) -> GraphState:
        workbench = state["workbench"]
        route_decision = state.get("route_decision")
        search_results = state.get("search_results") or []
        metadata = dict(state.get("metadata") or {})

        # Check if parameters are needed
        parameters_needed = state.get("parameters_needed")
        if parameters_needed:
            response: Dict[str, Any] = {
                "ok": False,
                "tool_name": parameters_needed["tool_name"],
                "parameters_needed": parameters_needed,
                "route": asdict(route_decision) if route_decision else None,
                "error": {
                    "message": f"Missing required parameters for {parameters_needed['tool_name']}",
                    "node": "parameter_validation_agent",
                    "type": "missing_parameters",
                }
            }
            return {"response": response}

        if workbench.tool_invocation:
            metadata.setdefault("tool_invocation", workbench.tool_invocation.to_serializable())

        response: Dict[str, Any] = {
            "ok": not _has_error(state),
            "tool_name": workbench.selected_tool,
            "results": workbench.results,
            "units": workbench.units,
            "steps": [step.to_dict() for step in workbench.steps],
            "warnings": workbench.warnings,
            "markdown": state.get("response_markdown"),
            "route": asdict(route_decision) if route_decision else None,
            "history_id": workbench.history_id,
            "search_results": [asdict(result) for result in search_results],
            "metadata": metadata,
        }

        if state.get("error"):
            response["error"] = state["error"]

        return {"response": response}


@dataclass(slots=True)
class WorkbenchGraph:
    """Compiled LangGraph application with helper invocation utilities."""

    app: Any
    config: GraphConfig

    def invoke(self, payload: GraphInput) -> Dict[str, Any]:
        state = default_state()
        state["payload"] = payload
        final_state = self.app.invoke(state)
        return dict(final_state.get("response") or {})

    async def ainvoke(self, payload: GraphInput) -> Dict[str, Any]:
        state = default_state()
        state["payload"] = payload
        final_state = await self.app.ainvoke(state)
        return dict(final_state.get("response") or {})


def build_graph(config: GraphConfig | None = None) -> WorkbenchGraph:
    """Construct and compile the LangGraph workflow for the workbench."""

    config = config or GraphConfig()

    graph = StateGraph(GraphState)

    graph.add_node("prepare_agent", PrepareAgent())
    graph.add_node("router_agent", RouterAgent(config))
    graph.add_node("parameter_validation_agent", ParameterValidationAgent(config))
    graph.add_node("tool_agent", ToolAgent(config))
    graph.add_node("explainer_agent", ExplainerAgent())
    graph.add_node("search_agent", SearchAgent(config))
    graph.add_node("history_agent", HistoryAgent(config))
    graph.add_node("finalizer_agent", FinalizerAgent(config))

    graph.set_entry_point("prepare_agent")
    graph.add_edge("prepare_agent", "router_agent")
    graph.add_edge("router_agent", "parameter_validation_agent")
    
    # Conditional routing from parameter validation
    def should_continue_to_tool(state: GraphState) -> str:
        if state.get("parameters_needed"):
            return "finalizer_agent"  # Skip to finalizer if parameters are needed
        return "tool_agent"  # Continue to tool execution if parameters are complete
    
    graph.add_conditional_edges(
        "parameter_validation_agent",
        should_continue_to_tool,
        {
            "tool_agent": "tool_agent",
            "finalizer_agent": "finalizer_agent",
        }
    )
    
    graph.add_edge("tool_agent", "explainer_agent")
    graph.add_edge("explainer_agent", "search_agent")
    graph.add_edge("search_agent", "history_agent")
    graph.add_edge("history_agent", "finalizer_agent")
    graph.add_edge("finalizer_agent", END)

    app = graph.compile()
    return WorkbenchGraph(app=app, config=config)
