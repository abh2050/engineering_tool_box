# LangGraph Orchestration Plan

## Goals
- Route user prompts to the correct engineering tool using either the OpenAI-backed or heuristic router.
- Execute the selected tool with validated inputs and capture calculation steps, results, units, and warnings inside `WorkbenchState`.
- Generate Markdown explanations, persist the run to history, and optionally surface related search hits for UI enrichment.
- Produce a response artifact that can be consumed by the Streamlit UI, LangGraph agents, and MCP tool adapters.

## State Contracts

### Graph Input Payload
```jsonc
{
  "user_query": "Calculate pressure drop for water in a 4" steel pipe",
  "selected_tool": null,          // optional manual override
  "auto_route": true,             // indicates whether router may choose tool
  "tool_inputs": {...},           // raw input dict destined for the tool
  "history_id": null              // optional prior run reference
}
```

### LangGraph State (`GraphState`)
| Key | Type | Description |
| --- | ---- | ----------- |
| `workbench` | `WorkbenchState` | Shared mutable state across nodes (steps, results, warnings, etc.). |
| `raw_inputs` | `dict[str, Any]` | Untouched user-provided payload for the tool.
| `route_decision` | `RouteDecision | None` | Output of router; includes tool name, confidence, reason.
| `response_markdown` | `str | None` | Human-friendly explanation assembled from steps/results.
| `search_results` | `list[SearchResult]` | Optional supplemental content from `SearchIndex`.
| `error` | `dict[str, Any] | None` | Captures error metadata if a node raises or validation fails.

The state obeys LangGraph's cumulative semantics—each node returns a shallow update merged into the running state.

## Node Responsibilities & Edges

1. **`prepare_state`**
   - Inputs: `GraphInput`
   - Tasks: instantiate `WorkbenchState` (via `init_state()`), assign `user_query`, `selected_tool`, `auto_route`, and stash `tool_inputs` in `raw_inputs`.
   - Next: `route_or_bypass`

2. **`route_or_bypass`**
   - Uses `Router.route` when `auto_route` is `True` and no manual selection.
   - Updates `workbench.selected_tool`, records `route_decision`.
   - Conditional edge: if routing fails (`error`), skip to `finalize`.
   - Next: `execute_tool`

3. **`execute_tool`**
   - Loads the module from `tools` package by name (`importlib`).
   - Invokes its `run(raw_inputs)` function.
   - Updates `workbench` with `inputs_normalized`, `results`, `units`, `steps`, `warnings`.
   - Creates `ToolInvocation` record including wall-clock timestamps.
   - On exception, populate `error` and short-circuit to `finalize`.
   - Next: `enrich_outputs`

4. **`enrich_outputs`**
   - Calls `steps_to_markdown` and `format_final_results` to build `response_markdown`.
   - Optionally executes `SearchIndex.search` using `user_query`, storing top hits.
   - Next: `log_history`

5. **`log_history`**
   - Persists run via `HistoryManager.log_run`.
   - Stores returned `history_id` inside `workbench` and updates `search_index.refresh_glossary()` when needed.
   - Next: `finalize`

6. **`finalize`**
   - Returns lightweight response dict:
     ```python
     {
       "ok": error is None,
       "tool_name": workbench.selected_tool,
       "results": workbench.results,
       "units": workbench.units,
       "steps": [step.to_dict() ...],
       "warnings": workbench.warnings,
       "markdown": response_markdown,
       "route": route_decision.model_dump() if present,
       "history_id": workbench.history_id,
       "search_results": [...]
     }
     ```
   - This payload is consumed by both the Streamlit UI and MCP adapter.

### Edge Diagram
```
START → prepare_state → route_or_bypass → execute_tool → enrich_outputs → log_history → finalize → END
                           ↘ (error) -----------------------------------↗
```

## Dependency Wiring
- `Router` injects tool metadata and optional OpenAI client.
- `HistoryManager` and `SearchIndex` share the SQLite-backed history DB.
- `explainer` utilities produce Markdown.
- `tools` package modules provide computation entry points.

These dependencies are bundled into a `GraphConfig` dataclass so `build_graph(config)` can be invoked from Streamlit, CLI scripts, or MCP servers with custom overrides (e.g., in-memory history for tests).

## Error Handling
- All nodes wrap their logic in try/except, converting exceptions to the shared `error` slot with `message`, `details`, and optional `traceback` (gated by debug flag).
- When `error` is set, downstream nodes should skip heavy work; `finalize` still returns a response with `ok=False` and any diagnostic text.

## Ties to MCP Adapter
- `CallToolResult` structured content will reference the `finalize` payload.
- The adapter exposes `list_tools` via `load_tool_metadata` and `call_tool` by running the graph with `auto_route=False` and `selected_tool` fixed, ensuring deterministic tool execution.

## Next Steps
1. Implement the actual node functions inside `core.graph`, honoring the interfaces above.
2. Create an MCP adapter in `core.adapters` that wraps `build_graph` and surfaces MCP-compatible responses.
3. Add smoke tests invoking the compiled graph with sample payloads per tool to verify end-to-end behavior.
