"""Streamlit UI for the Engineering Calculation Workbench."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Tuple

import streamlit as st
from fpdf import FPDF

from core.graph import GraphConfig, GraphInput, WorkbenchGraph, build_graph
from core.history import HistoryManager
from core.routing import ToolMetadata, load_tool_metadata
from core.search import SearchIndex, SearchResult


PAGE_TITLE = "Engineering Calculation Workbench"
SESSION_DEFAULTS = {
    "auto_route": True,
    "selected_tool": None,
    "unit_system": "SI",
    "save_history": True,
    "significant_figures": 4,
    "prompt_text": "",
    "search_query": "",
    "last_payload": None,
    "last_response": None,
    "pending_rerun": None,
    "chat_history": [],
}


@st.cache_resource(show_spinner=False)
def get_services() -> Tuple[WorkbenchGraph, HistoryManager, SearchIndex, List[ToolMetadata]]:
    history = HistoryManager()
    metadata = load_tool_metadata()
    config = GraphConfig(history=history)
    graph = build_graph(config)
    search_index = config.search_index or SearchIndex(tool_registry=metadata, history_manager=history)
    return graph, history, search_index, metadata


def ensure_session_state() -> None:
    for key, value in SESSION_DEFAULTS.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("tool_forms", {})
    st.session_state.setdefault("search_results", [])


def choose_unit(examples: Iterable[str] | None, unit_system: str) -> str:
    if not examples:
        return ""
    candidates = list(examples)
    if unit_system == "SI":
        for unit in candidates:
            if unit in {"m", "m^3/s", "Pa", "kg/m^3", "N", "J/kg/K", "W/m^2/K"}:
                return unit
            if "m" in unit and "ft" not in unit and "in" not in unit:
                return unit
    else:
        for unit in candidates:
            if unit in {"ft", "in", "gpm", "psi", "lb/ft^3", "lb/ft"}:
                return unit
            if any(s in unit for s in ("ft", "in", "psi", "lb")):
                return unit
    return candidates[0]


def render_sidebar(
    *,
    metadata: List[ToolMetadata],
    history: HistoryManager,
    search_index: SearchIndex,
) -> None:
    st.sidebar.title("Workbench Controls")

    # Search panel
    st.sidebar.subheader("Search tools, formulas, history")
    search_query = st.sidebar.text_input("Search", key="search_query", placeholder="Darcy, beam, run #…")
    if search_query:
        st.session_state["search_results"] = search_index.search(search_query, limit=8)
    else:
        st.session_state["search_results"] = []

    for result in st.session_state["search_results"]:
        with st.sidebar.expander(f"[{result.kind}] {result.title}", expanded=False):
            st.write(result.snippet)
            if result.kind == "tool" and st.button("Select tool", key=f"select_tool_{result.identifier}"):
                st.session_state["selected_tool"] = result.identifier.split(":", 1)[-1]
                st.session_state["auto_route"] = False
                st.experimental_rerun()
            elif result.kind == "history" and st.button("Re-run", key=f"rerun_{result.identifier}"):
                history_id = int(result.identifier.split(":", 1)[-1])
                st.session_state["pending_rerun"] = history_id
                st.experimental_rerun()

    st.sidebar.markdown("---")

    st.sidebar.subheader("Routing")
    auto_route = st.sidebar.checkbox("Auto-route", value=st.session_state["auto_route"], key="auto_route")
    tool_names = [meta.name for meta in metadata]
    selected_tool = st.sidebar.selectbox(
        "Tool picker",
        options=["(none)"] + tool_names,
        index=(tool_names.index(st.session_state["selected_tool"]) + 1) if st.session_state["selected_tool"] in tool_names else 0,
    )
    st.session_state["selected_tool"] = selected_tool if selected_tool != "(none)" else None

    # Unit system radio button - let Streamlit manage the state with the key
    st.sidebar.radio(
        "Unit system",
        options=["SI", "US customary"],
        index=0 if st.session_state.get("unit_system", "SI") == "SI" else 1,
        key="unit_system",
    )

    st.sidebar.subheader("Save options")
    st.sidebar.checkbox("Save to history", value=st.session_state["save_history"], key="save_history")

    st.sidebar.markdown("---")
    
    # Clear chat button in sidebar
    st.sidebar.subheader("Chat Controls")
    if st.sidebar.button("🗑️ Clear Chat History", use_container_width=True, type="secondary"):
        st.session_state.chat_history = []
        st.session_state["prompt_text"] = ""
        st.session_state["last_response"] = None
        st.session_state["last_payload"] = None
        st.sidebar.success("✨ Chat cleared!")
        st.rerun()
    
    if st.session_state.get("chat_history"):
        st.sidebar.caption(f"💬 {len(st.session_state['chat_history'])} messages in current chat")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Tool reference")
    for meta in metadata:
        with st.sidebar.expander(meta.name.replace("_", " ").title(), expanded=False):
            st.write(meta.description)
            inputs = meta.schema.get("input_schema", {}).get("properties", {})
            if inputs:
                st.caption("Inputs:")
                for name, info in inputs.items():
                    desc = info.get("description", "")
                    st.write(f"- **{name}** — {desc}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Recent history")
    for record in history.list_runs(limit=8):
        label = f"#{record['id']} • {record['tool_name']}"
        if st.sidebar.button(label, key=f"history_button_{record['id']}"):
            st.session_state["pending_rerun"] = int(record["id"])
            st.experimental_rerun()


def parse_tool_inputs(
    *,
    tool_name: str | None,
    metadata_map: Dict[str, ToolMetadata],
    unit_system: str,
) -> Dict[str, Any]:
    if not tool_name:
        return {}
    meta = metadata_map.get(tool_name)
    if not meta:
        return {}
    schema = meta.schema.get("input_schema", {})
    properties: Dict[str, Any] = schema.get("properties", {})
    required = set(schema.get("required", []))
    form_state = st.session_state["tool_forms"].setdefault(tool_name, {})
    collected: Dict[str, Any] = {}

    with st.expander("Advanced inputs", expanded=True):
        for prop_name, prop_schema in properties.items():
            prop_type = prop_schema.get("type")
            key_prefix = f"{tool_name}:{prop_name}"
            current_value = form_state.get(prop_name)

            include_field = True
            if prop_name not in required:
                toggle_key = f"{key_prefix}:toggle"
                default_toggle = st.session_state.get(toggle_key, current_value is not None)
                include_field = st.checkbox(
                    f"Provide `{prop_name}`",
                    value=default_toggle,
                    key=toggle_key,
                )
                if not include_field:
                    form_state.pop(prop_name, None)
                    continue

            if prop_type == "object" and set(prop_schema.get("properties", {}).keys()) >= {"value", "units"}:
                units_examples = prop_schema.get("properties", {}).get("units", {}).get("examples", [])
                default_units = (
                    current_value.get("units")
                    if isinstance(current_value, dict) and current_value.get("units")
                    else choose_unit(units_examples, unit_system)
                )
                default_value = (
                    float(current_value.get("value"))
                    if isinstance(current_value, dict) and current_value.get("value") is not None
                    else 0.0
                )
                cols = st.columns([2, 1])
                value = cols[0].number_input(
                    f"{prop_name} value",
                    value=default_value,
                    key=f"{key_prefix}:value",
                )
                units = cols[1].selectbox(
                    "Units",
                    options=units_examples or [default_units or ""],
                    index=(units_examples or [default_units or ""]).index(default_units) if default_units in (units_examples or []) else 0,
                    key=f"{key_prefix}:units",
                )
                collected[prop_name] = {"value": value, "units": units}
                form_state[prop_name] = collected[prop_name]
            elif prop_type == "string":
                if "enum" in prop_schema:
                    options = prop_schema["enum"]
                    default = current_value if current_value in options else options[0]
                    collected[prop_name] = st.selectbox(
                        prop_name,
                        options=options,
                        index=options.index(default),
                        key=f"{key_prefix}:enum",
                    )
                else:
                    default = current_value or ""
                    collected[prop_name] = st.text_input(
                        prop_name,
                        value=default,
                        key=f"{key_prefix}:text",
                    )
                form_state[prop_name] = collected[prop_name]
            elif prop_type == "number":
                default = float(current_value) if current_value is not None else float(prop_schema.get("default", 0.0))
                collected[prop_name] = st.number_input(
                    prop_name,
                    value=default,
                    key=f"{key_prefix}:number",
                )
                form_state[prop_name] = collected[prop_name]
            elif prop_type == "integer":
                default = int(current_value) if current_value is not None else int(prop_schema.get("default", 0))
                collected[prop_name] = st.number_input(
                    prop_name,
                    value=default,
                    step=1,
                    key=f"{key_prefix}:integer",
                )
                form_state[prop_name] = collected[prop_name]

    return collected


def execute_calculation(
    *, graph: WorkbenchGraph, payload: GraphInput, search_index: SearchIndex
) -> Dict[str, Any]:
    response = graph.invoke(payload)
    if payload.get("save_history", True) and response.get("ok"):
        # Trigger search index to include latest history entry
        if hasattr(search_index, "history_manager"):
            search_index.history_manager = graph.config.history
    return response


def render_final_results(response: Dict[str, Any]) -> None:
    st.markdown("### 📊 **Your Results**")
    results = response.get("results") or {}
    units = response.get("units") or {}
    precision_session = st.session_state.get("significant_figures", 4)
    
    if not results:
        st.info("ℹ️ No numeric results to display for this calculation.")
        return

    st.markdown("Here's what I calculated for you:")
    
    # Precision control in a more subtle way
    with st.expander("🎛️ Adjust precision", expanded=False):
        precision = st.slider(
            "Significant figures",
            min_value=2,
            max_value=6,
            value=precision_session,
            key="significant_figures",
            help="Choose how many significant figures to show in the results"
        )

    cols = st.columns(min(3, len(results)))
    for (name, value), col in zip(results.items(), cols):
        with col:
            unit = units.get(name, "")
            formatted = f"{value:.{precision}g}" if isinstance(value, (int, float)) else str(value)
            if unit:
                formatted = f"{formatted} {unit}"
            # Make metric names more readable
            friendly_name = name.replace("_", " ").title()
            st.metric(label=friendly_name, value=formatted)

    st.caption("💡 You can adjust the precision above or copy these results for your records.")

    # Copy-to-clipboard via HTML/JS hack
    summary_lines = []
    for name, value in results.items():
        unit = units.get(name, "")
        if isinstance(value, (int, float)):
            text_value = f"{value:.{precision}g}"
        else:
            text_value = str(value)
        line = f"{name}: {text_value} {unit}".strip()
        summary_lines.append(line)
    summary_text = "\n".join(summary_lines)
    summary_text = summary_text.replace("`", r"\`").replace("\\", r"\\\\")
    st.components.v1.html(
        f"""
        <button onclick="navigator.clipboard.writeText(`{summary_text}`);">Copy results to clipboard</button>
        """,
        height=40,
    )


def render_steps(response: Dict[str, Any]) -> None:
    steps = response.get("steps") or []
    if not steps:
        st.info("ℹ️ No detailed steps available for this calculation.")
        return

    st.markdown("### 🔢 **Step-by-Step Solution**")
    st.markdown("Here's how I solved this problem for you:")
    
    for step in steps:
        label = f"**Step {step.get('index')}:** {step.get('description')}"
        with st.expander(label, expanded=False):
            if step.get("equation_tex"):
                st.markdown("**📐 Equation:**")
                st.latex(step["equation_tex"])
            substitutions = step.get("substitutions") or []
            if substitutions:
                st.markdown("**🔢 Substituting values:**")
                for substitution in substitutions:
                    symbol = substitution.get("symbol")
                    value = substitution.get("value")
                    units = substitution.get("units", "")
                    expr = substitution.get("expression")
                    text = f"- `{symbol}` = {value} {units}".strip()
                    if expr:
                        text += f" (from {expr})"
                    st.markdown(text)
            if step.get("result_value") is not None:
                units = step.get("result_units", "")
                st.markdown(f"**✅ Result:** {step['result_value']} {units}".strip())


def render_warnings(response: Dict[str, Any]) -> None:
    warnings = response.get("warnings") or []
    if warnings:
        st.markdown("### ⚠️ **Important Notes**")
        st.markdown("I noticed a few things you should be aware of:")
        for warning in warnings:
            st.warning(f"💡 {warning}")
        st.caption("These are just suggestions to help ensure your design is safe and appropriate!")


def render_route_metadata(response: Dict[str, Any]) -> None:
    route = response.get("route")
    if route:
        st.info(
            f"Selected tool: **{route.get('tool_name', response.get('tool_name'))}**\n\n"
            f"Source: {route.get('source', 'unknown')} — {route.get('reason', '')}"
        )


def generate_markdown_report(response: Dict[str, Any]) -> str:
    lines = ["# Engineering Calculation Report", ""]
    lines.append(f"**Tool:** {response.get('tool_name', 'unknown')}")
    lines.append("")
    if response.get("results"):
        lines.append("## Final Results")
        for name, value in response["results"].items():
            unit = response.get("units", {}).get(name, "")
            lines.append(f"- **{name}**: {value} {unit}".strip())
        lines.append("")
    if response.get("steps"):
        lines.append("## Steps")
        for step in response["steps"]:
            lines.append(f"### Step {step.get('index')} — {step.get('description')}")
            if step.get("equation_tex"):
                lines.append(f"\\[{step['equation_tex']}\\]")
            substitutions = step.get("substitutions") or []
            for substitution in substitutions:
                value = substitution.get("value")
                units = substitution.get("units", "")
                symbol = substitution.get("symbol")
                expr = substitution.get("expression")
                line = f"* `{symbol}` = {value} {units}".strip()
                if expr:
                    line += f" (from {expr})"
                lines.append(line)
            if step.get("result_value") is not None:
                units = step.get("result_units", "")
                lines.append(f"Result: {step['result_value']} {units}".strip())
            lines.append("")
    if response.get("warnings"):
        lines.append("## Warnings")
        for warning in response["warnings"]:
            lines.append(f"- {warning}")
    return "\n".join(lines)


def generate_pdf_report(response: Dict[str, Any]) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Engineering Calculation Report", ln=True)
    pdf.set_font("Helvetica", size=12)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, f"Tool: {response.get('tool_name', 'unknown')}", ln=True)
    pdf.ln(2)

    results = response.get("results") or {}
    if results:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Final Results", ln=True)
        pdf.set_font("Helvetica", size=11)
        for name, value in results.items():
            unit = response.get("units", {}).get(name, "")
            # Format large numbers and truncate long text
            if isinstance(value, (int, float)):
                if abs(value) >= 1e6 or (abs(value) < 1e-3 and value != 0):
                    value_str = f"{value:.3e}"
                else:
                    value_str = f"{value:.6g}"
            else:
                value_str = str(value)
            
            # Truncate very long strings
            if len(value_str) > 80:
                value_str = value_str[:77] + "..."
            
            result_line = f"- {name}: {value_str} {unit}".strip()
            # Ensure line is not too long
            if len(result_line) > 120:
                result_line = result_line[:117] + "..."
            
            pdf.multi_cell(0, 6, result_line)
        pdf.ln(2)

    steps = response.get("steps") or []
    if steps:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Steps", ln=True)
        pdf.set_font("Helvetica", size=11)
        for step in steps:
            desc = step.get('description', '')
            if len(desc) > 100:
                desc = desc[:97] + "..."
            pdf.multi_cell(0, 6, f"Step {step.get('index')} — {desc}")
            
            if step.get("equation_tex"):
                equation = step['equation_tex']
                if len(equation) > 120:
                    equation = equation[:117] + "..."
                pdf.set_font("Helvetica", "I", 10)
                pdf.multi_cell(0, 6, f"Equation: {equation}")
                pdf.set_font("Helvetica", size=11)
                
            substitutions = step.get("substitutions") or []
            for substitution in substitutions:
                symbol = substitution.get('symbol', '')
                value = substitution.get('value', '')
                units = substitution.get('units', '')
                
                # Format the value
                if isinstance(value, (int, float)):
                    if abs(value) >= 1e6 or (abs(value) < 1e-3 and value != 0):
                        value_str = f"{value:.3e}"
                    else:
                        value_str = f"{value:.6g}"
                else:
                    value_str = str(value)
                
                line = f"  • {symbol} = {value_str} {units}".strip()
                if len(line) > 120:
                    line = line[:117] + "..."
                pdf.multi_cell(0, 5, line)
                
            if step.get("result_value") is not None:
                result_val = step['result_value']
                if isinstance(result_val, (int, float)):
                    if abs(result_val) >= 1e6 or (abs(result_val) < 1e-3 and result_val != 0):
                        result_str = f"{result_val:.3e}"
                    else:
                        result_str = f"{result_val:.6g}"
                else:
                    result_str = str(result_val)
                
                line = f"  Result: {result_str} {step.get('result_units', '')}".strip()
                if len(line) > 120:
                    line = line[:117] + "..."
                pdf.multi_cell(0, 5, line)
            pdf.ln(1)

    warnings = response.get("warnings") or []
    if warnings:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Warnings", ln=True)
        pdf.set_font("Helvetica", size=11)
        for warning in warnings:
            warning_text = str(warning)
            if len(warning_text) > 120:
                warning_text = warning_text[:117] + "..."
            pdf.multi_cell(0, 6, f"- {warning_text}")

    output = pdf.output(dest="S")
    if isinstance(output, str):
        return output.encode("latin1")
    elif isinstance(output, bytearray):
        return bytes(output)
    return output


def render_exports(response: Dict[str, Any]) -> None:
    st.markdown("### 📁 **Save Your Work**")
    st.markdown("Want to keep a copy? Download your calculation report:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        md_report = generate_markdown_report(response)
        st.download_button(
            "📄 Download as Markdown",
            data=md_report,
            file_name="calculation_report.md",
            mime="text/markdown",
            use_container_width=True
        )
    
    with col2:
        pdf_bytes = generate_pdf_report(response)
        st.download_button(
            "📋 Download as PDF",
            data=pdf_bytes,
            file_name="calculation_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )


def render_response(response: Dict[str, Any]) -> None:
    if not response:
        return
    
    # Handle parameter collection requests with a friendly tone
    if response.get("parameters_needed"):
        params_info = response["parameters_needed"]
        tool_name = params_info["tool_name"]
        missing_params = params_info["missing_parameters"]
        provided_inputs = params_info.get("provided_inputs", {})
        
        st.markdown("### 🎯 Almost there!")
        st.info(f"Perfect! I know you need a **{tool_name.replace('_', ' ')}** calculation. I just need a few more details to get you the exact results:")
        
        # Create a form for the missing parameters
        with st.form(f"parameter_form_{tool_name}"):
            st.markdown("#### 📝 Please fill in these details:")
            
            collected_inputs = dict(provided_inputs)  # Start with already provided inputs
            
            for i, param in enumerate(missing_params):
                param_name = param["name"]
                param_desc = param["description"]
                param_type = param["type"]
                examples = param.get("examples", [])
                param_properties = param.get("properties", {})
                
                # Make parameter names more friendly
                friendly_name = param_name.replace("_", " ").title()
                
                st.markdown(f"**{i+1}. {friendly_name}**")
                if param_desc:
                    st.caption(f"ℹ️ {param_desc}")
                
                # Handle object types (with value and units)
                if param_type == "object" and "value" in param_properties and "units" in param_properties:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        value = st.number_input(f"Value", key=f"{tool_name}_{param_name}_value", format="%.6g")
                    with col2:
                        unit_examples = param_properties.get("units", {}).get("examples", [])
                        if unit_examples:
                            units = st.selectbox(f"Units", unit_examples, key=f"{tool_name}_{param_name}_units")
                        else:
                            units = st.text_input(f"Units", key=f"{tool_name}_{param_name}_units", placeholder="e.g., m, N, Pa")
                    
                    if value is not None and units:
                        collected_inputs[param_name] = {"value": value, "units": units}
                
                # Handle string types (like load_case)
                elif param_type == "string":
                    enum_values = param.get("enum")
                    if enum_values:
                        selected = st.selectbox(f"Select {param_name}", enum_values, key=f"{tool_name}_{param_name}_enum")
                        collected_inputs[param_name] = selected
                    else:
                        text_value = st.text_input(f"Enter {param_name}", key=f"{tool_name}_{param_name}_text")
                        if text_value:
                            collected_inputs[param_name] = text_value
                
                # Handle number types
                elif param_type == "number":
                    num_value = st.number_input(f"Enter {param_name}", key=f"{tool_name}_{param_name}_number", format="%.6g")
                    if num_value is not None:
                        collected_inputs[param_name] = num_value
                
                # Handle integer types
                elif param_type == "integer":
                    int_value = st.number_input(f"Enter {param_name}", key=f"{tool_name}_{param_name}_integer", step=1)
                    if int_value is not None:
                        collected_inputs[param_name] = int_value
                
                if examples:
                    st.caption(f"💡 Examples: {', '.join(examples)}")
                st.divider()
            
            # Submit button with friendly messaging
            col1, col2 = st.columns([3, 1])
            with col1:
                submitted = st.form_submit_button("🚀 Calculate Now!", type="primary", use_container_width=True)
            with col2:
                st.caption("All set? Let's go!")
            
            if submitted:
                # Add completion message to chat
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"Perfect! I have all the information I need. Let me run the {tool_name.replace('_', ' ')} calculation for you now! ⚙️"
                })
                
                # Store the collected inputs and trigger a rerun
                st.session_state["auto_submit"] = True
                st.session_state["auto_submit_tool"] = tool_name
                st.session_state["auto_submit_inputs"] = collected_inputs
                st.rerun()
        
        return  # Don't show other response elements when collecting parameters
    
    # Normal response handling with friendly messaging
    if response.get("ok"):
        st.success("🎉 **Calculation Complete!** Here are your results:")
        st.markdown("---")
    else:
        error_msg = response.get("error", {}).get("message", "Calculation failed.")
        
        # Check if this is a validation error for missing required fields
        if "Field required" in error_msg and response.get("tool_name"):
            st.error("🤔 **Oops! I need a bit more info**")
            st.info(f"📝 No worries! Just fill in the required parameters for **{response.get('tool_name')}** above, and we'll get this sorted out!")
            with st.expander("🔍 Technical details (for the curious)"):
                st.text(error_msg)
        else:
            st.error(f"😅 **Something went wrong:** {error_msg}")
            st.info("💡 Try rephrasing your question or check if all parameters are correct. I'm here to help!")

    render_route_metadata(response)
    render_final_results(response)
    render_steps(response)
    render_warnings(response)
    if response.get("search_results"):
        st.subheader("Related references")
        for result in response["search_results"]:
            st.markdown(f"- **{result.get('title')}** ({result.get('kind')}) — {result.get('snippet')}")
    render_exports(response)


def apply_rerun_if_requested(
    *, history: HistoryManager, graph: WorkbenchGraph, search_index: SearchIndex
) -> None:
    pending = st.session_state.get("pending_rerun")
    if not pending:
        return
    record = history.get_run(int(pending))
    st.session_state["pending_rerun"] = None
    if not record:
        st.warning(f"Run #{pending} not found.")
        return

    tool_name = record.get("tool_name")
    st.session_state["prompt_text"] = record.get("user_query", "")
    st.session_state["auto_route"] = False
    st.session_state["selected_tool"] = tool_name
    st.session_state["tool_forms"].setdefault(tool_name, {}).update(record.get("inputs", {}))

    payload: GraphInput = {
        "user_query": record.get("user_query", f"Re-run of {tool_name}"),
        "selected_tool": tool_name,
        "auto_route": False,
        "tool_inputs": record.get("inputs", {}),
        "history_id": record.get("id"),
        "save_history": st.session_state.get("save_history", True),
    }
    response = execute_calculation(graph=graph, payload=payload, search_index=search_index)
    st.session_state["last_payload"] = payload
    st.session_state["last_response"] = response


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide", page_icon="🛠️")
    ensure_session_state()
    graph, history, search_index, metadata = get_services()
    metadata_map = {meta.name: meta for meta in metadata}

    # Handle auto-submit from parameter collection
    if st.session_state.get("auto_submit"):
        tool_name = st.session_state.get("auto_submit_tool")
        tool_inputs = st.session_state.get("auto_submit_inputs", {})
        
        # Clear the auto-submit flags
        st.session_state["auto_submit"] = False
        st.session_state["auto_submit_tool"] = None
        st.session_state["auto_submit_inputs"] = None
        
        # Execute the calculation with the collected parameters
        payload: GraphInput = {
            "user_query": f"Calculate {tool_name} with provided parameters",
            "selected_tool": tool_name,
            "auto_route": False,  # Use the specific tool
            "tool_inputs": tool_inputs,
            "save_history": st.session_state.get("save_history", True),
        }
        response = execute_calculation(graph=graph, payload=payload, search_index=search_index)
        st.session_state["last_payload"] = payload
        st.session_state["last_response"] = response

    apply_rerun_if_requested(history=history, graph=graph, search_index=search_index)

    render_sidebar(metadata=metadata, history=history, search_index=search_index)

    st.title("🛠️ Engineering Assistant")
    st.markdown(
        "👋 **Hey there!** I'm your friendly engineering assistant. Just tell me what calculation you need help with, "
        "and I'll guide you through it step by step. Try asking something like:"
    )
    
    # Add some example prompts in a friendly way
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("💡 *\"Calculate beam deflection for a 4m steel beam\"*")
    with col2:
        st.info("💡 *\"What's the pressure drop in my pipe system?\"*")
    with col3:
        st.info("💡 *\"Help me size a heat exchanger\"*")

    # Chat-like interface
    st.markdown("---")
    
    # Initialize chat history if not present
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Clear button at the top of chat area
    if st.session_state.chat_history:
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("🗑️ Clear Chat", type="secondary", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state["prompt_text"] = ""
                st.session_state["last_response"] = None
                st.session_state["last_payload"] = None
                st.success("✨ Chat cleared! Ready for a fresh start!")
                st.rerun()
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### 💬 Our Conversation")
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                with st.chat_message("user", avatar="👤"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant", avatar="🛠️"):
                    st.write(message["content"])

    # Chat input
    with st.form("calculation_form", clear_on_submit=True):
        st.markdown("### 💭 What can I help you calculate today?")
        
        prompt = st.text_area(
            "Type your engineering question here...",
            height=80,
            value=st.session_state.get("prompt_text", ""),
            key="prompt_text",
            placeholder="Example: I need to calculate the deflection of a simply supported beam..."
        )

        # Move tool inputs to an expander to keep the chat feel clean
        if st.session_state.get("selected_tool"):
            with st.expander("🔧 Advanced Parameters (Optional)", expanded=False):
                tool_inputs = parse_tool_inputs(
                    tool_name=st.session_state.get("selected_tool"),
                    metadata_map=metadata_map,
                    unit_system=st.session_state.get("unit_system", "SI"),
                )
        else:
            tool_inputs = {}

        col1, col2 = st.columns([3, 1])
        with col1:
            submitted = st.form_submit_button("🚀 Let's Calculate!", type="primary", use_container_width=True)
        with col2:
            clear_clicked = st.form_submit_button("�️ Clear", use_container_width=True, help="Clear the chat and start fresh")
            
        if clear_clicked:
            st.session_state.chat_history = []
            st.session_state["prompt_text"] = ""
            st.session_state["last_response"] = None
            st.session_state["last_payload"] = None
            st.success("✨ Chat cleared! Ready for a fresh start!")
            st.rerun()

    response: Dict[str, Any] | None = st.session_state.get("last_response")

    if submitted and prompt.strip():
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user", 
            "content": prompt.strip()
        })
        
        payload: GraphInput = {
            "user_query": prompt or "",
            "selected_tool": st.session_state.get("selected_tool") if not st.session_state.get("auto_route", True) else None,
            "auto_route": st.session_state.get("auto_route", True),
            "tool_inputs": tool_inputs,
            "save_history": st.session_state.get("save_history", True),
        }
        response = execute_calculation(graph=graph, payload=payload, search_index=search_index)
        st.session_state["last_payload"] = payload
        st.session_state["last_response"] = response
        
        # Add assistant response to chat history
        if response:
            if response.get("parameters_needed"):
                assistant_msg = f"I can help you with that! I identified this as a **{response['parameters_needed']['tool_name']}** calculation. Let me gather the required parameters from you."
            elif response.get("ok"):
                assistant_msg = f"Great! I've completed the **{response.get('tool_name', 'calculation')}** for you. Here are the results:"
            else:
                error_msg = response.get("error", {}).get("message", "Something went wrong")
                assistant_msg = f"I encountered an issue: {error_msg}. Let me help you fix this!"
            
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": assistant_msg
            })
        
        st.rerun()

    if response:
        render_response(response)


if __name__ == "__main__":
    main()
