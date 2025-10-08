"""Formatting helpers that turn computation steps into rich Markdown."""

from __future__ import annotations

from typing import Iterable, Sequence

from .state import Step, Substitution


def steps_to_markdown(steps: Sequence[Step], precision: int = 4) -> str:
    """Render a list of :class:`Step` objects into Markdown bullet points."""

    if not steps:
        return "*(No calculation steps recorded.)*"

    lines: list[str] = []
    for step in steps:
        header = f"**Step {step.index}:** {step.description}"
        lines.append(header)
        if step.equation_tex:
            lines.append(f"  \\[{step.equation_tex}\\]")
        if step.substitutions:
            lines.append("  Substitutions:")
            for substitution in step.substitutions:
                lines.append(_format_substitution(substitution, precision))
        if step.result_value is not None:
            units = f" {step.result_units}" if step.result_units else ""
            if isinstance(step.result_value, (int, float)):
                rendered_value = f"{step.result_value:.{precision}g}"
            else:
                rendered_value = str(step.result_value)
            lines.append(f"  Result: `{rendered_value}{units}`")
        lines.append("")
    return "\n".join(lines).strip()


def format_final_results(results: dict[str, float], units: dict[str, str], precision: int = 4) -> str:
    """Generate Markdown for the final numeric outputs."""

    if not results:
        return "*(No results computed.)*"

    lines = ["| Quantity | Value |", "| --- | --- |"]
    for key, value in results.items():
        unit = units.get(key, "")
        formatted = f"{value:.{precision}g}" if isinstance(value, (int, float)) else value
        if unit:
            formatted = f"{formatted} {unit}"
        lines.append(f"| `{key}` | {formatted} |")
    return "\n".join(lines)


def summarize_warnings(warnings: Iterable[str]) -> str:
    """Combine warnings into a Markdown-friendly block."""

    warnings = list(warnings)
    if not warnings:
        return ""
    return "\n".join(f"- ⚠️ {message}" for message in warnings)


def _format_substitution(substitution: Substitution, precision: int) -> str:
    symbol = substitution.symbol
    value = _format_optional_number(substitution.value, precision)
    units = f" {substitution.units}" if substitution.units else ""
    expression = f" ← {substitution.expression}" if substitution.expression else ""
    return f"  • `{symbol}` = {value}{units}{expression}"


def _format_optional_number(value: float | None, precision: int) -> str:
    if value is None:
        return "(unspecified)"
    return f"{value:.{precision}g}"

