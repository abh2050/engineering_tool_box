"""Pump hydraulic power and NPSH availability calculator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError, model_validator

from core.state import Step, Substitution
from core.units import UnitField, get_unit_registry


ureg = get_unit_registry()
GRAVITY = 9.80665  # m/s^2


FLOW_FIELD = UnitField("volumetric_flow_rate", ["meter ** 3 / second"])
HEAD_FIELD = UnitField("head", ["meter"])
DENSITY_FIELD = UnitField("density", ["kilogram / meter ** 3"])
PRESSURE_FIELD = UnitField("pressure", ["pascal"])
ELEVATION_FIELD = UnitField("elevation", ["meter"])
LOSS_FIELD = UnitField("loss", ["meter"])


class PumpPowerInputs(BaseModel):
    volumetric_flow_rate: Any
    head: Any
    density: Any
    efficiency: Any
    suction_pressure: Any
    vapor_pressure: Any
    suction_elevation: Any
    suction_losses: Any
    npsh_required: Optional[Any] = None

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="before")
    def _coerce_units(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        parsed = dict(values)
        coercion_map = {
            "volumetric_flow_rate": FLOW_FIELD,
            "head": HEAD_FIELD,
            "density": DENSITY_FIELD,
            "suction_pressure": PRESSURE_FIELD,
            "vapor_pressure": PRESSURE_FIELD,
            "suction_elevation": ELEVATION_FIELD,
            "suction_losses": LOSS_FIELD,
            "npsh_required": HEAD_FIELD,
        }
        for key, unit_field in coercion_map.items():
            raw = parsed.get(key)
            if raw is None:
                continue
            parsed[key] = unit_field.validate(raw)

        eff_raw = parsed.get("efficiency")
        if isinstance(eff_raw, dict):
            value = eff_raw.get("value")
            units = eff_raw.get("units", "")
            if value is None:
                raise ValueError("efficiency requires value")
            if units.lower() in {"%", "percent"}:
                parsed["efficiency"] = float(value) / 100.0
            else:
                parsed["efficiency"] = float(value)
        elif isinstance(eff_raw, (int, float)):
            parsed["efficiency"] = float(eff_raw)
        else:
            raise ValueError("efficiency must be a numeric value or {value, units}")

        return parsed

    @model_validator(mode="after")
    def _validate_efficiency(self) -> "PumpPowerInputs":
        if not (0 < self.efficiency <= 1.0):
            raise ValueError("efficiency must be between 0 and 1 (exclusive of 0)")
        
        # Validate positive values
        if self.volumetric_flow_rate.magnitude <= 0:
            raise ValueError("Flow rate must be positive")
        if self.head.magnitude <= 0:
            raise ValueError("Head must be positive")
        if self.density.magnitude <= 0:
            raise ValueError("Density must be positive")
        if self.suction_pressure.magnitude <= 0:
            raise ValueError("Suction pressure must be positive")
        if self.vapor_pressure.magnitude < 0:
            raise ValueError("Vapor pressure must be non-negative")
        if self.suction_losses.magnitude < 0:
            raise ValueError("Suction losses must be non-negative")
        
        return self


@dataclass(slots=True)
class PumpPowerResult:
    hydraulic_power: float
    input_power: float
    npsha: float
    npshr: Optional[float]
    npsh_margin: Optional[float]
    adequate_npsh: Optional[bool]
    suction_pressure_head: float
    vapor_pressure_head: float


def run(raw_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Compute hydraulic/input power and net positive suction head availability."""

    try:
        inputs = PumpPowerInputs(**raw_inputs)
    except ValidationError as exc:  # pragma: no cover
        raise ValueError(exc.errors())

    flow = inputs.volumetric_flow_rate.to("meter ** 3 / second").magnitude
    head = inputs.head.to("meter").magnitude
    density = inputs.density.to("kilogram / meter ** 3").magnitude
    suction_pressure = inputs.suction_pressure.to("pascal").magnitude
    vapor_pressure = inputs.vapor_pressure.to("pascal").magnitude
    elevation = inputs.suction_elevation.to("meter").magnitude
    losses = inputs.suction_losses.to("meter").magnitude
    npshr = inputs.npsh_required.to("meter").magnitude if inputs.npsh_required is not None else None

    hydraulic_power = density * GRAVITY * flow * head
    input_power = hydraulic_power / inputs.efficiency
    suction_head = suction_pressure / (density * GRAVITY)
    vapor_head = vapor_pressure / (density * GRAVITY)
    npsha = suction_head - vapor_head + elevation - losses

    margin = npsha - npshr if npshr is not None else None
    adequate = margin is None or margin >= 0

    result = PumpPowerResult(
        hydraulic_power=hydraulic_power,
        input_power=input_power,
        npsha=npsha,
        npshr=npshr,
        npsh_margin=margin,
        adequate_npsh=adequate,
        suction_pressure_head=suction_head,
        vapor_pressure_head=vapor_head,
    )

    steps = _build_steps(
        density=density,
        flow=flow,
        head=head,
        efficiency=inputs.efficiency,
        hydraulic_power=hydraulic_power,
        input_power=input_power,
        suction_head=suction_head,
        vapor_head=vapor_head,
        elevation=elevation,
        losses=losses,
        npsha=npsha,
        npshr=npshr,
        margin=margin,
    )

    warnings: List[str] = []
    if npshr is not None and margin is not None and margin < 0:
        warnings.append("NPSHa is below NPSHr. Pump may cavitate.")
    
    # Additional warnings
    if inputs.efficiency < 0.5:
        warnings.append("Very low pump efficiency (<50%) - consider pump selection optimization")
    
    if margin is not None and margin < 0.5:  # Less than 0.5 meters margin
        warnings.append("Low NPSH margin - risk of cavitation")
    
    # High head warning
    if head > 30:  # More than 30m
        warnings.append("Very high head application - verify pump selection")
    
    # High suction elevation warning  
    elevation_m = inputs.suction_elevation.to("meter").magnitude
    if elevation_m > 6:  # More than 6m suction lift
        warnings.append("High suction lift - risk of cavitation or pump performance issues")

    return {
        "results": {
            "hydraulic_power": result.hydraulic_power,
            "input_power": result.input_power,
            "npsha": result.npsha,
            "npshr": result.npshr,
            "npsh_margin": result.npsh_margin,
            "adequate_npsh": result.adequate_npsh,
            "suction_pressure_head": result.suction_pressure_head,
            "vapor_pressure_head": result.vapor_pressure_head,
        },
        "units": {
            "hydraulic_power": "W",
            "input_power": "W",
            "npsha": "m",
            "npshr": "m",
            "npsh_margin": "m",
            "adequate_npsh": "",
            "suction_pressure_head": "m",
            "vapor_pressure_head": "m",
        },
        "steps": steps,
        "warnings": warnings,
        "metadata": {"efficiency": inputs.efficiency},
    }


def _build_steps(
    *,
    density: float,
    flow: float,
    head: float,
    efficiency: float,
    hydraulic_power: float,
    input_power: float,
    suction_head: float,
    vapor_head: float,
    elevation: float,
    losses: float,
    npsha: float,
    npshr: Optional[float],
    margin: Optional[float],
) -> List[Step]:
    steps: List[Step] = []
    steps.append(
        Step(
            index=1,
            description="Hydraulic power from flow and head",
            equation_tex=r"P_h = \rho g Q H",
            substitutions=[
                Substitution(r"\rho", density, "kg/m^3"),
                Substitution("g", GRAVITY, "m/s^2"),
                Substitution("Q", flow, "m^3/s"),
                Substitution("H", head, "m"),
            ],
            result_value=hydraulic_power,
            result_units="W",
        )
    )

    steps.append(
        Step(
            index=2,
            description="Input shaft power from efficiency",
            equation_tex=r"P_{input} = \frac{P_h}{\eta}",
            substitutions=[
                Substitution("P_h", hydraulic_power, "W"),
                Substitution(r"\eta", efficiency, ""),
            ],
            result_value=input_power,
            result_units="W",
        )
    )

    steps.append(
        Step(
            index=3,
            description="Convert suction and vapor pressure to head",
            equation_tex=r"h = \frac{P}{\rho g}",
            substitutions=[
                Substitution(r"P_s", suction_head * density * GRAVITY, "Pa"),
                Substitution(r"P_v", vapor_head * density * GRAVITY, "Pa"),
            ],
            result_value=suction_head,
            result_units="m",
        )
    )

    steps.append(
        Step(
            index=4,
            description="Compute NPSHa",
            equation_tex=r"NPSH_a = \left(\frac{P_s - P_v}{\rho g}\right) + z - h_f",
            substitutions=[
                Substitution(r"P_s", suction_head * density * GRAVITY, "Pa"),
                Substitution(r"P_v", vapor_head * density * GRAVITY, "Pa"),
                Substitution("z", elevation, "m"),
                Substitution("h_f", losses, "m"),
            ],
            result_value=npsha,
            result_units="m",
        )
    )

    if npshr is not None:
        steps.append(
            Step(
                index=5,
                description="Compare NPSHa to required NPSHr",
                equation_tex=r"\text{margin} = NPSH_a - NPSH_r",
                substitutions=[
                    Substitution("NPSH_a", npsha, "m"),
                    Substitution("NPSH_r", npshr, "m"),
                ],
                result_value=margin,
                result_units="m",
            )
        )

    return steps

