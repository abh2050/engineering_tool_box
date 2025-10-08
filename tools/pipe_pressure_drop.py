"""Darcy–Weisbach pressure drop calculator with Colebrook–White friction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError, model_validator

from core.state import Step, Substitution
from core.units import UnitField, get_unit_registry


ureg = get_unit_registry()


FLOW_FIELD = UnitField("volumetric_flow_rate", ["meter ** 3 / second"])
VELOCITY_FIELD = UnitField("velocity", ["meter / second"])
DIAMETER_FIELD = UnitField("diameter", ["meter"])
LENGTH_FIELD = UnitField("length", ["meter"])
ROUGHNESS_FIELD = UnitField("roughness", ["meter"])
DENSITY_FIELD = UnitField("density", ["kilogram / meter ** 3"])
VISCOSITY_FIELD = UnitField("dynamic_viscosity", ["pascal * second"])


class PipePressureDropInputs(BaseModel):
    """Validated inputs for the pipe pressure drop calculation."""

    volumetric_flow_rate: Optional[Any] = Field(default=None, description="Volumetric flow rate")
    velocity: Optional[Any] = Field(default=None, description="Flow velocity")
    diameter: Any = Field(description="Pipe internal diameter")
    length: Any = Field(description="Pipe length")
    roughness: Any = Field(description="Absolute roughness")
    density: Any = Field(description="Fluid density")
    dynamic_viscosity: Any = Field(description="Fluid dynamic viscosity")

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="before")
    def _coerce_units(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        parsed = dict(values)
        field_map = {
            "volumetric_flow_rate": FLOW_FIELD,
            "velocity": VELOCITY_FIELD,
            "diameter": DIAMETER_FIELD,
            "length": LENGTH_FIELD,
            "roughness": ROUGHNESS_FIELD,
            "density": DENSITY_FIELD,
            "dynamic_viscosity": VISCOSITY_FIELD,
        }
        for key, field in field_map.items():
            raw = parsed.get(key)
            if raw is None:
                continue
            parsed[key] = field.validate(raw)
        return parsed

    @model_validator(mode="after")
    def _validate_flow_inputs(self) -> "PipePressureDropInputs":
        if self.volumetric_flow_rate is None and self.velocity is None:
            raise ValueError("Provide either volumetric_flow_rate or velocity.")
        
        # Validate positive values
        if self.volumetric_flow_rate is not None and self.volumetric_flow_rate.magnitude <= 0:
            raise ValueError("Flow rate must be positive.")
        if self.velocity is not None and self.velocity.magnitude <= 0:
            raise ValueError("Velocity must be positive.")
        if self.diameter.magnitude <= 0:
            raise ValueError("Diameter must be positive.")
        if self.length.magnitude <= 0:
            raise ValueError("Length must be positive.")
        if self.roughness.magnitude < 0:
            raise ValueError("Roughness must be non-negative.")
        if self.density.magnitude <= 0:
            raise ValueError("Density must be positive.")
        if self.dynamic_viscosity.magnitude <= 0:
            raise ValueError("Dynamic viscosity must be positive.")
        
        return self


@dataclass(slots=True)
class PipePressureDropResult:
    delta_p: float
    friction_factor: float
    reynolds_number: float
    velocity: float
    volumetric_flow_rate: float
    head_loss: float
    method: str
    iterations: int


def _colebrook_white(reynolds: float, rel_roughness: float) -> tuple[float, str, int]:
    """Solve for the Darcy friction factor using the Colebrook–White equation."""

    if reynolds < 2300:
        return 64.0 / reynolds, "Laminar", 0

    if reynolds == 0:
        raise ValueError("Reynolds number must be positive for Colebrook iteration.")

    # Initial guess using Swamee-Jain explicit approximation
    guess = 0.25 / (math.log10(rel_roughness / 3.7 + 5.74 / reynolds ** 0.9)) ** 2
    friction = max(guess, 1e-6)

    for iteration in range(1, 51):
        inv_sqrt_f = -2.0 * math.log10(rel_roughness / 3.7 + 2.51 / (reynolds * math.sqrt(friction)))
        new_friction = 1.0 / (inv_sqrt_f ** 2)
        if abs(new_friction - friction) < 1e-8:
            return new_friction, "Turbulent", iteration
        friction = new_friction

    return friction, "Turbulent (max iterations)", 50


def run(raw_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Compute pressure drop, Reynolds number, and friction factor."""

    try:
        inputs = PipePressureDropInputs(**raw_inputs)
    except ValidationError as exc:  # pragma: no cover - complex pydantic output
        raise ValueError(exc.errors())

    diameter_m = inputs.diameter.to("meter").magnitude
    length_m = inputs.length.to("meter").magnitude
    roughness_m = inputs.roughness.to("meter").magnitude
    density = inputs.density.to("kilogram / meter ** 3").magnitude
    viscosity = inputs.dynamic_viscosity.to("pascal * second").magnitude

    area = math.pi * (diameter_m ** 2) / 4.0

    if inputs.volumetric_flow_rate is not None:
        volumetric_flow = inputs.volumetric_flow_rate.to("meter ** 3 / second").magnitude
        velocity = volumetric_flow / area
    else:
        velocity = inputs.velocity.to("meter / second").magnitude
        volumetric_flow = velocity * area

    reynolds = density * velocity * diameter_m / viscosity
    rel_roughness = roughness_m / diameter_m if diameter_m else 0.0

    friction, regime, iterations = _colebrook_white(reynolds, rel_roughness)
    head_loss = friction * (length_m / diameter_m) * (velocity ** 2) / (2 * 9.80665)
    delta_p = friction * (length_m / diameter_m) * density * velocity ** 2 / 2.0

    result = PipePressureDropResult(
        delta_p=delta_p,
        friction_factor=friction,
        reynolds_number=reynolds,
        velocity=velocity,
        volumetric_flow_rate=volumetric_flow,
        head_loss=head_loss,
        method=regime,
        iterations=iterations,
    )

    steps = _build_steps(
        diameter_m,
        area,
        volumetric_flow,
        velocity,
        density,
        viscosity,
        reynolds,
        rel_roughness,
        friction,
        delta_p,
        head_loss,
        length_m,
    )

    warnings: List[str] = []
    if reynolds < 2300:
        warnings.append("Flow is laminar; Colebrook–White not required. Using 64/Re.")
    elif iterations >= 50:
        warnings.append("Colebrook–White solver reached maximum iterations; friction factor may be approximate.")

    return {
        "results": {
            "delta_p": result.delta_p,
            "friction_factor": result.friction_factor,
            "reynolds_number": result.reynolds_number,
            "velocity": result.velocity,
            "volumetric_flow_rate": result.volumetric_flow_rate,
            "head_loss": result.head_loss,
        },
        "units": {
            "delta_p": "Pa",
            "friction_factor": "",
            "reynolds_number": "",
            "velocity": "m/s",
            "volumetric_flow_rate": "m^3/s",
            "head_loss": "m",
        },
        "steps": steps,
        "warnings": warnings,
        "metadata": {
            "flow_regime": regime,
            "colebrook_iterations": iterations,
        },
    }


def _build_steps(
    diameter_m: float,
    area: float,
    volumetric_flow: float,
    velocity: float,
    density: float,
    viscosity: float,
    reynolds: float,
    rel_roughness: float,
    friction: float,
    delta_p: float,
    head_loss: float,
    length_m: float,
) -> List[Step]:
    steps: List[Step] = []

    steps.append(
        Step(
            index=1,
            description="Compute pipe cross-sectional area",
            equation_tex=r"A = \frac{\pi D^2}{4}",
            substitutions=[
                Substitution("D", diameter_m, "m"),
            ],
            result_value=area,
            result_units="m^2",
        )
    )

    steps.append(
        Step(
            index=2,
            description="Calculate flow velocity",
            equation_tex=r"v = \frac{Q}{A}",
            substitutions=[
                Substitution("Q", volumetric_flow, "m^3/s"),
                Substitution("A", area, "m^2"),
            ],
            result_value=velocity,
            result_units="m/s",
        )
    )

    steps.append(
        Step(
            index=3,
            description="Determine Reynolds number",
            equation_tex=r"Re = \frac{\rho v D}{\mu}",
            substitutions=[
                Substitution(r"\rho", density, "kg/m^3"),
                Substitution("v", velocity, "m/s"),
                Substitution("D", diameter_m, "m"),
                Substitution(r"\mu", viscosity, "Pa·s"),
            ],
            result_value=reynolds,
        )
    )

    steps.append(
        Step(
            index=4,
            description="Solve for Darcy friction factor (Colebrook–White)",
            equation_tex=r"\frac{1}{\sqrt{f}} = -2\log_{10}\left(\frac{\varepsilon/D}{3.7} + \frac{2.51}{Re\sqrt{f}}\right)",
            substitutions=[
                Substitution(r"\varepsilon/D", rel_roughness, ""),
                Substitution("Re", reynolds, ""),
            ],
            result_value=friction,
        )
    )

    steps.append(
        Step(
            index=5,
            description="Compute pressure drop",
            equation_tex=r"\Delta P = f \frac{L}{D} \frac{\rho v^2}{2}",
            substitutions=[
                Substitution("f", friction, ""),
                Substitution("L", length_m, "m"),
                Substitution("D", diameter_m, "m"),
                Substitution(r"\rho", density, "kg/m^3"),
                Substitution("v", velocity, "m/s"),
            ],
            result_value=delta_p,
            result_units="Pa",
        )
    )

    steps.append(
        Step(
            index=6,
            description="Compute head loss",
            equation_tex=r"h_f = \frac{\Delta P}{\rho g}",
            substitutions=[
                Substitution(r"\Delta P", delta_p, "Pa"),
                Substitution(r"\rho", density, "kg/m^3"),
                Substitution("g", 9.80665, "m/s^2"),
            ],
            result_value=head_loss,
            result_units="m",
        )
    )

    return steps

