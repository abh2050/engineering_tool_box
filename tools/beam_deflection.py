"""Euler–Bernoulli beam deflection calculator for simply supported beams."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError, model_validator

from core.state import Step, Substitution
from core.units import UnitField, get_unit_registry


ureg = get_unit_registry()


LOAD_CASES = {"point_load_center", "uniform_load"}

LENGTH_FIELD = UnitField("length", ["meter"])
ELASTIC_MODULUS_FIELD = UnitField("elastic_modulus", ["pascal"])
MOMENT_INERTIA_FIELD = UnitField("moment_of_inertia", ["meter ** 4"])
POINT_LOAD_FIELD = UnitField("point_load", ["newton"])
UDL_FIELD = UnitField("distributed_load", ["newton / meter"])


class BeamDeflectionInputs(BaseModel):
    load_case: str = Field(description="Load case identifier: point_load_center or uniform_load")
    length: Any
    elastic_modulus: Any
    moment_of_inertia: Any
    point_load: Optional[Any] = None
    distributed_load: Optional[Any] = None
    samples: int = Field(default=21, ge=3, le=201)

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="before")
    def _coerce_units(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        parsed = dict(values)
        field_map = {
            "length": LENGTH_FIELD,
            "elastic_modulus": ELASTIC_MODULUS_FIELD,
            "moment_of_inertia": MOMENT_INERTIA_FIELD,
            "point_load": POINT_LOAD_FIELD,
            "distributed_load": UDL_FIELD,
        }
        for key, unit_field in field_map.items():
            raw = parsed.get(key)
            if raw is None:
                continue
            parsed[key] = unit_field.validate(raw)
        return parsed

    @model_validator(mode="after")
    def _validate_case(self) -> "BeamDeflectionInputs":
        if self.load_case not in LOAD_CASES:
            raise ValueError(f"load_case must be one of {sorted(LOAD_CASES)}")
        if self.load_case == "point_load_center" and self.point_load is None:
            raise ValueError("point_load is required for the point_load_center case")
        if self.load_case == "uniform_load" and self.distributed_load is None:
            raise ValueError("distributed_load is required for the uniform_load case")
        
        # Validate positive values
        if self.length.magnitude <= 0:
            raise ValueError("Length must be positive")
        if self.elastic_modulus.magnitude <= 0:
            raise ValueError("Elastic modulus must be positive")
        if self.moment_of_inertia.magnitude <= 0:
            raise ValueError("Moment of inertia must be positive")
        if self.point_load is not None and self.point_load.magnitude <= 0:
            raise ValueError("Load must be positive")
        if self.distributed_load is not None and self.distributed_load.magnitude <= 0:
            raise ValueError("Load must be positive")
        
        return self


@dataclass(slots=True)
class BeamDeflectionResult:
    max_deflection: float
    max_slope: float
    reaction_a: float
    reaction_b: float
    shear_support: float
    moment_max: float
    shear_profile: List[Tuple[float, float]]
    moment_profile: List[Tuple[float, float]]
    load_case: str


def run(raw_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate maximum deflection, slopes, and reactions for the given beam."""

    try:
        inputs = BeamDeflectionInputs(**raw_inputs)
    except ValidationError as exc:  # pragma: no cover
        raise ValueError(exc.errors())

    L = inputs.length.to("meter").magnitude
    E = inputs.elastic_modulus.to("pascal").magnitude
    I = inputs.moment_of_inertia.to("meter ** 4").magnitude

    if inputs.load_case == "point_load_center":
        P = inputs.point_load.to("newton").magnitude
        result = _compute_point_load_case(L, E, I, P, inputs.samples)
    else:
        w = inputs.distributed_load.to("newton / meter").magnitude
        result = _compute_uniform_load_case(L, E, I, w, inputs.samples)

    steps = _build_steps(inputs.load_case, L, E, I, result, point_load=inputs.point_load, distributed_load=inputs.distributed_load)

    units = {
        "max_deflection": "m",
        "max_slope": "rad",
        "reaction_a": "N",
        "reaction_b": "N",
        "shear_support": "N",
        "moment_max": "N·m",
    }

    # Generate warnings
    warnings = []
    
    # Deflection warning (assuming L/250 is a typical limit)
    if result.max_deflection > L / 250:
        warnings.append("Large deflection exceeds typical L/250 limit - check serviceability")
    
    # High stress warning (basic check)
    if result.moment_max > 0:  # Avoid division by zero
        if I < 1e-6:  # Very small moment of inertia
            warnings.append("Very small moment of inertia - verify beam section adequacy")
    
    # Load magnitude warnings
    if inputs.load_case == "point_load_center" and inputs.point_load is not None:
        if inputs.point_load.to("newton").magnitude > 100000:  # 100 kN
            warnings.append("Very high point load - verify structural adequacy")
    elif inputs.load_case == "uniform_load" and inputs.distributed_load is not None:
        if inputs.distributed_load.to("newton / meter").magnitude > 10000:  # 10 kN/m
            warnings.append("Very high distributed load - verify structural adequacy")

    return {
        "results": {
            "max_deflection": result.max_deflection,
            "max_slope": result.max_slope,
            "reaction_a": result.reaction_a,
            "reaction_b": result.reaction_b,
            "shear_support": result.shear_support,
            "moment_max": result.moment_max,
            "shear_profile": result.shear_profile,
            "moment_profile": result.moment_profile,
        },
        "units": units,
        "steps": steps,
        "warnings": warnings,
        "metadata": {"load_case": inputs.load_case},
    }


def _compute_point_load_case(L: float, E: float, I: float, P: float, samples: int) -> BeamDeflectionResult:
    reactions = P / 2.0
    max_deflection = P * L ** 3 / (48.0 * E * I)
    max_slope = P * L ** 2 / (16.0 * E * I)
    shear_support = reactions
    moment_max = P * L / 4.0
    shear_profile = _shear_profile_point_load(L, P, samples)
    moment_profile = _moment_profile_point_load(L, P, samples)
    return BeamDeflectionResult(
        max_deflection=max_deflection,
        max_slope=max_slope,
        reaction_a=reactions,
        reaction_b=reactions,
        shear_support=shear_support,
        moment_max=moment_max,
        shear_profile=shear_profile,
        moment_profile=moment_profile,
        load_case="point_load_center",
    )


def _compute_uniform_load_case(L: float, E: float, I: float, w: float, samples: int) -> BeamDeflectionResult:
    reactions = w * L / 2.0
    max_deflection = 5 * w * L ** 4 / (384.0 * E * I)
    max_slope = w * L ** 3 / (24.0 * E * I)
    shear_support = reactions
    moment_max = w * L ** 2 / 8.0
    shear_profile = _shear_profile_uniform(L, w, samples)
    moment_profile = _moment_profile_uniform(L, w, samples)
    return BeamDeflectionResult(
        max_deflection=max_deflection,
        max_slope=max_slope,
        reaction_a=reactions,
        reaction_b=reactions,
        shear_support=shear_support,
        moment_max=moment_max,
        shear_profile=shear_profile,
        moment_profile=moment_profile,
        load_case="uniform_load",
    )


def _shear_profile_point_load(L: float, P: float, samples: int) -> List[Tuple[float, float]]:
    positions = [i * L / (samples - 1) for i in range(samples)]
    profile = []
    for x in positions:
        shear = P / 2.0 if x <= L / 2 else -P / 2.0
        profile.append((x, shear))
    return profile


def _moment_profile_point_load(L: float, P: float, samples: int) -> List[Tuple[float, float]]:
    positions = [i * L / (samples - 1) for i in range(samples)]
    profile = []
    for x in positions:
        if x <= L / 2:
            moment = P * x / 2.0
        else:
            moment = P * (L - x) / 2.0
        profile.append((x, moment))
    return profile


def _shear_profile_uniform(L: float, w: float, samples: int) -> List[Tuple[float, float]]:
    positions = [i * L / (samples - 1) for i in range(samples)]
    profile = []
    reaction = w * L / 2.0
    for x in positions:
        shear = reaction - w * x
        profile.append((x, shear))
    return profile


def _moment_profile_uniform(L: float, w: float, samples: int) -> List[Tuple[float, float]]:
    positions = [i * L / (samples - 1) for i in range(samples)]
    reaction = w * L / 2.0
    profile = []
    for x in positions:
        moment = reaction * x - w * x ** 2 / 2.0
        profile.append((x, moment))
    return profile


def _build_steps(
    load_case: str,
    length: float,
    elastic_modulus: float,
    inertia: float,
    result: BeamDeflectionResult,
    *,
    point_load,
    distributed_load,
) -> List[Step]:
    steps: List[Step] = []
    idx = 1

    if load_case == "point_load_center":
        P = point_load.to("newton").magnitude
        steps.append(
            Step(
                index=idx,
                description="Compute support reactions",
                equation_tex=r"R_A = R_B = \frac{P}{2}",
                substitutions=[Substitution("P", P, "N")],
                result_value=result.reaction_a,
                result_units="N",
            )
        )
        idx += 1
        steps.append(
            Step(
                index=idx,
                description="Maximum midspan deflection",
                equation_tex=r"y_{max} = \frac{P L^3}{48 E I}",
                substitutions=[
                    Substitution("P", P, "N"),
                    Substitution("L", length, "m"),
                    Substitution("E", elastic_modulus, "Pa"),
                    Substitution("I", inertia, "m^4"),
                ],
                result_value=result.max_deflection,
                result_units="m",
            )
        )
        idx += 1
        steps.append(
            Step(
                index=idx,
                description="Slope at supports",
                equation_tex=r"\theta = \frac{P L^2}{16 E I}",
                substitutions=[
                    Substitution("P", P, "N"),
                    Substitution("L", length, "m"),
                    Substitution("E", elastic_modulus, "Pa"),
                    Substitution("I", inertia, "m^4"),
                ],
                result_value=result.max_slope,
                result_units="rad",
            )
        )
    else:
        w = distributed_load.to("newton / meter").magnitude
        steps.append(
            Step(
                index=idx,
                description="Compute support reactions",
                equation_tex=r"R_A = R_B = \frac{w L}{2}",
                substitutions=[
                    Substitution("w", w, "N/m"),
                    Substitution("L", length, "m"),
                ],
                result_value=result.reaction_a,
                result_units="N",
            )
        )
        idx += 1
        steps.append(
            Step(
                index=idx,
                description="Maximum midspan deflection",
                equation_tex=r"y_{max} = \frac{5 w L^4}{384 E I}",
                substitutions=[
                    Substitution("w", w, "N/m"),
                    Substitution("L", length, "m"),
                    Substitution("E", elastic_modulus, "Pa"),
                    Substitution("I", inertia, "m^4"),
                ],
                result_value=result.max_deflection,
                result_units="m",
            )
        )
        idx += 1
        steps.append(
            Step(
                index=idx,
                description="Slope at supports",
                equation_tex=r"\theta = \frac{w L^3}{24 E I}",
                substitutions=[
                    Substitution("w", w, "N/m"),
                    Substitution("L", length, "m"),
                    Substitution("E", elastic_modulus, "Pa"),
                    Substitution("I", inertia, "m^4"),
                ],
                result_value=result.max_slope,
                result_units="rad",
            )
        )

    steps.append(
        Step(
            index=idx + 1,
            description="Maximum bending moment",
            equation_tex=r"M_{max} = R_A \frac{L}{2}",
            substitutions=[
                Substitution("R_A", result.reaction_a, "N"),
                Substitution("L", length, "m"),
            ],
            result_value=result.moment_max,
            result_units="N·m",
        )
    )

    return steps

