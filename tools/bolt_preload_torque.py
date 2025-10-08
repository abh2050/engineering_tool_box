"""Bolted joint preload and torque calculator with friction breakdown."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError, model_validator

from core.state import Step, Substitution
from core.units import UnitField


FORCE_FIELD = UnitField("force", ["newton"])
DIAMETER_FIELD = UnitField("diameter", ["meter"])
PITCH_FIELD = UnitField("thread_pitch", ["meter"])


class BoltTorqueInputs(BaseModel):
    preload: Any
    nominal_diameter: Any
    thread_pitch: Any
    nut_factor: float = Field(gt=0)
    friction_thread: float = Field(gt=0)
    friction_bearing: float = Field(gt=0)
    external_load: Optional[Any] = None
    proof_load: Optional[Any] = None

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="before")
    def _coerce_units(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        parsed = dict(values)
        mapping = {
            "preload": FORCE_FIELD,
            "nominal_diameter": DIAMETER_FIELD,
            "thread_pitch": PITCH_FIELD,
            "external_load": FORCE_FIELD,
            "proof_load": FORCE_FIELD,
        }
        for key, unit_field in mapping.items():
            raw = parsed.get(key)
            if raw is None:
                continue
            parsed[key] = unit_field.validate(raw)
        return parsed
    
    @model_validator(mode="after")
    def _validate_inputs(self) -> "BoltTorqueInputs":
        # Validate positive values
        if self.preload.magnitude <= 0:
            raise ValueError("Preload must be positive")
        if self.nominal_diameter.magnitude <= 0:
            raise ValueError("Nominal diameter must be positive")
        if self.thread_pitch.magnitude <= 0:
            raise ValueError("Thread pitch must be positive")
        if self.nut_factor <= 0:
            raise ValueError("Nut factor must be positive")
        if self.friction_thread <= 0:
            raise ValueError("Thread friction must be positive")
        if self.friction_bearing <= 0:
            raise ValueError("Bearing friction must be positive")
        
        # Validate realistic ranges
        if self.nut_factor > 1.0:
            raise ValueError("Nut factor should typically be less than 1.0")
        if self.friction_thread > 1.0:
            raise ValueError("Thread friction coefficient should be less than 1.0")
        if self.friction_bearing > 1.0:
            raise ValueError("Bearing friction coefficient should be less than 1.0")
        
        # External load validation
        if self.external_load is not None and self.external_load.magnitude < 0:
            raise ValueError("External load must be non-negative")
        
        # Proof load validation
        if self.proof_load is not None and self.proof_load.magnitude <= 0:
            raise ValueError("Proof load must be positive")
        
        return self


@dataclass(slots=True)
class BoltTorqueResult:
    recommended_torque: float
    thread_torque: float
    bearing_torque: float
    torque_from_breakdown: float
    preload_stress_area: float
    preload_stress: float
    clamp_safety_factor: Optional[float]
    clamp_residual: Optional[float]
    proof_safety_factor: Optional[float]
    effective_nut_factor: float


THREAD_HALF_ANGLE_RAD = math.radians(30.0)


def run(raw_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Compute bolt torque recommendations and safety factors."""

    try:
        inputs = BoltTorqueInputs(**raw_inputs)
    except ValidationError as exc:  # pragma: no cover
        raise ValueError(exc.errors())

    preload = inputs.preload.to("newton").magnitude
    nominal_diameter = inputs.nominal_diameter.to("meter").magnitude
    pitch = inputs.thread_pitch.to("meter").magnitude
    external_load = inputs.external_load.to("newton").magnitude if inputs.external_load is not None else None
    proof_load = inputs.proof_load.to("newton").magnitude if inputs.proof_load is not None else None

    tensile_area = _tensile_stress_area(nominal_diameter, pitch)
    preload_stress = preload / tensile_area

    recommended_torque = inputs.nut_factor * preload * nominal_diameter

    pitch_diameter = nominal_diameter - 0.64952 * pitch
    lead_angle = math.atan(pitch / (math.pi * pitch_diameter))
    thread_torque = preload * pitch_diameter / 2.0 * (
        math.tan(lead_angle) + inputs.friction_thread / math.cos(THREAD_HALF_ANGLE_RAD)
    )
    bearing_diameter = 1.5 * nominal_diameter
    bearing_torque = inputs.friction_bearing * preload * bearing_diameter / 2.0
    torque_breakdown = thread_torque + bearing_torque

    effective_k = torque_breakdown / (preload * nominal_diameter)

    clamp_safety = None
    residual = None
    if external_load is not None and external_load > 0:
        clamp_safety = preload / external_load
        residual = preload - external_load

    proof_safety = None
    if proof_load is not None and proof_load > 0:
        proof_safety = proof_load / preload

    result = BoltTorqueResult(
        recommended_torque=recommended_torque,
        thread_torque=thread_torque,
        bearing_torque=bearing_torque,
        torque_from_breakdown=torque_breakdown,
        preload_stress_area=tensile_area,
        preload_stress=preload_stress,
        clamp_safety_factor=clamp_safety,
        clamp_residual=residual,
        proof_safety_factor=proof_safety,
        effective_nut_factor=effective_k,
    )

    steps = _build_steps(
        preload=preload,
        nominal_diameter=nominal_diameter,
        pitch=pitch,
        tensile_area=tensile_area,
        recommended_torque=recommended_torque,
        thread_torque=thread_torque,
        bearing_torque=bearing_torque,
        torque_breakdown=torque_breakdown,
        clamp_safety=clamp_safety,
        residual=residual,
        proof_safety=proof_safety,
        inputs=inputs,
    )

    warnings: List[str] = []
    if residual is not None and residual <= 0:
        warnings.append("External load exceeds preload; joint may separate.")
    if proof_safety is not None and proof_safety < 1.0:
        warnings.append("Preload exceeds proof load; bolt may yield.")
    if proof_safety is not None and 1.0 <= proof_safety < 1.5:
        warnings.append("Low proof load safety factor - consider reducing preload or increasing proof load")
    
    # Additional warnings
    if clamp_safety is not None and clamp_safety < 2.0:
        warnings.append("Low clamping safety factor - consider increasing preload")
    
    if effective_k > 0.5:
        warnings.append("High effective nut factor - verify friction coefficients")
    
    if inputs.nut_factor > 0.4:
        warnings.append("High nut factor - consider lubrication or different fastener")
    
    # Stress warnings
    if preload_stress > 400e6:  # 400 MPa is typical limit for many steels
        warnings.append("High preload stress - verify bolt material capacity")

    return {
        "results": {
            "recommended_torque": result.recommended_torque,
            "thread_torque": result.thread_torque,
            "bearing_torque": result.bearing_torque,
            "torque_from_breakdown": result.torque_from_breakdown,
            "preload_stress_area": result.preload_stress_area,
            "preload_stress": result.preload_stress,
            "clamp_safety_factor": result.clamp_safety_factor,
            "clamp_residual": result.clamp_residual,
            "proof_safety_factor": result.proof_safety_factor,
            "effective_nut_factor": result.effective_nut_factor,
        },
        "units": {
            "recommended_torque": "N·m",
            "thread_torque": "N·m",
            "bearing_torque": "N·m",
            "torque_from_breakdown": "N·m",
            "preload_stress_area": "m^2",
            "preload_stress": "Pa",
            "clamp_safety_factor": "",
            "clamp_residual": "N",
            "proof_safety_factor": "",
            "effective_nut_factor": "",
        },
        "steps": steps,
        "warnings": warnings,
        "metadata": {
            "thread_pitch_diameter": pitch_diameter,
            "bearing_diameter": bearing_diameter,
        },
    }


def _tensile_stress_area(d: float, p: float) -> float:
    return math.pi / 4.0 * (d - 0.9382 * p) ** 2


def _build_steps(
    *,
    preload: float,
    nominal_diameter: float,
    pitch: float,
    tensile_area: float,
    recommended_torque: float,
    thread_torque: float,
    bearing_torque: float,
    torque_breakdown: float,
    clamp_safety: Optional[float],
    residual: Optional[float],
    proof_safety: Optional[float],
    inputs: BoltTorqueInputs,
) -> List[Step]:
    steps: List[Step] = []

    steps.append(
        Step(
            index=1,
            description="Determine tensile stress area",
            equation_tex=r"A_t = \frac{\pi}{4}(d - 0.9382p)^2",
            substitutions=[
                Substitution("d", nominal_diameter, "m"),
                Substitution("p", pitch, "m"),
            ],
            result_value=tensile_area,
            result_units="m^2",
        )
    )

    steps.append(
        Step(
            index=2,
            description="Torque via nut factor relationship",
            equation_tex=r"T = K F d",
            substitutions=[
                Substitution("K", inputs.nut_factor, ""),
                Substitution("F", preload, "N"),
                Substitution("d", nominal_diameter, "m"),
            ],
            result_value=recommended_torque,
            result_units="N·m",
        )
    )

    steps.append(
        Step(
            index=3,
            description="Thread torque contribution",
            equation_tex=r"T_{thread} = \frac{F d_m}{2} \left(\tan\lambda + \frac{\mu_t}{\cos\alpha}\right)",
            substitutions=[
                Substitution("F", preload, "N"),
                Substitution("d_m", nominal_diameter - 0.64952 * pitch, "m"),
                Substitution(r"\lambda", math.degrees(math.atan(pitch / (math.pi * (nominal_diameter - 0.64952 * pitch)))), "deg"),
                Substitution(r"\mu_t", inputs.friction_thread, ""),
                Substitution(r"\alpha", 60.0, "deg"),
            ],
            result_value=thread_torque,
            result_units="N·m",
        )
    )

    steps.append(
        Step(
            index=4,
            description="Bearing friction torque",
            equation_tex=r"T_{bearing} = \mu_b F \frac{d_b}{2}",
            substitutions=[
                Substitution(r"\mu_b", inputs.friction_bearing, ""),
                Substitution("F", preload, "N"),
                Substitution("d_b", 1.5 * nominal_diameter, "m"),
            ],
            result_value=bearing_torque,
            result_units="N·m",
        )
    )

    steps.append(
        Step(
            index=5,
            description="Torque balance",
            equation_tex=r"T = T_{thread} + T_{bearing}",
            substitutions=[
                Substitution("T_{thread}", thread_torque, "N·m"),
                Substitution("T_{bearing}", bearing_torque, "N·m"),
            ],
            result_value=torque_breakdown,
            result_units="N·m",
        )
    )

    if clamp_safety is not None:
        steps.append(
            Step(
                index=6,
                description="Clamp safety factor",
                equation_tex=r"SF_{clamp} = \frac{F_i}{F_{ext}}",
                substitutions=[
                    Substitution("F_i", preload, "N"),
                    Substitution("F_{ext}", preload / clamp_safety, "N"),
                ],
                result_value=clamp_safety,
            )
        )
        if residual is not None:
            steps.append(
                Step(
                    index=7,
                    description="Residual clamp load",
                    equation_tex=r"F_{res} = F_i - F_{ext}",
                    substitutions=[
                        Substitution("F_i", preload, "N"),
                        Substitution("F_{ext}", preload - residual, "N"),
                    ],
                    result_value=residual,
                    result_units="N",
                )
            )

    if proof_safety is not None:
        steps.append(
            Step(
                index=8,
                description="Proof load safety factor",
                equation_tex=r"SF_{proof} = \frac{F_{proof}}{F_i}",
                substitutions=[
                    Substitution("F_{proof}", preload * proof_safety, "N"),
                    Substitution("F_i", preload, "N"),
                ],
                result_value=proof_safety,
            )
        )

    return steps

