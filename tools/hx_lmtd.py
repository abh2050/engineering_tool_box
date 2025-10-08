"""Counter-current heat exchanger siz    @model_validator(mode="before")
    def _coerce_units(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        parsed = dict(values)
        mapping = {
            "mass_flow_hot": MASS_FLOW_FIELD,
            "mass_flow_cold": MASS_FLOW_FIELD,
            "cp_hot": HEAT_CAPACITY_FIELD,
            "cp_cold": HEAT_CAPACITY_FIELD,
            "t_hot_in": TEMPERATURE_FIELD,
            "t_hot_out": TEMPERATURE_FIELD,
            "t_cold_in": TEMPERATURE_FIELD,
            "t_cold_out": TEMPERATURE_FIELD,
            "overall_heat_transfer_coefficient": U_FIELD,
        }
        for key, unit_field in mapping.items():
            raw = parsed.get(key)
            if raw is None:
                continue
            parsed[key] = unit_field.validate(raw)
        return parsedd."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError, model_validator

from core.state import Step, Substitution
from core.units import UnitField, get_unit_registry


ureg = get_unit_registry()


MASS_FLOW_FIELD = UnitField("mass_flow_rate", ["kilogram / second"])
HEAT_CAPACITY_FIELD = UnitField("heat_capacity", ["joule / kilogram / kelvin", "Btu_per_lb_degF"])
TEMPERATURE_FIELD = UnitField("temperature", ["kelvin"])
U_FIELD = UnitField("overall_heat_transfer_coefficient", ["watt / meter ** 2 / kelvin", "british_thermal_unit / hour / foot**2 / degree_Fahrenheit"])


class HeatExchangerInputs(BaseModel):
    mass_flow_hot: Any
    mass_flow_cold: Any
    cp_hot: Any
    cp_cold: Any
    t_hot_in: Any
    t_hot_out: Optional[Any] = None
    t_cold_in: Any
    t_cold_out: Optional[Any] = None
    overall_heat_transfer_coefficient: Any
    correction_factor: float = Field(default=1.0, gt=0, le=1.2)

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="before")
    def _coerce_units(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        parsed = dict(values)
        unit_map = {
            "mass_flow_hot": MASS_FLOW_FIELD,
            "mass_flow_cold": MASS_FLOW_FIELD,
            "cp_hot": HEAT_CAPACITY_FIELD,
            "cp_cold": HEAT_CAPACITY_FIELD,
            "t_hot_in": TEMPERATURE_FIELD,
            "t_hot_out": TEMPERATURE_FIELD,
            "t_cold_in": TEMPERATURE_FIELD,
            "t_cold_out": TEMPERATURE_FIELD,
            "overall_heat_transfer_coefficient": U_FIELD,
        }
        for key, unit_field in unit_map.items():
            raw = parsed.get(key)
            if raw is None:
                continue
            parsed[key] = unit_field.validate(raw)
        return parsed

    @model_validator(mode="after")
    def _validate_temperatures(self) -> "HeatExchangerInputs":
        if self.t_hot_out is None and self.t_cold_out is None:
            raise ValueError("Provide at least one outlet temperature (hot or cold).")
        
        # Validate positive values
        if self.mass_flow_hot.magnitude <= 0:
            raise ValueError("Hot fluid mass flow rate must be positive")
        if self.mass_flow_cold.magnitude <= 0:
            raise ValueError("Cold fluid mass flow rate must be positive")
        if self.cp_hot.magnitude <= 0:
            raise ValueError("Hot fluid heat capacity must be positive")
        if self.cp_cold.magnitude <= 0:
            raise ValueError("Cold fluid heat capacity must be positive")
        if self.overall_heat_transfer_coefficient.magnitude <= 0:
            raise ValueError("Overall heat transfer coefficient must be positive")
        
        # Temperature validation
        t_hot_in_k = self.t_hot_in.to("kelvin").magnitude
        t_cold_in_k = self.t_cold_in.to("kelvin").magnitude
        
        if t_hot_in_k <= t_cold_in_k:
            raise ValueError("Hot inlet temperature must be greater than cold inlet temperature")
            
        if self.t_hot_out is not None:
            t_hot_out_k = self.t_hot_out.to("kelvin").magnitude
            if t_hot_out_k >= t_hot_in_k:
                raise ValueError("Hot outlet temperature must be less than hot inlet temperature")
                
        if self.t_cold_out is not None:
            t_cold_out_k = self.t_cold_out.to("kelvin").magnitude
            if t_cold_out_k <= t_cold_in_k:
                raise ValueError("Cold outlet temperature must be greater than cold inlet temperature")
        
        return self


@dataclass(slots=True)
class HeatExchangerResult:
    duty: float
    lmtd: float
    area: float
    hot_outlet: float
    cold_outlet: float
    delta_t1: float
    delta_t2: float
    heat_capacity_hot: float
    heat_capacity_cold: float
    heat_capacity_ratio: float


def run(raw_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Compute heat duty, LMTD, and required surface area."""

    try:
        inputs = HeatExchangerInputs(**raw_inputs)
    except ValidationError as exc:  # pragma: no cover
        raise ValueError(exc.errors())

    m_hot = inputs.mass_flow_hot.to("kilogram / second").magnitude
    m_cold = inputs.mass_flow_cold.to("kilogram / second").magnitude
    cp_hot = inputs.cp_hot.to("joule / kilogram / kelvin").magnitude
    cp_cold = inputs.cp_cold.to("joule / kilogram / kelvin").magnitude
    Th_in = inputs.t_hot_in.to("kelvin").magnitude
    Tc_in = inputs.t_cold_in.to("kelvin").magnitude
    Th_out = inputs.t_hot_out.to("kelvin").magnitude if inputs.t_hot_out is not None else None
    Tc_out = inputs.t_cold_out.to("kelvin").magnitude if inputs.t_cold_out is not None else None
    U = inputs.overall_heat_transfer_coefficient.to("watt / meter ** 2 / kelvin").magnitude
    F = inputs.correction_factor

    C_hot = m_hot * cp_hot
    C_cold = m_cold * cp_cold

    duty, Th_out_final, Tc_out_final = _solve_temperatures(Th_in, Tc_in, Th_out, Tc_out, C_hot, C_cold)

    delta_t1 = Th_in - Tc_out_final
    delta_t2 = Th_out_final - Tc_in
    if delta_t1 <= 0 or delta_t2 <= 0:
        raise ValueError("Temperature differences must be positive for counter-current LMTD.")

    lmtd = _safe_lmtd(delta_t1, delta_t2)
    area = duty / (U * F * lmtd)
    C_ratio = min(C_hot, C_cold) / max(C_hot, C_cold)

    result = HeatExchangerResult(
        duty=duty,
        lmtd=lmtd,
        area=area,
        hot_outlet=Th_out_final,
        cold_outlet=Tc_out_final,
        delta_t1=delta_t1,
        delta_t2=delta_t2,
        heat_capacity_hot=C_hot,
        heat_capacity_cold=C_cold,
        heat_capacity_ratio=C_ratio,
    )

    steps = _build_steps(
        duty=duty,
        C_hot=C_hot,
        C_cold=C_cold,
        delta_t1=delta_t1,
        delta_t2=delta_t2,
        lmtd=lmtd,
        area=area,
        Th_in=Th_in,
        Tc_in=Tc_in,
        Th_out=Th_out_final,
        Tc_out=Tc_out_final,
        U=U,
        F=F,
    )

    warnings: List[str] = []
    if abs(delta_t1 - delta_t2) < 1e-3:
        warnings.append("Temperature differences nearly equal; LMTD approximated by arithmetic mean.")
    
    # Additional warnings
    if min(delta_t1, delta_t2) < 5:  # Less than 5K temperature difference
        warnings.append("Low temperature difference - may result in very large heat transfer area")
    
    if C_ratio < 0.1:
        warnings.append("Very unbalanced heat capacity rates - consider flow optimization")
    
    if U < 50:  # Very low overall heat transfer coefficient
        warnings.append("Low overall heat transfer coefficient - verify heat exchanger type selection")
    
    if area > 1000:  # Very large area
        warnings.append("Very large heat transfer area required - consider alternative design")

    return {
        "results": {
            "duty": result.duty,
            "lmtd": result.lmtd,
            "area": result.area,
            "hot_outlet_temp": ureg.Quantity(result.hot_outlet, "kelvin").to("degC").magnitude,
            "cold_outlet_temp": ureg.Quantity(result.cold_outlet, "kelvin").to("degC").magnitude,
            "delta_t1": result.delta_t1,
            "delta_t2": result.delta_t2,
            "heat_capacity_rate_hot": result.heat_capacity_hot,
            "heat_capacity_rate_cold": result.heat_capacity_cold,
            "heat_capacity_ratio": result.heat_capacity_ratio,
        },
        "units": {
            "duty": "W",
            "lmtd": "K",
            "area": "m^2",
            "hot_outlet_temp": "°C",
            "cold_outlet_temp": "°C",
            "delta_t1": "K",
            "delta_t2": "K",
            "heat_capacity_rate_hot": "W/K",
            "heat_capacity_rate_cold": "W/K",
            "heat_capacity_ratio": "",
        },
        "steps": steps,
        "warnings": warnings,
        "metadata": {"correction_factor": F},
    }


def _solve_temperatures(
    Th_in: float,
    Tc_in: float,
    Th_out: Optional[float],
    Tc_out: Optional[float],
    C_hot: float,
    C_cold: float,
) -> tuple[float, float, float]:
    if Th_out is not None and Tc_out is not None:
        Q_hot = C_hot * (Th_in - Th_out)
        Q_cold = C_cold * (Tc_out - Tc_in)
        if abs(Q_hot - Q_cold) > 0.02 * max(abs(Q_hot), abs(Q_cold), 1.0):
            raise ValueError("Hot and cold side heat duties do not balance within 2%.")
        duty = 0.5 * (Q_hot + Q_cold)
        return duty, Th_out, Tc_out

    if Tc_out is None and Th_out is None:
        raise ValueError("At least one outlet temperature must be provided.")

    if Tc_out is None:
        # compute cold outlet from hot outlet duty
        duty = C_hot * (Th_in - Th_out)
        Tc_out = Tc_in + duty / C_cold
        return duty, Th_out, Tc_out

    # else Th_out is None; compute from cold outlet target
    duty = C_cold * (Tc_out - Tc_in)
    Th_out = Th_in - duty / C_hot
    if Th_out <= Tc_in:
        raise ValueError("Computed hot outlet temperature must remain above cold inlet temperature for counter-current flow.")
    return duty, Th_out, Tc_out


def _safe_lmtd(delta_t1: float, delta_t2: float) -> float:
    if abs(delta_t1 - delta_t2) < 1e-6:
        return (delta_t1 + delta_t2) / 2.0
    return (delta_t1 - delta_t2) / math.log(delta_t1 / delta_t2)


def _build_steps(
    *,
    duty: float,
    C_hot: float,
    C_cold: float,
    delta_t1: float,
    delta_t2: float,
    lmtd: float,
    area: float,
    Th_in: float,
    Tc_in: float,
    Th_out: float,
    Tc_out: float,
    U: float,
    F: float,
) -> List[Step]:
    steps: List[Step] = []

    steps.append(
        Step(
            index=1,
            description="Energy balance to determine heat duty",
            equation_tex=r"Q = C_{cold} (T_{c,out} - T_{c,in}) = C_{hot} (T_{h,in} - T_{h,out})",
            substitutions=[
                Substitution(r"C_{cold}", C_cold, "W/K"),
                Substitution(r"C_{hot}", C_hot, "W/K"),
                Substitution(r"T_{c,out}", Tc_out, "K"),
                Substitution(r"T_{c,in}", Tc_in, "K"),
                Substitution(r"T_{h,in}", Th_in, "K"),
                Substitution(r"T_{h,out}", Th_out, "K"),
            ],
            result_value=duty,
            result_units="W",
        )
    )

    steps.append(
        Step(
            index=2,
            description="Log mean temperature difference",
            equation_tex=r"\Delta T_{lm} = \frac{\Delta T_1 - \Delta T_2}{\ln(\Delta T_1 / \Delta T_2)}",
            substitutions=[
                Substitution(r"\Delta T_1", delta_t1, "K"),
                Substitution(r"\Delta T_2", delta_t2, "K"),
            ],
            result_value=lmtd,
            result_units="K",
        )
    )

    steps.append(
        Step(
            index=3,
            description="Required surface area",
            equation_tex=r"A = \frac{Q}{U F \Delta T_{lm}}",
            substitutions=[
                Substitution("Q", duty, "W"),
                Substitution("U", U, "W/m^2/K"),
                Substitution("F", F, ""),
                Substitution(r"\Delta T_{lm}", lmtd, "K"),
            ],
            result_value=area,
            result_units="m^2",
        )
    )

    return steps

