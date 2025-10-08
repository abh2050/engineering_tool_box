"""Unit tests covering the five engineering calculation tools."""

import pytest

from tools import (
    beam_deflection,
    bolt_preload_torque,
    hx_lmtd,
    pipe_pressure_drop,
    pump_power_npsh,
)


def quantity(value, units):
    return {"value": value, "units": units}


def test_pipe_pressure_drop_water_example():
    result = pipe_pressure_drop.run(
        {
            "volumetric_flow_rate": quantity(0.012, "m^3/s"),
            "diameter": quantity(0.1, "m"),
            "length": quantity(50, "m"),
            "roughness": quantity(0.000045, "m"),
            "density": quantity(998, "kg/m^3"),
            "dynamic_viscosity": quantity(1e-3, "Pa*s"),
        }
    )

    outputs = result["results"]
    assert outputs["velocity"] == pytest.approx(1.5279, rel=1e-3)
    assert pytest.approx(outputs["friction_factor"], rel=5e-2) == 0.019
    assert pytest.approx(outputs["delta_p"], rel=5e-2) == 1.1e4
    assert len(result["steps"]) >= 6


def test_beam_deflection_point_load():
    result = beam_deflection.run(
        {
            "load_case": "point_load_center",
            "length": quantity(4.0, "m"),
            "elastic_modulus": quantity(200e9, "Pa"),
            "moment_of_inertia": quantity(8e-6, "m^4"),
            "point_load": quantity(10e3, "N"),
        }
    )
    outputs = result["results"]
    assert pytest.approx(outputs["max_deflection"], rel=1e-3) == 8.33e-3
    assert pytest.approx(outputs["moment_max"], rel=1e-6) == 1.0e4
    assert result["metadata"]["load_case"] == "point_load_center"


def test_pump_power_npsh():
    result = pump_power_npsh.run(
        {
            "volumetric_flow_rate": quantity(0.05, "m^3/s"),
            "head": quantity(30, "m"),
            "density": quantity(998, "kg/m^3"),
            "efficiency": {"value": 75, "units": "%"},
            "suction_pressure": quantity(120000, "Pa"),
            "vapor_pressure": quantity(3170, "Pa"),
            "suction_elevation": quantity(2.0, "m"),
            "suction_losses": quantity(1.5, "m"),
            "npsh_required": quantity(10.0, "m"),
        }
    )
    outputs = result["results"]
    assert pytest.approx(outputs["hydraulic_power"], rel=1e-3) == 1.468e4
    assert outputs["npsha"] == pytest.approx(12.437, rel=1e-3)
    assert outputs["adequate_npsh"] is True


def test_heat_exchanger_lmtd():
    result = hx_lmtd.run(
        {
            "mass_flow_hot": quantity(1.2, "kg/s"),
            "mass_flow_cold": quantity(1.0, "kg/s"),
            "cp_hot": quantity(2200, "J/kg/K"),
            "cp_cold": quantity(4184, "J/kg/K"),
            "t_hot_in": quantity(120, "degC"),
            "t_cold_in": quantity(20, "degC"),
            "t_cold_out": quantity(60, "degC"),
            "overall_heat_transfer_coefficient": quantity(250, "W/m^2/K"),
            "correction_factor": 0.95,
        }
    )

    outputs = result["results"]
    assert pytest.approx(outputs["duty"], rel=5e-3) == 1.6736e5
    assert pytest.approx(outputs["lmtd"], rel=5e-3) == 47.3
    assert pytest.approx(outputs["area"], rel=5e-3) == 14.9


def test_bolt_preload_torque():
    result = bolt_preload_torque.run(
        {
            "preload": quantity(30e3, "N"),
            "nominal_diameter": quantity(0.016, "m"),
            "thread_pitch": quantity(0.002, "m"),
            "nut_factor": 0.2,
            "friction_thread": 0.15,
            "friction_bearing": 0.12,
            "external_load": quantity(10e3, "N"),
            "proof_load": quantity(45e3, "N"),
        }
    )
    outputs = result["results"]
    assert pytest.approx(outputs["recommended_torque"], rel=1e-3) == 96.0
    assert outputs["clamp_safety_factor"] > 2.9
    assert outputs["proof_safety_factor"] > 1.4
    assert len(result["steps"]) >= 5

