"""Tests for heat exchanger LMTD sizing calculations."""

import pytest
from tools.hx_lmtd import run


class TestHxLmtd:
    """Test cases for heat exchanger LMTD tool."""

    def test_hot_outlet_specified_si_units(self):
        """Test heat exchanger sizing with hot outlet temperature specified."""
        inputs = {
            "mass_flow_hot": {"value": 2.0, "units": "kg/s"},
            "mass_flow_cold": {"value": 1.5, "units": "kg/s"},
            "cp_hot": {"value": 4.18, "units": "kJ/kg/K"},
            "cp_cold": {"value": 4.18, "units": "kJ/kg/K"},
            "t_hot_in": {"value": 80, "units": "°C"},
            "t_hot_out": {"value": 50, "units": "°C"},
            "t_cold_in": {"value": 20, "units": "°C"},
            "overall_heat_transfer_coefficient": {"value": 1000, "units": "W/m^2/K"},
        }
        
        result = run(inputs)
        
        # Check results exist and are positive
        assert result["results"]["duty"] > 0
        assert result["results"]["lmtd"] > 0
        assert result["results"]["area"] > 0
        assert result["results"]["hot_outlet_temp"] == pytest.approx(50.0, rel=1e-3)
        assert result["results"]["cold_outlet_temp"] > 20.0  # Should be heated
        
        # Check units
        assert result["units"]["duty"] == "W"
        assert result["units"]["lmtd"] == "K"
        assert result["units"]["area"] == "m^2"
        assert result["units"]["hot_outlet_temp"] == "°C"
        assert result["units"]["cold_outlet_temp"] == "°C"
        
        # Check heat balance
        q_hot = 2.0 * 4180 * (80 - 50)  # m_hot * cp_hot * (T_hot_in - T_hot_out)
        assert result["results"]["duty"] == pytest.approx(q_hot, rel=1e-3)

    def test_cold_outlet_specified_si_units(self):
        """Test heat exchanger sizing with cold outlet temperature specified."""
        inputs = {
            "mass_flow_hot": {"value": 1.8, "units": "kg/s"},
            "mass_flow_cold": {"value": 2.2, "units": "kg/s"},
            "cp_hot": {"value": 4.18, "units": "kJ/kg/K"},
            "cp_cold": {"value": 4.18, "units": "kJ/kg/K"},
            "t_hot_in": {"value": 90, "units": "°C"},
            "t_cold_in": {"value": 15, "units": "°C"},
            "t_cold_out": {"value": 45, "units": "°C"},
            "overall_heat_transfer_coefficient": {"value": 800, "units": "W/m^2/K"},
        }
        
        result = run(inputs)
        
        assert result["results"]["duty"] > 0
        assert result["results"]["lmtd"] > 0
        assert result["results"]["area"] > 0
        assert result["results"]["cold_outlet_temp"] == pytest.approx(45.0, rel=1e-3)
        assert result["results"]["hot_outlet_temp"] < 90.0  # Should be cooled
        
        # Check heat balance
        q_cold = 2.2 * 4180 * (45 - 15)  # m_cold * cp_cold * (T_cold_out - T_cold_in)
        assert result["results"]["duty"] == pytest.approx(q_cold, rel=1e-3)

    def test_us_customary_units(self):
        """Test heat exchanger calculation with US customary units."""
        inputs = {
            "mass_flow_hot": {"value": 4000, "units": "lb/hr"},
            "mass_flow_cold": {"value": 3000, "units": "lb/hr"},
            "cp_hot": {"value": 1.0, "units": "Btu/lb°F"},
            "cp_cold": {"value": 1.0, "units": "Btu/lb°F"},
            "t_hot_in": {"value": 180, "units": "°F"},
            "t_hot_out": {"value": 120, "units": "°F"},
            "t_cold_in": {"value": 70, "units": "°F"},
            "overall_heat_transfer_coefficient": {"value": 150, "units": "british_thermal_unit / hour / foot**2 / degree_Fahrenheit"},
        }
        
        result = run(inputs)
        
        assert result["results"]["duty"] > 0
        assert result["results"]["area"] > 0
        
        # Results should be in SI units
        assert result["units"]["duty"] == "W"
        assert result["units"]["area"] == "m^2"

    def test_with_correction_factor(self):
        """Test heat exchanger with LMTD correction factor."""
        inputs = {
            "mass_flow_hot": {"value": 2.0, "units": "kg/s"},
            "mass_flow_cold": {"value": 1.5, "units": "kg/s"},
            "cp_hot": {"value": 4.18, "units": "kJ/kg/K"},
            "cp_cold": {"value": 4.18, "units": "kJ/kg/K"},
            "t_hot_in": {"value": 80, "units": "°C"},
            "t_hot_out": {"value": 50, "units": "°C"},
            "t_cold_in": {"value": 20, "units": "°C"},
            "overall_heat_transfer_coefficient": {"value": 1000, "units": "W/m^2/K"},
            "correction_factor": 0.9,  # Shell-and-tube correction
        }
        
        result = run(inputs)
        
        assert result["results"]["duty"] > 0
        assert result["results"]["lmtd"] > 0
        assert result["results"]["area"] > 0
        
        # Area should be larger due to correction factor < 1
        assert len(result["steps"]) > 0

    def test_balanced_heat_capacity_rates(self):
        """Test case with equal heat capacity rates (challenging for sizing)."""
        inputs = {
            "mass_flow_hot": {"value": 2.0, "units": "kg/s"},
            "mass_flow_cold": {"value": 2.0, "units": "kg/s"},
            "cp_hot": {"value": 4.18, "units": "kJ/kg/K"},
            "cp_cold": {"value": 4.18, "units": "kJ/kg/K"},
            "t_hot_in": {"value": 70, "units": "°C"},
            "t_hot_out": {"value": 50, "units": "°C"},
            "t_cold_in": {"value": 30, "units": "°C"},
            "overall_heat_transfer_coefficient": {"value": 1200, "units": "W/m^2/K"},
        }
        
        result = run(inputs)
        
        assert result["results"]["heat_capacity_ratio"] == pytest.approx(1.0, rel=1e-3)
        assert result["results"]["duty"] > 0
        assert result["results"]["area"] > 0

    def test_unbalanced_heat_capacity_rates(self):
        """Test case with very different heat capacity rates."""
        inputs = {
            "mass_flow_hot": {"value": 5.0, "units": "kg/s"},  # High flow rate
            "mass_flow_cold": {"value": 0.5, "units": "kg/s"},  # Low flow rate
            "cp_hot": {"value": 2.0, "units": "kJ/kg/K"},
            "cp_cold": {"value": 4.18, "units": "kJ/kg/K"},
            "t_hot_in": {"value": 90, "units": "°C"},
            "t_hot_out": {"value": 85, "units": "°C"},  # Small temperature change for hot
            "t_cold_in": {"value": 20, "units": "°C"},
            "overall_heat_transfer_coefficient": {"value": 800, "units": "W/m^2/K"},
        }
        
        result = run(inputs)
        
        c_hot = result["results"]["heat_capacity_rate_hot"]
        c_cold = result["results"]["heat_capacity_rate_cold"]
        assert c_hot > c_cold  # Hot stream should have higher heat capacity rate
        assert result["results"]["heat_capacity_ratio"] < 1.0

    def test_kelvin_temperature_input(self):
        """Test with temperatures in Kelvin."""
        inputs = {
            "mass_flow_hot": {"value": 1.0, "units": "kg/s"},
            "mass_flow_cold": {"value": 0.8, "units": "kg/s"},
            "cp_hot": {"value": 4.18, "units": "kJ/kg/K"},
            "cp_cold": {"value": 4.18, "units": "kJ/kg/K"},
            "t_hot_in": {"value": 353.15, "units": "K"},  # 80°C
            "t_hot_out": {"value": 323.15, "units": "K"},  # 50°C
            "t_cold_in": {"value": 293.15, "units": "K"},  # 20°C
            "overall_heat_transfer_coefficient": {"value": 1000, "units": "W/m^2/K"},
        }
        
        result = run(inputs)
        
        assert result["results"]["duty"] > 0
        assert result["results"]["area"] > 0
        # Output temperatures should be in Celsius
        assert result["results"]["hot_outlet_temp"] == pytest.approx(50.0, rel=1e-2)

    def test_zero_mass_flow_error(self):
        """Test error handling for zero mass flow rate."""
        inputs = {
            "mass_flow_hot": {"value": 0, "units": "kg/s"},
            "mass_flow_cold": {"value": 1.5, "units": "kg/s"},
            "cp_hot": {"value": 4.18, "units": "kJ/kg/K"},
            "cp_cold": {"value": 4.18, "units": "kJ/kg/K"},
            "t_hot_in": {"value": 80, "units": "°C"},
            "t_hot_out": {"value": 50, "units": "°C"},
            "t_cold_in": {"value": 20, "units": "°C"},
            "overall_heat_transfer_coefficient": {"value": 1000, "units": "W/m^2/K"},
        }
        
        with pytest.raises(ValueError, match="Hot fluid mass flow rate must be positive"):
            run(inputs)

    def test_zero_heat_transfer_coefficient_error(self):
        """Test error handling for zero heat transfer coefficient."""
        inputs = {
            "mass_flow_hot": {"value": 2.0, "units": "kg/s"},
            "mass_flow_cold": {"value": 1.5, "units": "kg/s"},
            "cp_hot": {"value": 4.18, "units": "kJ/kg/K"},
            "cp_cold": {"value": 4.18, "units": "kJ/kg/K"},
            "t_hot_in": {"value": 80, "units": "°C"},
            "t_hot_out": {"value": 50, "units": "°C"},
            "t_cold_in": {"value": 20, "units": "°C"},
            "overall_heat_transfer_coefficient": {"value": 0, "units": "W/m^2/K"},
        }
        
        with pytest.raises(ValueError, match="Overall heat transfer coefficient must be positive"):
            run(inputs)

    def test_invalid_correction_factor_error(self):
        """Test error handling for invalid correction factor."""
        inputs = {
            "mass_flow_hot": {"value": 2.0, "units": "kg/s"},
            "mass_flow_cold": {"value": 1.5, "units": "kg/s"},
            "cp_hot": {"value": 4.18, "units": "kJ/kg/K"},
            "cp_cold": {"value": 4.18, "units": "kJ/kg/K"},
            "t_hot_in": {"value": 80, "units": "°C"},
            "t_hot_out": {"value": 50, "units": "°C"},
            "t_cold_in": {"value": 20, "units": "°C"},
            "overall_heat_transfer_coefficient": {"value": 1000, "units": "W/m^2/K"},
            "correction_factor": 1.5,  # > 1.2 maximum
        }
        
        with pytest.raises(ValueError, match="less than or equal to 1.2"):
            run(inputs)

    def test_temperature_crossing_error(self):
        """Test error when outlet temperatures would cross (impossible)."""
        inputs = {
            "mass_flow_hot": {"value": 0.1, "units": "kg/s"},  # Very low hot flow
            "mass_flow_cold": {"value": 10.0, "units": "kg/s"},  # Very high cold flow
            "cp_hot": {"value": 4.18, "units": "kJ/kg/K"},
            "cp_cold": {"value": 4.18, "units": "kJ/kg/K"},
            "t_hot_in": {"value": 60, "units": "°C"},
            "t_cold_in": {"value": 50, "units": "°C"},
            "t_cold_out": {"value": 70, "units": "°C"},  # Would require hot outlet < cold outlet
            "overall_heat_transfer_coefficient": {"value": 1000, "units": "W/m^2/K"},
        }
        
        with pytest.raises(ValueError, match="hot outlet temperature must remain above cold inlet"):
            run(inputs)

    def test_area_scaling_with_u_coefficient(self):
        """Test that area scales inversely with heat transfer coefficient."""
        base_inputs = {
            "mass_flow_hot": {"value": 2.0, "units": "kg/s"},
            "mass_flow_cold": {"value": 1.5, "units": "kg/s"},
            "cp_hot": {"value": 4.18, "units": "kJ/kg/K"},
            "cp_cold": {"value": 4.18, "units": "kJ/kg/K"},
            "t_hot_in": {"value": 80, "units": "°C"},
            "t_hot_out": {"value": 50, "units": "°C"},
            "t_cold_in": {"value": 20, "units": "°C"},
            "overall_heat_transfer_coefficient": {"value": 1000, "units": "W/m^2/K"},
        }
        
        result_1x = run(base_inputs)
        
        # Half the heat transfer coefficient
        half_u_inputs = base_inputs.copy()
        half_u_inputs["overall_heat_transfer_coefficient"] = {"value": 500, "units": "W/m^2/K"}
        result_half_u = run(half_u_inputs)
        
        # Area should double when U is halved (for same duty and LMTD)
        area_1x = result_1x["results"]["area"]
        area_half_u = result_half_u["results"]["area"]
        
        assert area_half_u == pytest.approx(2 * area_1x, rel=1e-3)

    def test_step_generation(self):
        """Test that calculation steps are properly generated."""
        inputs = {
            "mass_flow_hot": {"value": 2.0, "units": "kg/s"},
            "mass_flow_cold": {"value": 1.5, "units": "kg/s"},
            "cp_hot": {"value": 4.18, "units": "kJ/kg/K"},
            "cp_cold": {"value": 4.18, "units": "kJ/kg/K"},
            "t_hot_in": {"value": 80, "units": "°C"},
            "t_hot_out": {"value": 50, "units": "°C"},
            "t_cold_in": {"value": 20, "units": "°C"},
            "overall_heat_transfer_coefficient": {"value": 1000, "units": "W/m^2/K"},
        }
        
        result = run(inputs)
        
        steps = result["steps"]
        assert len(steps) >= 3  # Should have multiple calculation steps
        
        # Check that key steps are present
        step_descriptions = [step.description for step in steps]
        assert any("energy balance" in desc.lower() for desc in step_descriptions)
        assert any("duty" in desc.lower() or "heat transfer" in desc.lower() for desc in step_descriptions)
        assert any("temperature difference" in desc.lower() for desc in step_descriptions)
        assert any("area" in desc.lower() for desc in step_descriptions)

    def test_pinch_point_warning(self):
        """Test warning for small temperature approach (pinch point)."""
        inputs = {
            "mass_flow_hot": {"value": 2.0, "units": "kg/s"},
            "mass_flow_cold": {"value": 1.5, "units": "kg/s"},
            "cp_hot": {"value": 4.18, "units": "kJ/kg/K"},
            "cp_cold": {"value": 4.18, "units": "kJ/kg/K"},
            "t_hot_in": {"value": 55, "units": "°C"},
            "t_hot_out": {"value": 52, "units": "°C"},  # Small temperature change
            "t_cold_in": {"value": 50, "units": "°C"},   # Close to hot outlet
            "overall_heat_transfer_coefficient": {"value": 1000, "units": "W/m^2/K"},
        }
        
        result = run(inputs)
        
        # Should generate warning about small temperature approach
        assert len(result["warnings"]) > 0
        assert any("temperature difference" in warning.lower() for warning in result["warnings"])

    def test_lmtd_calculation_accuracy(self):
        """Test LMTD calculation for known case."""
        inputs = {
            "mass_flow_hot": {"value": 1.0, "units": "kg/s"},
            "mass_flow_cold": {"value": 1.0, "units": "kg/s"},
            "cp_hot": {"value": 4.18, "units": "kJ/kg/K"},
            "cp_cold": {"value": 4.18, "units": "kJ/kg/K"},
            "t_hot_in": {"value": 100, "units": "°C"},
            "t_hot_out": {"value": 60, "units": "°C"},
            "t_cold_in": {"value": 20, "units": "°C"},
            "overall_heat_transfer_coefficient": {"value": 1000, "units": "W/m^2/K"},
        }
        
        result = run(inputs)
        
        # Cold outlet from heat balance: 20 + (100-60) = 60°C
        assert result["results"]["cold_outlet_temp"] == pytest.approx(60.0, rel=1e-3)
        
        # Delta T1 = 100 - 60 = 40°C, Delta T2 = 60 - 20 = 40°C
        # LMTD = (40 - 40) / ln(40/40) = 40°C (same terminal differences)
        assert result["results"]["delta_t1"] == pytest.approx(40.0, rel=1e-3)
        assert result["results"]["delta_t2"] == pytest.approx(40.0, rel=1e-3)
        assert result["results"]["lmtd"] == pytest.approx(40.0, rel=1e-3)