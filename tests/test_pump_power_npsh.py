"""Tests for pump power and NPSH calculations."""

import pytest
from tools.pump_power_npsh import run


class TestPumpPowerNpsh:
    """Test cases for pump power and NPSH tool."""

    def test_basic_pump_calculation_si_units(self):
        """Test basic pump power calculation with SI units."""
        inputs = {
            "volumetric_flow_rate": {"value": 0.05, "units": "m^3/s"},
            "head": {"value": 50, "units": "m"},
            "density": {"value": 1000, "units": "kg/m^3"},
            "efficiency": 0.75,  # 75% efficiency as fraction
            "suction_pressure": {"value": 101325, "units": "Pa"},
            "vapor_pressure": {"value": 2500, "units": "Pa"},
            "suction_elevation": {"value": 2.0, "units": "m"},
            "suction_losses": {"value": 1.5, "units": "m"},
        }
        
        result = run(inputs)
        
        # Check basic results
        assert result["results"]["hydraulic_power"] > 0
        assert result["results"]["input_power"] > 0
        assert result["results"]["npsha"] > 0
        
        # Input power should be higher than hydraulic power due to efficiency
        assert result["results"]["input_power"] > result["results"]["hydraulic_power"]
        
        # Check units
        assert result["units"]["hydraulic_power"] == "W"
        assert result["units"]["input_power"] == "W"
        assert result["units"]["npsha"] == "m"
        
        # Check steps are generated
        assert len(result["steps"]) > 0

    def test_efficiency_as_percentage(self):
        """Test pump calculation with efficiency specified as percentage."""
        inputs = {
            "volumetric_flow_rate": {"value": 0.05, "units": "m^3/s"},
            "head": {"value": 50, "units": "m"},
            "density": {"value": 1000, "units": "kg/m^3"},
            "efficiency": {"value": 75, "units": "%"},  # 75% as percentage
            "suction_pressure": {"value": 101325, "units": "Pa"},
            "vapor_pressure": {"value": 2500, "units": "Pa"},
            "suction_elevation": {"value": 2.0, "units": "m"},
            "suction_losses": {"value": 1.5, "units": "m"},
        }
        
        result = run(inputs)
        
        assert result["results"]["hydraulic_power"] > 0
        assert result["results"]["input_power"] > 0
        assert result["results"]["input_power"] > result["results"]["hydraulic_power"]

    def test_efficiency_as_fraction_with_units(self):
        """Test pump calculation with efficiency as fraction with units."""
        inputs = {
            "volumetric_flow_rate": {"value": 0.05, "units": "m^3/s"},
            "head": {"value": 50, "units": "m"},
            "density": {"value": 1000, "units": "kg/m^3"},
            "efficiency": {"value": 0.8, "units": "fraction"},
            "suction_pressure": {"value": 101325, "units": "Pa"},
            "vapor_pressure": {"value": 2500, "units": "Pa"},
            "suction_elevation": {"value": 2.0, "units": "m"},
            "suction_losses": {"value": 1.5, "units": "m"},
        }
        
        result = run(inputs)
        
        assert result["results"]["hydraulic_power"] > 0
        assert result["results"]["input_power"] > 0

    def test_with_npsh_required(self):
        """Test calculation including NPSHr and margin calculation."""
        inputs = {
            "volumetric_flow_rate": {"value": 0.03, "units": "m^3/s"},
            "head": {"value": 40, "units": "m"},
            "density": {"value": 998, "units": "kg/m^3"},
            "efficiency": 0.72,
            "suction_pressure": {"value": 101325, "units": "Pa"},
            "vapor_pressure": {"value": 3200, "units": "Pa"},
            "suction_elevation": {"value": 1.0, "units": "m"},
            "suction_losses": {"value": 0.8, "units": "m"},
            "npsh_required": {"value": 3.5, "units": "m"},
        }
        
        result = run(inputs)
        
        assert result["results"]["npsha"] > 0
        assert result["results"]["npshr"] == 3.5
        assert result["results"]["npsh_margin"] is not None
        assert result["results"]["adequate_npsh"] is not None
        
        # NPSH margin = NPSHa - NPSHr
        expected_margin = result["results"]["npsha"] - 3.5
        assert result["results"]["npsh_margin"] == pytest.approx(expected_margin, rel=1e-3)

    def test_adequate_npsh_true(self):
        """Test case where NPSH is adequate."""
        inputs = {
            "volumetric_flow_rate": {"value": 0.02, "units": "m^3/s"},
            "head": {"value": 30, "units": "m"},
            "density": {"value": 1000, "units": "kg/m^3"},
            "efficiency": 0.8,
            "suction_pressure": {"value": 200000, "units": "Pa"},  # High suction pressure
            "vapor_pressure": {"value": 2500, "units": "Pa"},
            "suction_elevation": {"value": -2.0, "units": "m"},  # Below pump (flooded suction)
            "suction_losses": {"value": 0.5, "units": "m"},
            "npsh_required": {"value": 3.0, "units": "m"},
        }
        
        result = run(inputs)
        
        assert result["results"]["adequate_npsh"] is True
        assert result["results"]["npsh_margin"] > 0

    def test_inadequate_npsh(self):
        """Test case where NPSH is inadequate."""
        inputs = {
            "volumetric_flow_rate": {"value": 0.05, "units": "m^3/s"},
            "head": {"value": 60, "units": "m"},
            "density": {"value": 1000, "units": "kg/m^3"},
            "efficiency": 0.75,
            "suction_pressure": {"value": 101325, "units": "Pa"},  # Atmospheric
            "vapor_pressure": {"value": 50000, "units": "Pa"},  # High vapor pressure (hot liquid)
            "suction_elevation": {"value": 5.0, "units": "m"},  # High suction lift
            "suction_losses": {"value": 3.0, "units": "m"},  # High losses
            "npsh_required": {"value": 8.0, "units": "m"},  # High requirement
        }
        
        result = run(inputs)
        
        assert result["results"]["adequate_npsh"] is False
        assert result["results"]["npsh_margin"] < 0
        assert len(result["warnings"]) > 0  # Should generate cavitation warning

    def test_us_customary_units(self):
        """Test calculation with US customary units."""
        inputs = {
            "volumetric_flow_rate": {"value": 500, "units": "gpm"},
            "head": {"value": 150, "units": "ft"},
            "density": {"value": 62.4, "units": "lb/ft^3"},
            "efficiency": {"value": 78, "units": "%"},
            "suction_pressure": {"value": 14.7, "units": "psi"},
            "vapor_pressure": {"value": 0.5, "units": "psi"},
            "suction_elevation": {"value": 8, "units": "ft"},
            "suction_losses": {"value": 5, "units": "ft"},
        }
        
        result = run(inputs)
        
        assert result["results"]["hydraulic_power"] > 0
        assert result["results"]["input_power"] > 0
        assert result["results"]["npsha"] > 0
        
        # Results should be in SI units
        assert result["units"]["hydraulic_power"] == "W"
        assert result["units"]["npsha"] == "m"

    def test_zero_flow_rate_error(self):
        """Test error handling for zero flow rate."""
        inputs = {
            "volumetric_flow_rate": {"value": 0, "units": "m^3/s"},
            "head": {"value": 50, "units": "m"},
            "density": {"value": 1000, "units": "kg/m^3"},
            "efficiency": 0.75,
            "suction_pressure": {"value": 101325, "units": "Pa"},
            "vapor_pressure": {"value": 2500, "units": "Pa"},
            "suction_elevation": {"value": 2.0, "units": "m"},
            "suction_losses": {"value": 1.5, "units": "m"},
        }
        
        with pytest.raises(ValueError, match="Flow rate must be positive"):
            run(inputs)

    def test_negative_head_error(self):
        """Test error handling for negative head."""
        inputs = {
            "volumetric_flow_rate": {"value": 0.05, "units": "m^3/s"},
            "head": {"value": -10, "units": "m"},
            "density": {"value": 1000, "units": "kg/m^3"},
            "efficiency": 0.75,
            "suction_pressure": {"value": 101325, "units": "Pa"},
            "vapor_pressure": {"value": 2500, "units": "Pa"},
            "suction_elevation": {"value": 2.0, "units": "m"},
            "suction_losses": {"value": 1.5, "units": "m"},
        }
        
        with pytest.raises(ValueError, match="Head must be positive"):
            run(inputs)

    def test_invalid_efficiency_error(self):
        """Test error handling for efficiency outside valid range."""
        inputs = {
            "volumetric_flow_rate": {"value": 0.05, "units": "m^3/s"},
            "head": {"value": 50, "units": "m"},
            "density": {"value": 1000, "units": "kg/m^3"},
            "efficiency": 1.5,  # 150% - impossible
            "suction_pressure": {"value": 101325, "units": "Pa"},
            "vapor_pressure": {"value": 2500, "units": "Pa"},
            "suction_elevation": {"value": 2.0, "units": "m"},
            "suction_losses": {"value": 1.5, "units": "m"},
        }
        
        with pytest.raises(ValueError, match="efficiency must be between 0 and 1"):
            run(inputs)

    def test_low_efficiency_warning(self):
        """Test warning for very low efficiency."""
        inputs = {
            "volumetric_flow_rate": {"value": 0.05, "units": "m^3/s"},
            "head": {"value": 50, "units": "m"},
            "density": {"value": 1000, "units": "kg/m^3"},
            "efficiency": 0.3,  # Very low efficiency
            "suction_pressure": {"value": 101325, "units": "Pa"},
            "vapor_pressure": {"value": 2500, "units": "Pa"},
            "suction_elevation": {"value": 2.0, "units": "m"},
            "suction_losses": {"value": 1.5, "units": "m"},
        }
        
        result = run(inputs)
        
        assert len(result["warnings"]) > 0
        assert any("very low" in warning.lower() and "efficiency" in warning.lower() for warning in result["warnings"])

    def test_high_suction_lift_warning(self):
        """Test warning for high suction lift."""
        inputs = {
            "volumetric_flow_rate": {"value": 0.05, "units": "m^3/s"},
            "head": {"value": 50, "units": "m"},
            "density": {"value": 1000, "units": "kg/m^3"},
            "efficiency": 0.75,
            "suction_pressure": {"value": 101325, "units": "Pa"},
            "vapor_pressure": {"value": 2500, "units": "Pa"},
            "suction_elevation": {"value": 8.0, "units": "m"},  # High suction lift
            "suction_losses": {"value": 1.5, "units": "m"},
        }
        
        result = run(inputs)
        
        assert len(result["warnings"]) > 0
        assert any("suction lift" in warning.lower() for warning in result["warnings"])

    def test_power_scaling_with_flow(self):
        """Test that power scales correctly with flow rate."""
        base_inputs = {
            "volumetric_flow_rate": {"value": 0.02, "units": "m^3/s"},
            "head": {"value": 50, "units": "m"},
            "density": {"value": 1000, "units": "kg/m^3"},
            "efficiency": 0.75,
            "suction_pressure": {"value": 101325, "units": "Pa"},
            "vapor_pressure": {"value": 2500, "units": "Pa"},
            "suction_elevation": {"value": 2.0, "units": "m"},
            "suction_losses": {"value": 1.5, "units": "m"},
        }
        
        result_1x = run(base_inputs)
        
        # Double the flow rate
        double_flow_inputs = base_inputs.copy()
        double_flow_inputs["volumetric_flow_rate"] = {"value": 0.04, "units": "m^3/s"}
        result_2x = run(double_flow_inputs)
        
        # Power should double with double flow (at same head)
        power_1x = result_1x["results"]["hydraulic_power"]
        power_2x = result_2x["results"]["hydraulic_power"]
        
        assert power_2x == pytest.approx(2 * power_1x, rel=1e-3)

    def test_step_generation(self):
        """Test that calculation steps are properly generated."""
        inputs = {
            "volumetric_flow_rate": {"value": 0.05, "units": "m^3/s"},
            "head": {"value": 50, "units": "m"},
            "density": {"value": 1000, "units": "kg/m^3"},
            "efficiency": 0.75,
            "suction_pressure": {"value": 101325, "units": "Pa"},
            "vapor_pressure": {"value": 2500, "units": "Pa"},
            "suction_elevation": {"value": 2.0, "units": "m"},
            "suction_losses": {"value": 1.5, "units": "m"},
            "npsh_required": {"value": 4.0, "units": "m"},
        }
        
        result = run(inputs)
        
        steps = result["steps"]
        assert len(steps) >= 3  # Should have multiple calculation steps
        
        # Check that key steps are present
        step_descriptions = [step.description for step in steps]
        assert any("hydraulic power" in desc.lower() for desc in step_descriptions)
        assert any("input" in desc.lower() and "power" in desc.lower() for desc in step_descriptions)
        assert any("npsh" in desc.lower() for desc in step_descriptions)

    def test_pressure_head_conversions(self):
        """Test pressure to head conversions are correct."""
        inputs = {
            "volumetric_flow_rate": {"value": 0.05, "units": "m^3/s"},
            "head": {"value": 50, "units": "m"},
            "density": {"value": 1000, "units": "kg/m^3"},
            "efficiency": 0.75,
            "suction_pressure": {"value": 101325, "units": "Pa"},  # 1 atm
            "vapor_pressure": {"value": 2500, "units": "Pa"},
            "suction_elevation": {"value": 0, "units": "m"},
            "suction_losses": {"value": 0, "units": "m"},
        }
        
        result = run(inputs)
        
        # 1 atm ≈ 10.33 m of water head
        expected_suction_head = 101325 / (1000 * 9.81)  # P / (ρ * g)
        assert result["results"]["suction_pressure_head"] == pytest.approx(expected_suction_head, rel=1e-3)
        
        expected_vapor_head = 2500 / (1000 * 9.81)
        assert result["results"]["vapor_pressure_head"] == pytest.approx(expected_vapor_head, rel=1e-3)