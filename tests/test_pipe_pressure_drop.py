"""Tests for pipe pressure drop calculations using Darcy-Weisbach equation."""

import pytest
from tools.pipe_pressure_drop import run


class TestPipePressureDrop:
    """Test cases for pipe pressure drop tool."""

    def test_volumetric_flow_input(self):
        """Test calculation with volumetric flow rate input."""
        inputs = {
            "volumetric_flow_rate": {"value": 0.012, "units": "m^3/s"},
            "diameter": {"value": 0.1, "units": "m"},
            "length": {"value": 50, "units": "m"},
            "roughness": {"value": 0.000045, "units": "m"},
            "density": {"value": 998, "units": "kg/m^3"},
            "dynamic_viscosity": {"value": 1e-3, "units": "Pa*s"},
        }
        
        result = run(inputs)
        
        assert result["results"]["delta_p"] > 0
        assert result["results"]["friction_factor"] > 0
        assert result["results"]["reynolds_number"] > 0
        assert result["results"]["velocity"] > 0
        assert result["results"]["volumetric_flow_rate"] == pytest.approx(0.012, rel=1e-3)
        assert result["results"]["head_loss"] > 0
        assert len(result["steps"]) > 0
        assert result["units"]["delta_p"] == "Pa"
        assert result["units"]["velocity"] == "m/s"

    def test_velocity_input(self):
        """Test calculation with velocity input instead of volumetric flow."""
        inputs = {
            "velocity": {"value": 1.5, "units": "m/s"},
            "diameter": {"value": 0.1, "units": "m"},
            "length": {"value": 50, "units": "m"},
            "roughness": {"value": 0.000045, "units": "m"},
            "density": {"value": 998, "units": "kg/m^3"},
            "dynamic_viscosity": {"value": 1e-3, "units": "Pa*s"},
        }
        
        result = run(inputs)
        
        assert result["results"]["velocity"] == pytest.approx(1.5, rel=1e-3)
        assert result["results"]["volumetric_flow_rate"] > 0
        assert result["results"]["delta_p"] > 0
        assert result["results"]["reynolds_number"] > 0

    def test_unit_conversions(self):
        """Test with US customary units."""
        inputs = {
            "volumetric_flow_rate": {"value": 100, "units": "gpm"},
            "diameter": {"value": 4, "units": "in"},
            "length": {"value": 100, "units": "ft"},
            "roughness": {"value": 1.8, "units": "mils"},
            "density": {"value": 62.4, "units": "lb/ft^3"},
            "dynamic_viscosity": {"value": 1, "units": "cP"},
        }
        
        result = run(inputs)
        
        assert result["results"]["delta_p"] > 0
        assert result["results"]["friction_factor"] > 0
        assert result["results"]["reynolds_number"] > 0
        assert len(result["steps"]) > 0

    def test_high_reynolds_turbulent_flow(self):
        """Test turbulent flow conditions."""
        inputs = {
            "volumetric_flow_rate": {"value": 0.1, "units": "m^3/s"},
            "diameter": {"value": 0.2, "units": "m"},
            "length": {"value": 100, "units": "m"},
            "roughness": {"value": 0.00005, "units": "m"},
            "density": {"value": 1000, "units": "kg/m^3"},
            "dynamic_viscosity": {"value": 1e-3, "units": "Pa*s"},
        }
        
        result = run(inputs)
        
        # High Re should give turbulent flow
        assert result["results"]["reynolds_number"] > 4000
        assert result["results"]["friction_factor"] > 0
        assert result["results"]["delta_p"] > 0

    def test_laminar_flow(self):
        """Test laminar flow conditions."""
        inputs = {
            "volumetric_flow_rate": {"value": 0.0001, "units": "m^3/s"},  # Very low flow
            "diameter": {"value": 0.025, "units": "m"},  # Small diameter  
            "length": {"value": 10, "units": "m"},
            "roughness": {"value": 0.000001, "units": "m"},
            "density": {"value": 1000, "units": "kg/m^3"},
            "dynamic_viscosity": {"value": 0.1, "units": "Pa*s"},  # Very high viscosity
        }
        
        result = run(inputs)
        
        # Low Re should give laminar flow
        assert result["results"]["reynolds_number"] < 2300
        assert result["results"]["friction_factor"] > 0

    def test_smooth_pipe(self):
        """Test very smooth pipe (low roughness)."""
        inputs = {
            "volumetric_flow_rate": {"value": 0.05, "units": "m^3/s"},
            "diameter": {"value": 0.15, "units": "m"},
            "length": {"value": 75, "units": "m"},
            "roughness": {"value": 1e-6, "units": "m"},  # Very smooth
            "density": {"value": 998, "units": "kg/m^3"},
            "dynamic_viscosity": {"value": 1e-3, "units": "Pa*s"},
        }
        
        result = run(inputs)
        
        assert result["results"]["delta_p"] > 0
        assert result["results"]["friction_factor"] > 0

    def test_rough_pipe(self):
        """Test rough pipe (high roughness)."""
        inputs = {
            "volumetric_flow_rate": {"value": 0.05, "units": "m^3/s"},
            "diameter": {"value": 0.15, "units": "m"},
            "length": {"value": 75, "units": "m"},
            "roughness": {"value": 0.001, "units": "m"},  # Very rough
            "density": {"value": 998, "units": "kg/m^3"},
            "dynamic_viscosity": {"value": 1e-3, "units": "Pa*s"},
        }
        
        result = run(inputs)
        
        assert result["results"]["delta_p"] > 0
        assert result["results"]["friction_factor"] > 0

    def test_zero_flow_rate(self):
        """Test error handling for zero flow rate."""
        inputs = {
            "volumetric_flow_rate": {"value": 0, "units": "m^3/s"},
            "diameter": {"value": 0.1, "units": "m"},
            "length": {"value": 50, "units": "m"},
            "roughness": {"value": 0.000045, "units": "m"},
            "density": {"value": 998, "units": "kg/m^3"},
            "dynamic_viscosity": {"value": 1e-3, "units": "Pa*s"},
        }
        
        with pytest.raises(ValueError, match="Flow rate must be positive"):
            run(inputs)

    def test_negative_diameter(self):
        """Test error handling for negative diameter."""
        inputs = {
            "volumetric_flow_rate": {"value": 0.01, "units": "m^3/s"},
            "diameter": {"value": -0.1, "units": "m"},
            "length": {"value": 50, "units": "m"},
            "roughness": {"value": 0.000045, "units": "m"},
            "density": {"value": 998, "units": "kg/m^3"},
            "dynamic_viscosity": {"value": 1e-3, "units": "Pa*s"},
        }
        
        with pytest.raises(ValueError, match="Diameter must be positive"):
            run(inputs)

    def test_step_generation(self):
        """Test that calculation steps are properly generated."""
        inputs = {
            "volumetric_flow_rate": {"value": 0.012, "units": "m^3/s"},
            "diameter": {"value": 0.1, "units": "m"},
            "length": {"value": 50, "units": "m"},
            "roughness": {"value": 0.000045, "units": "m"},
            "density": {"value": 998, "units": "kg/m^3"},
            "dynamic_viscosity": {"value": 1e-3, "units": "Pa*s"},
        }
        
        result = run(inputs)
        
        steps = result["steps"]
        assert len(steps) >= 4  # Should have multiple calculation steps
        
        # Check that key steps are present
        step_descriptions = [step.description for step in steps]
        assert any("velocity" in desc.lower() for desc in step_descriptions)
        assert any("reynolds" in desc.lower() for desc in step_descriptions)
        assert any("friction" in desc.lower() for desc in step_descriptions)
        assert any("pressure" in desc.lower() for desc in step_descriptions)

    def test_consistent_units_in_results(self):
        """Test that all results use consistent SI units."""
        inputs = {
            "volumetric_flow_rate": {"value": 50, "units": "gpm"},  # US units input
            "diameter": {"value": 6, "units": "in"},
            "length": {"value": 200, "units": "ft"},
            "roughness": {"value": 2, "units": "mils"},
            "density": {"value": 62.4, "units": "lb/ft^3"},
            "dynamic_viscosity": {"value": 1, "units": "cP"},
        }
        
        result = run(inputs)
        
        # Results should be in SI units regardless of input units
        assert result["units"]["delta_p"] == "Pa"
        assert result["units"]["velocity"] == "m/s"
        assert result["units"]["volumetric_flow_rate"] == "m^3/s"
        assert result["units"]["head_loss"] == "m"