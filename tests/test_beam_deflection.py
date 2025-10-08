"""Tests for beam deflection calculations using Euler-Bernoulli beam theory."""

import pytest
from tools.beam_deflection import run


class TestBeamDeflection:
    """Test cases for beam deflection tool."""

    def test_point_load_center_si_units(self):
        """Test simply supported beam with central point load (SI units)."""
        inputs = {
            "load_case": "point_load_center",
            "length": {"value": 4.0, "units": "m"},
            "elastic_modulus": {"value": 200e9, "units": "Pa"},
            "moment_of_inertia": {"value": 8e-6, "units": "m^4"},
            "point_load": {"value": 10000, "units": "N"},
        }
        
        result = run(inputs)
        
        # Check results exist and are positive
        assert result["results"]["max_deflection"] > 0
        assert result["results"]["max_slope"] > 0
        assert result["results"]["reaction_a"] > 0
        assert result["results"]["reaction_b"] > 0
        assert result["results"]["moment_max"] > 0
        
        # Check equilibrium: reactions should sum to applied load
        total_reaction = result["results"]["reaction_a"] + result["results"]["reaction_b"]
        assert total_reaction == pytest.approx(10000, rel=1e-3)
        
        # For symmetric loading, reactions should be equal
        assert result["results"]["reaction_a"] == pytest.approx(result["results"]["reaction_b"], rel=1e-3)
        
        # Check units are correct
        assert result["units"]["max_deflection"] == "m"
        assert result["units"]["max_slope"] == "rad"
        assert result["units"]["reaction_a"] == "N"
        assert result["units"]["moment_max"] == "N·m"
        
        # Check profiles exist
        assert len(result["results"]["shear_profile"]) > 0
        assert len(result["results"]["moment_profile"]) > 0

    def test_uniform_load_si_units(self):
        """Test simply supported beam with uniform load (SI units)."""
        inputs = {
            "load_case": "uniform_load",
            "length": {"value": 6.0, "units": "m"},
            "elastic_modulus": {"value": 200e9, "units": "Pa"},
            "moment_of_inertia": {"value": 1e-5, "units": "m^4"},
            "distributed_load": {"value": 5000, "units": "N/m"},
        }
        
        result = run(inputs)
        
        # Check results exist and are positive
        assert result["results"]["max_deflection"] > 0
        assert result["results"]["max_slope"] > 0
        assert result["results"]["reaction_a"] > 0
        assert result["results"]["reaction_b"] > 0
        assert result["results"]["moment_max"] > 0
        
        # Total applied load = distributed_load * length
        total_load = 5000 * 6.0
        total_reaction = result["results"]["reaction_a"] + result["results"]["reaction_b"]
        assert total_reaction == pytest.approx(total_load, rel=1e-3)
        
        # For uniform loading, reactions should be equal
        assert result["results"]["reaction_a"] == pytest.approx(result["results"]["reaction_b"], rel=1e-3)

    def test_point_load_us_units(self):
        """Test beam deflection with US customary units."""
        inputs = {
            "load_case": "point_load_center",
            "length": {"value": 12, "units": "ft"},
            "elastic_modulus": {"value": 29e6, "units": "psi"},
            "moment_of_inertia": {"value": 100, "units": "in^4"},
            "point_load": {"value": 5000, "units": "lbf"},
        }
        
        result = run(inputs)
        
        assert result["results"]["max_deflection"] > 0
        assert result["results"]["reaction_a"] > 0
        assert result["results"]["reaction_b"] > 0
        
        # Results should be in SI units
        assert result["units"]["max_deflection"] == "m"
        assert result["units"]["reaction_a"] == "N"

    def test_steel_beam_realistic_case(self):
        """Test realistic steel beam scenario."""
        inputs = {
            "load_case": "uniform_load",
            "length": {"value": 8.0, "units": "m"},
            "elastic_modulus": {"value": 200e9, "units": "Pa"},  # Steel
            "moment_of_inertia": {"value": 2.14e-5, "units": "m^4"},  # W14x30 steel beam
            "distributed_load": {"value": 3000, "units": "N/m"},  # Typical floor load
        }
        
        result = run(inputs)
        
        # Deflection should be reasonable for a steel beam
        assert result["results"]["max_deflection"] > 0
        assert result["results"]["max_deflection"] < 0.1  # Less than 10cm seems reasonable
        assert len(result["steps"]) > 0

    def test_wood_beam_case(self):
        """Test wood beam with lower elastic modulus."""
        inputs = {
            "load_case": "point_load_center",
            "length": {"value": 3.0, "units": "m"},
            "elastic_modulus": {"value": 12e9, "units": "Pa"},  # Wood
            "moment_of_inertia": {"value": 1e-6, "units": "m^4"},
            "point_load": {"value": 2000, "units": "N"},
        }
        
        result = run(inputs)
        
        assert result["results"]["max_deflection"] > 0
        assert result["results"]["moment_max"] > 0

    def test_custom_sample_points(self):
        """Test with custom number of sample points for profiles."""
        inputs = {
            "load_case": "point_load_center",
            "length": {"value": 4.0, "units": "m"},
            "elastic_modulus": {"value": 200e9, "units": "Pa"},
            "moment_of_inertia": {"value": 8e-6, "units": "m^4"},
            "point_load": {"value": 10000, "units": "N"},
            "samples": 11,
        }
        
        result = run(inputs)
        
        # Should have exactly 11 sample points
        assert len(result["results"]["shear_profile"]) == 11
        assert len(result["results"]["moment_profile"]) == 11
        
        # First point should be at x=0, last at x=length
        assert result["results"]["shear_profile"][0][0] == pytest.approx(0.0)
        assert result["results"]["shear_profile"][-1][0] == pytest.approx(4.0)

    def test_maximum_sample_points(self):
        """Test with maximum allowed sample points."""
        inputs = {
            "load_case": "uniform_load",
            "length": {"value": 5.0, "units": "m"},
            "elastic_modulus": {"value": 200e9, "units": "Pa"},
            "moment_of_inertia": {"value": 1e-5, "units": "m^4"},
            "distributed_load": {"value": 1000, "units": "N/m"},
            "samples": 201,  # Maximum allowed
        }
        
        result = run(inputs)
        
        assert len(result["results"]["shear_profile"]) == 201
        assert len(result["results"]["moment_profile"]) == 201

    def test_zero_load_error(self):
        """Test error handling for zero load."""
        inputs = {
            "load_case": "point_load_center",
            "length": {"value": 4.0, "units": "m"},
            "elastic_modulus": {"value": 200e9, "units": "Pa"},
            "moment_of_inertia": {"value": 8e-6, "units": "m^4"},
            "point_load": {"value": 0, "units": "N"},
        }
        
        with pytest.raises(ValueError, match="Load must be positive"):
            run(inputs)

    def test_negative_length_error(self):
        """Test error handling for negative length."""
        inputs = {
            "load_case": "point_load_center",
            "length": {"value": -4.0, "units": "m"},
            "elastic_modulus": {"value": 200e9, "units": "Pa"},
            "moment_of_inertia": {"value": 8e-6, "units": "m^4"},
            "point_load": {"value": 1000, "units": "N"},
        }
        
        with pytest.raises(ValueError, match="Length must be positive"):
            run(inputs)

    def test_zero_elastic_modulus_error(self):
        """Test error handling for zero elastic modulus."""
        inputs = {
            "load_case": "point_load_center",
            "length": {"value": 4.0, "units": "m"},
            "elastic_modulus": {"value": 0, "units": "Pa"},
            "moment_of_inertia": {"value": 8e-6, "units": "m^4"},
            "point_load": {"value": 1000, "units": "N"},
        }
        
        with pytest.raises(ValueError, match="Elastic modulus must be positive"):
            run(inputs)

    def test_wrong_load_for_case_error(self):
        """Test error when wrong load type is provided for load case."""
        inputs = {
            "load_case": "point_load_center",
            "length": {"value": 4.0, "units": "m"},
            "elastic_modulus": {"value": 200e9, "units": "Pa"},
            "moment_of_inertia": {"value": 8e-6, "units": "m^4"},
            "distributed_load": {"value": 1000, "units": "N/m"},  # Wrong load type
        }
        
        with pytest.raises(ValueError, match="point_load is required"):
            run(inputs)

    def test_step_generation_point_load(self):
        """Test that calculation steps are properly generated for point load."""
        inputs = {
            "load_case": "point_load_center",
            "length": {"value": 4.0, "units": "m"},
            "elastic_modulus": {"value": 200e9, "units": "Pa"},
            "moment_of_inertia": {"value": 8e-6, "units": "m^4"},
            "point_load": {"value": 10000, "units": "N"},
        }
        
        result = run(inputs)
        
        steps = result["steps"]
        assert len(steps) >= 3  # Should have multiple calculation steps
        
        # Check that key steps are present
        step_descriptions = [step.description for step in steps]
        assert any("reaction" in desc.lower() for desc in step_descriptions)
        assert any("deflection" in desc.lower() for desc in step_descriptions)
        assert any("moment" in desc.lower() for desc in step_descriptions)

    def test_step_generation_uniform_load(self):
        """Test that calculation steps are properly generated for uniform load."""
        inputs = {
            "load_case": "uniform_load",
            "length": {"value": 6.0, "units": "m"},
            "elastic_modulus": {"value": 200e9, "units": "Pa"},
            "moment_of_inertia": {"value": 1e-5, "units": "m^4"},
            "distributed_load": {"value": 5000, "units": "N/m"},
        }
        
        result = run(inputs)
        
        steps = result["steps"]
        assert len(steps) >= 3
        
        # Check specific steps for uniform load
        step_descriptions = [step.description for step in steps]
        assert any("reaction" in desc.lower() for desc in step_descriptions)
        assert any("deflection" in desc.lower() for desc in step_descriptions)

    def test_shear_moment_profiles_point_load(self):
        """Test shear and moment profile characteristics for point load."""
        inputs = {
            "load_case": "point_load_center",
            "length": {"value": 4.0, "units": "m"},
            "elastic_modulus": {"value": 200e9, "units": "Pa"},
            "moment_of_inertia": {"value": 8e-6, "units": "m^4"},
            "point_load": {"value": 10000, "units": "N"},
            "samples": 21,
        }
        
        result = run(inputs)
        
        shear_profile = result["results"]["shear_profile"]
        moment_profile = result["results"]["moment_profile"]
        
        # For point load at center, shear should be constant in each half
        # and moment should be maximum at center
        mid_index = len(moment_profile) // 2
        max_moment_calc = max(point[1] for point in moment_profile)
        center_moment = moment_profile[mid_index][1]
        
        # Maximum moment should occur near center
        assert abs(center_moment - max_moment_calc) < abs(0.1 * max_moment_calc)

    def test_deflection_scaling(self):
        """Test that deflection scales properly with load."""
        base_inputs = {
            "load_case": "point_load_center",
            "length": {"value": 4.0, "units": "m"},
            "elastic_modulus": {"value": 200e9, "units": "Pa"},
            "moment_of_inertia": {"value": 8e-6, "units": "m^4"},
            "point_load": {"value": 10000, "units": "N"},
        }
        
        result_1x = run(base_inputs)
        
        # Double the load
        double_load_inputs = base_inputs.copy()
        double_load_inputs["point_load"] = {"value": 20000, "units": "N"}
        result_2x = run(double_load_inputs)
        
        # Deflection should double with double load (linear elastic)
        deflection_1x = result_1x["results"]["max_deflection"]
        deflection_2x = result_2x["results"]["max_deflection"]
        
        assert deflection_2x == pytest.approx(2 * deflection_1x, rel=1e-3)