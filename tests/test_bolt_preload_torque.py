"""Tests for bolt preload and torque calculations."""

import pytest
from tools.bolt_preload_torque import run


class TestBoltPreloadTorque:
    """Test cases for bolt preload and torque tool."""

    def test_basic_bolt_calculation_si_units(self):
        """Test basic bolt preload calculation with SI units."""
        inputs = {
            "preload": {"value": 50000, "units": "N"},
            "nominal_diameter": {"value": 0.016, "units": "m"},  # M16 bolt
            "thread_pitch": {"value": 0.002, "units": "m"},     # 2mm pitch
            "nut_factor": 0.2,  # Typical K factor
            "friction_thread": 0.15,
            "friction_bearing": 0.12,
        }
        
        result = run(inputs)
        
        # Check basic results
        assert result["results"]["recommended_torque"] > 0
        assert result["results"]["thread_torque"] > 0
        assert result["results"]["bearing_torque"] > 0
        assert result["results"]["torque_from_breakdown"] > 0
        assert result["results"]["preload_stress_area"] > 0
        assert result["results"]["preload_stress"] > 0
        assert result["results"]["effective_nut_factor"] > 0
        
        # Check units
        assert result["units"]["recommended_torque"] == "N·m"
        assert result["units"]["thread_torque"] == "N·m"
        assert result["units"]["bearing_torque"] == "N·m"
        assert result["units"]["preload_stress_area"] == "m^2"
        assert result["units"]["preload_stress"] == "Pa"
        
        # Torque breakdown should approximately equal K-factor torque
        breakdown_torque = result["results"]["torque_from_breakdown"]
        k_factor_torque = result["results"]["recommended_torque"]
        assert abs(breakdown_torque - k_factor_torque) / k_factor_torque < 0.1  # Within 10%
        
        # Check steps are generated
        assert len(result["steps"]) > 0

    def test_with_external_load_and_proof_load(self):
        """Test calculation including external load and proof load safety."""
        inputs = {
            "preload": {"value": 30000, "units": "N"},
            "nominal_diameter": {"value": 0.012, "units": "m"},  # M12 bolt
            "thread_pitch": {"value": 0.00175, "units": "m"},   # 1.75mm pitch
            "nut_factor": 0.18,
            "friction_thread": 0.14,
            "friction_bearing": 0.10,
            "external_load": {"value": 15000, "units": "N"},
            "proof_load": {"value": 70000, "units": "N"},
        }
        
        result = run(inputs)
        
        assert result["results"]["clamp_safety_factor"] is not None
        assert result["results"]["clamp_residual"] is not None
        assert result["results"]["proof_safety_factor"] is not None
        assert result["results"]["effective_nut_factor"] is not None
        
        # Clamp residual = preload - external_load
        expected_residual = 30000 - 15000
        assert result["results"]["clamp_residual"] == pytest.approx(expected_residual, rel=1e-3)
        
        # Clamp safety factor = preload / external_load
        expected_clamp_sf = 30000 / 15000
        assert result["results"]["clamp_safety_factor"] == pytest.approx(expected_clamp_sf, rel=1e-3)
        
        # Proof safety factor = proof_load / preload
        expected_proof_sf = 70000 / 30000
        assert result["results"]["proof_safety_factor"] == pytest.approx(expected_proof_sf, rel=1e-3)

    def test_us_customary_units(self):
        """Test calculation with US customary units."""
        inputs = {
            "preload": {"value": 10000, "units": "lbf"},
            "nominal_diameter": {"value": 0.5, "units": "in"},
            "thread_pitch": {"value": 0.05, "units": "in"},  # 20 TPI = 0.05" pitch
            "nut_factor": 0.2,
            "friction_thread": 0.15,
            "friction_bearing": 0.12,
        }
        
        result = run(inputs)
        
        assert result["results"]["recommended_torque"] > 0
        assert result["results"]["preload_stress"] > 0
        
        # Results should be in SI units
        assert result["units"]["recommended_torque"] == "N·m"
        assert result["units"]["preload_stress"] == "Pa"

    def test_metric_coarse_thread_m10(self):
        """Test M10 x 1.5 metric coarse thread."""
        inputs = {
            "preload": {"value": 20000, "units": "N"},
            "nominal_diameter": {"value": 0.010, "units": "m"},
            "thread_pitch": {"value": 0.0015, "units": "m"},
            "nut_factor": 0.2,
            "friction_thread": 0.15,
            "friction_bearing": 0.12,
        }
        
        result = run(inputs)
        
        assert result["results"]["recommended_torque"] > 0
        # Stress area for M10x1.5 ≈ 58 mm²
        expected_stress_area = 58e-6  # m²
        assert result["results"]["preload_stress_area"] == pytest.approx(expected_stress_area, rel=0.1)

    def test_metric_fine_thread_m12(self):
        """Test M12 x 1.25 metric fine thread."""
        inputs = {
            "preload": {"value": 35000, "units": "N"},
            "nominal_diameter": {"value": 0.012, "units": "m"},
            "thread_pitch": {"value": 0.00125, "units": "m"},  # Fine pitch
            "nut_factor": 0.18,
            "friction_thread": 0.14,
            "friction_bearing": 0.11,
        }
        
        result = run(inputs)
        
        assert result["results"]["recommended_torque"] > 0
        # Fine threads have larger stress area than coarse
        assert result["results"]["preload_stress_area"] > 84e-6  # > M12 coarse area

    def test_high_strength_bolt(self):
        """Test high-strength bolt with high preload."""
        inputs = {
            "preload": {"value": 150000, "units": "N"},
            "nominal_diameter": {"value": 0.020, "units": "m"},  # M20
            "thread_pitch": {"value": 0.0025, "units": "m"},
            "nut_factor": 0.15,  # Well-lubricated
            "friction_thread": 0.10,
            "friction_bearing": 0.08,
            "proof_load": {"value": 250000, "units": "N"},
        }
        
        result = run(inputs)
        
        assert result["results"]["recommended_torque"] > 0
        assert result["results"]["proof_safety_factor"] > 1.0
        # High preload should generate high stress
        assert result["results"]["preload_stress"] > 400e6  # > 400 MPa

    def test_zero_preload_error(self):
        """Test error handling for zero preload."""
        inputs = {
            "preload": {"value": 0, "units": "N"},
            "nominal_diameter": {"value": 0.016, "units": "m"},
            "thread_pitch": {"value": 0.002, "units": "m"},
            "nut_factor": 0.2,
            "friction_thread": 0.15,
            "friction_bearing": 0.12,
        }
        
        with pytest.raises(ValueError, match="Preload must be positive"):
            run(inputs)

    def test_negative_diameter_error(self):
        """Test error handling for negative diameter."""
        inputs = {
            "preload": {"value": 50000, "units": "N"},
            "nominal_diameter": {"value": -0.016, "units": "m"},
            "thread_pitch": {"value": 0.002, "units": "m"},
            "nut_factor": 0.2,
            "friction_thread": 0.15,
            "friction_bearing": 0.12,
        }
        
        with pytest.raises(ValueError, match="Nominal diameter must be positive"):
            run(inputs)

    def test_zero_thread_pitch_error(self):
        """Test error handling for zero thread pitch."""
        inputs = {
            "preload": {"value": 50000, "units": "N"},
            "nominal_diameter": {"value": 0.016, "units": "m"},
            "thread_pitch": {"value": 0, "units": "m"},
            "nut_factor": 0.2,
            "friction_thread": 0.15,
            "friction_bearing": 0.12,
        }
        
        with pytest.raises(ValueError, match="Thread pitch must be positive"):
            run(inputs)

    def test_negative_friction_error(self):
        """Test error handling for negative friction coefficients."""
        inputs = {
            "preload": {"value": 50000, "units": "N"},
            "nominal_diameter": {"value": 0.016, "units": "m"},
            "thread_pitch": {"value": 0.002, "units": "m"},
            "nut_factor": 0.2,
            "friction_thread": -0.15,  # Negative friction
            "friction_bearing": 0.12,
        }
        
        with pytest.raises(ValueError, match="greater than 0"):
            run(inputs)

    def test_insufficient_preload_warning(self):
        """Test warning for preload less than external load."""
        inputs = {
            "preload": {"value": 10000, "units": "N"},
            "nominal_diameter": {"value": 0.016, "units": "m"},
            "thread_pitch": {"value": 0.002, "units": "m"},
            "nut_factor": 0.2,
            "friction_thread": 0.15,
            "friction_bearing": 0.12,
            "external_load": {"value": 15000, "units": "N"},  # Higher than preload
        }
        
        result = run(inputs)
        
        assert len(result["warnings"]) > 0
        assert any("joint may separate" in warning.lower() for warning in result["warnings"])
        assert result["results"]["clamp_safety_factor"] < 1.0

    def test_low_proof_safety_warning(self):
        """Test warning for low proof load safety factor."""
        inputs = {
            "preload": {"value": 45000, "units": "N"},
            "nominal_diameter": {"value": 0.016, "units": "m"},
            "thread_pitch": {"value": 0.002, "units": "m"},
            "nut_factor": 0.2,
            "friction_thread": 0.15,
            "friction_bearing": 0.12,
            "proof_load": {"value": 50000, "units": "N"},  # Low safety margin
        }
        
        result = run(inputs)
        
        assert len(result["warnings"]) > 0
        assert any("proof load" in warning.lower() for warning in result["warnings"])
        assert result["results"]["proof_safety_factor"] < 1.5

    def test_nut_factor_scaling(self):
        """Test that torque scales linearly with nut factor."""
        base_inputs = {
            "preload": {"value": 50000, "units": "N"},
            "nominal_diameter": {"value": 0.016, "units": "m"},
            "thread_pitch": {"value": 0.002, "units": "m"},
            "nut_factor": 0.2,
            "friction_thread": 0.15,
            "friction_bearing": 0.12,
        }
        
        result_1x = run(base_inputs)
        
        # Double the nut factor
        double_k_inputs = base_inputs.copy()
        double_k_inputs["nut_factor"] = 0.4
        result_2x = run(double_k_inputs)
        
        # Torque should double with double K factor
        torque_1x = result_1x["results"]["recommended_torque"]
        torque_2x = result_2x["results"]["recommended_torque"]
        
        assert torque_2x == pytest.approx(2 * torque_1x, rel=1e-3)

    def test_friction_contribution_breakdown(self):
        """Test that thread and bearing torques sum correctly."""
        inputs = {
            "preload": {"value": 40000, "units": "N"},
            "nominal_diameter": {"value": 0.016, "units": "m"},
            "thread_pitch": {"value": 0.002, "units": "m"},
            "nut_factor": 0.2,
            "friction_thread": 0.15,
            "friction_bearing": 0.12,
        }
        
        result = run(inputs)
        
        thread_torque = result["results"]["thread_torque"]
        bearing_torque = result["results"]["bearing_torque"]
        total_breakdown = result["results"]["torque_from_breakdown"]
        
        # Sum should equal total
        assert total_breakdown == pytest.approx(thread_torque + bearing_torque, rel=1e-6)
        
        # Both components should be positive
        assert thread_torque > 0
        assert bearing_torque > 0

    def test_step_generation(self):
        """Test that calculation steps are properly generated."""
        inputs = {
            "preload": {"value": 50000, "units": "N"},
            "nominal_diameter": {"value": 0.016, "units": "m"},
            "thread_pitch": {"value": 0.002, "units": "m"},
            "nut_factor": 0.2,
            "friction_thread": 0.15,
            "friction_bearing": 0.12,
            "external_load": {"value": 20000, "units": "N"},
            "proof_load": {"value": 120000, "units": "N"},
        }
        
        result = run(inputs)
        
        steps = result["steps"]
        assert len(steps) >= 5  # Should have multiple calculation steps
        
        # Check that key steps are present
        step_descriptions = [step.description for step in steps]
        assert any("stress area" in desc.lower() for desc in step_descriptions)
        assert any("torque" in desc.lower() for desc in step_descriptions)
        assert any("thread torque" in desc.lower() for desc in step_descriptions)
        assert any("bearing friction" in desc.lower() for desc in step_descriptions)
        assert any("safety" in desc.lower() for desc in step_descriptions)

    def test_stress_area_calculation(self):
        """Test stress area calculation for known thread."""
        inputs = {
            "preload": {"value": 30000, "units": "N"},
            "nominal_diameter": {"value": 0.008, "units": "m"},  # M8
            "thread_pitch": {"value": 0.00125, "units": "m"},   # M8x1.25
            "nut_factor": 0.2,
            "friction_thread": 0.15,
            "friction_bearing": 0.12,
        }
        
        result = run(inputs)
        
        # M8x1.25 stress area ≈ 36.6 mm²
        expected_stress_area = 36.6e-6  # m²
        assert result["results"]["preload_stress_area"] == pytest.approx(expected_stress_area, rel=0.05)
        
        # Stress = Force / Area
        expected_stress = 30000 / expected_stress_area
        assert result["results"]["preload_stress"] == pytest.approx(expected_stress, rel=0.05)

    def test_effective_nut_factor_calculation(self):
        """Test that effective K factor is calculated correctly from friction breakdown."""
        inputs = {
            "preload": {"value": 50000, "units": "N"},
            "nominal_diameter": {"value": 0.016, "units": "m"},
            "thread_pitch": {"value": 0.002, "units": "m"},
            "nut_factor": 0.2,
            "friction_thread": 0.15,
            "friction_bearing": 0.12,
        }
        
        result = run(inputs)
        
        # Effective K = breakdown_torque / (preload * diameter)
        breakdown_torque = result["results"]["torque_from_breakdown"]
        preload = 50000
        diameter = 0.016
        expected_k_eff = breakdown_torque / (preload * diameter)
        
        assert result["results"]["effective_nut_factor"] == pytest.approx(expected_k_eff, rel=1e-6)

    def test_large_diameter_bolt(self):
        """Test calculation for large diameter bolt."""
        inputs = {
            "preload": {"value": 300000, "units": "N"},
            "nominal_diameter": {"value": 0.030, "units": "m"},  # M30
            "thread_pitch": {"value": 0.0035, "units": "m"},
            "nut_factor": 0.15,
            "friction_thread": 0.12,
            "friction_bearing": 0.10,
        }
        
        result = run(inputs)
        
        assert result["results"]["recommended_torque"] > 0
        # Large bolt should have large stress area
        assert result["results"]["preload_stress_area"] > 500e-6  # > 500 mm²