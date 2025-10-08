"""Centralized unit handling utilities built on top of pint."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable

from pint import Quantity, UnitRegistry


@lru_cache(maxsize=1)
def get_unit_registry() -> UnitRegistry:
    """Return a singleton :class:`~pint.UnitRegistry` instance.

    The registry is configured for engineering calculations with support for
    common aliases and contexts. The function is memoized to avoid recreating
    the registry, which is both expensive and can lead to mismatched quantity
    comparisons.
    """

    registry = UnitRegistry(autoconvert_offset_to_baseunit=True)
    registry.default_format = ".5gP"
    # Helpful shorthands
    registry.define("percent = 1 / 100 = pct")
    registry.define("rpm = revolution / minute")
    # US customary flow units
    registry.define("gpm = gallon / minute")
    # US customary length units for roughness
    registry.define("mils = 0.001 * inch = mil")
    # US thermal units
    registry.define("Btu_per_lb_degF = Btu / (pound * degree_Fahrenheit)")
    registry.define("Btu_per_hr_ft2_degF = Btu / (hour * foot**2 * degree_Fahrenheit)")
    # Alternative formats for thermal units
    registry.define("lbdegreeF = pound * degree_Fahrenheit")
    # Add aliases for common thermal unit formats with different syntaxes
    registry.define("Btu_per_lb_F = Btu_per_lb_degF")
    registry.define("Btu_per_hr_ft_2_F = Btu_per_hr_ft2_degF")
    # Direct unit definition for the problematic format
    registry.define("Btu_hr_ft2_degF = Btu / hour / foot**2 / degree_Fahrenheit")
    registry.define("@alias Btu_hr_ft2_degF = Btu/hr/ft^2°F")
    return registry


def u(value: Any, units: str | None = None) -> Quantity:
    """Create a :class:`~pint.Quantity` coerced to the provided units.

    Parameters
    ----------
    value:
        Numeric value or a :class:`~pint.Quantity`.
    units:
        Unit string interpretable by pint. If ``value`` is already a quantity
        and ``units`` is provided, the quantity is converted to the requested
        units.
    """

    registry = get_unit_registry()
    if isinstance(value, Quantity):
        return value.to(units) if units else value
    if units is None:
        raise ValueError("Units must be provided when creating a quantity from a plain number.")
    return registry.Quantity(value, units)


def format_quantity(quantity: Quantity, precision: int = 4) -> str:
    """Render a quantity with a sensible precision for UI display."""

    magnitude = round(quantity.magnitude, precision)
    return f"{magnitude} {quantity.units:~P}"


@dataclass
class UnitField:
    """Utility descriptor for pydantic models capturing value + units inputs."""

    name: str
    allowed_units: Iterable[str] | None = None

    def validate(self, value: Any) -> Quantity:
        """Validate and normalize values against optional allowed units."""

        if isinstance(value, dict):
            magnitude = value.get("value")
            units = value.get("units")
        else:
            magnitude, units = value, None

        if magnitude is None:
            raise ValueError(f"{self.name} requires a numeric value.")

        if units is None:
            raise ValueError(f"{self.name} requires explicit units.")

        quantity = u(magnitude, units)
        if self.allowed_units:
            registry = get_unit_registry()
            quantity_dim = quantity.dimensionality
            if not any(quantity_dim == registry.Quantity(1, u).dimensionality for u in self.allowed_units):
                allowed = ", ".join(self.allowed_units)
                raise ValueError(f"{self.name} must use units compatible with: {allowed}")
        return quantity

