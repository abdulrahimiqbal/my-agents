"""Comprehensive unit converter for physics calculations."""

from typing import List, Dict, Any, Optional, Tuple
from langchain_core.tools import tool
import math


# Unit conversion database organized by physical quantity
UNIT_CONVERSIONS = {
    "length": {
        "base_unit": "meter",
        "units": {
            # Metric
            "meter": 1.0, "m": 1.0,
            "kilometer": 1000.0, "km": 1000.0,
            "centimeter": 0.01, "cm": 0.01,
            "millimeter": 0.001, "mm": 0.001,
            "micrometer": 1e-6, "μm": 1e-6, "micron": 1e-6,
            "nanometer": 1e-9, "nm": 1e-9,
            "picometer": 1e-12, "pm": 1e-12,
            "femtometer": 1e-15, "fm": 1e-15,
            "angstrom": 1e-10, "Å": 1e-10,
            
            # Imperial/US
            "inch": 0.0254, "in": 0.0254,
            "foot": 0.3048, "ft": 0.3048,
            "yard": 0.9144, "yd": 0.9144,
            "mile": 1609.344,
            
            # Astronomical
            "astronomical_unit": 1.495978707e11, "au": 1.495978707e11,
            "light_year": 9.4607304725808e15, "ly": 9.4607304725808e15,
            "parsec": 3.0856775814913673e16, "pc": 3.0856775814913673e16,
            
            # Nuclear
            "fermi": 1e-15,
            "bohr_radius": 5.29177210903e-11,
        }
    },
    
    "mass": {
        "base_unit": "kilogram",
        "units": {
            # Metric
            "kilogram": 1.0, "kg": 1.0,
            "gram": 0.001, "g": 0.001,
            "milligram": 1e-6, "mg": 1e-6,
            "microgram": 1e-9, "μg": 1e-9,
            "tonne": 1000.0, "metric_ton": 1000.0,
            
            # Imperial/US
            "pound": 0.45359237, "lb": 0.45359237, "lbs": 0.45359237,
            "ounce": 0.0283495231, "oz": 0.0283495231,
            "stone": 6.35029318,
            "ton": 907.18474, "short_ton": 907.18474,
            "long_ton": 1016.0469088,
            
            # Atomic/Nuclear
            "atomic_mass_unit": 1.66053906660e-27, "u": 1.66053906660e-27, "amu": 1.66053906660e-27,
            "electron_mass": 9.1093837015e-31,
            "proton_mass": 1.67262192369e-27,
            "neutron_mass": 1.67492749804e-27,
            
            # Astronomical
            "solar_mass": 1.98847e30,
            "earth_mass": 5.9722e24,
        }
    },
    
    "time": {
        "base_unit": "second",
        "units": {
            "second": 1.0, "s": 1.0, "sec": 1.0,
            "minute": 60.0, "min": 60.0,
            "hour": 3600.0, "h": 3600.0, "hr": 3600.0,
            "day": 86400.0,
            "week": 604800.0,
            "month": 2.628e6,  # Average month
            "year": 3.15576e7, "yr": 3.15576e7,
            
            # Scientific
            "millisecond": 0.001, "ms": 0.001,
            "microsecond": 1e-6, "μs": 1e-6,
            "nanosecond": 1e-9, "ns": 1e-9,
            "picosecond": 1e-12, "ps": 1e-12,
            "femtosecond": 1e-15, "fs": 1e-15,
            
            # Physics
            "planck_time": 5.391247e-44,
        }
    },
    
    "energy": {
        "base_unit": "joule",
        "units": {
            # SI
            "joule": 1.0, "j": 1.0,
            "kilojoule": 1000.0, "kj": 1000.0,
            "megajoule": 1e6, "mj": 1e6,
            
            # Common energy units
            "calorie": 4.184, "cal": 4.184,
            "kilocalorie": 4184.0, "kcal": 4184.0,
            "btu": 1055.06, "british_thermal_unit": 1055.06,
            "watt_hour": 3600.0, "wh": 3600.0,
            "kilowatt_hour": 3.6e6, "kwh": 3.6e6,
            
            # Physics/Atomic
            "electron_volt": 1.602176634e-19, "ev": 1.602176634e-19,
            "kiloelectron_volt": 1.602176634e-16, "kev": 1.602176634e-16,
            "megaelectron_volt": 1.602176634e-13, "mev": 1.602176634e-13,
            "gigaelectron_volt": 1.602176634e-10, "gev": 1.602176634e-10,
            "teraelectron_volt": 1.602176634e-7, "tev": 1.602176634e-7,
            
            # Nuclear
            "rydberg": 2.1798723611035e-18,
            "hartree": 4.3597447222071e-18,
        }
    },
    
    "force": {
        "base_unit": "newton",
        "units": {
            "newton": 1.0, "n": 1.0,
            "kilonewton": 1000.0, "kn": 1000.0,
            "meganewton": 1e6, "mn": 1e6,
            "dyne": 1e-5,
            "pound_force": 4.4482216152605, "lbf": 4.4482216152605,
            "kilogram_force": 9.80665, "kgf": 9.80665,
        }
    },
    
    "pressure": {
        "base_unit": "pascal",
        "units": {
            "pascal": 1.0, "pa": 1.0,
            "kilopascal": 1000.0, "kpa": 1000.0,
            "megapascal": 1e6, "mpa": 1e6,
            "gigapascal": 1e9, "gpa": 1e9,
            "bar": 1e5,
            "millibar": 100.0, "mbar": 100.0,
            "atmosphere": 101325.0, "atm": 101325.0,
            "torr": 133.322387415,
            "mmhg": 133.322387415,
            "psi": 6894.757293168, "pounds_per_square_inch": 6894.757293168,
        }
    },
    
    "temperature": {
        # Temperature requires special handling due to offsets
        "base_unit": "kelvin",
        "units": {
            "kelvin": {"scale": 1.0, "offset": 0.0}, "k": {"scale": 1.0, "offset": 0.0},
            "celsius": {"scale": 1.0, "offset": 273.15}, "c": {"scale": 1.0, "offset": 273.15},
            "fahrenheit": {"scale": 5.0/9.0, "offset": 459.67*5.0/9.0}, "f": {"scale": 5.0/9.0, "offset": 459.67*5.0/9.0},
            "rankine": {"scale": 5.0/9.0, "offset": 0.0}, "r": {"scale": 5.0/9.0, "offset": 0.0},
        }
    },
    
    "frequency": {
        "base_unit": "hertz",
        "units": {
            "hertz": 1.0, "hz": 1.0,
            "kilohertz": 1000.0, "khz": 1000.0,
            "megahertz": 1e6, "mhz": 1e6,
            "gigahertz": 1e9, "ghz": 1e9,
            "terahertz": 1e12, "thz": 1e12,
            "rpm": 1.0/60.0, "revolutions_per_minute": 1.0/60.0,
        }
    },
    
    "velocity": {
        "base_unit": "meter_per_second",
        "units": {
            "meter_per_second": 1.0, "m/s": 1.0, "mps": 1.0,
            "kilometer_per_hour": 1.0/3.6, "km/h": 1.0/3.6, "kph": 1.0/3.6,
            "mile_per_hour": 0.44704, "mph": 0.44704,
            "foot_per_second": 0.3048, "ft/s": 0.3048, "fps": 0.3048,
            "knot": 0.514444, "kn": 0.514444,
            "speed_of_light": 299792458.0, "c": 299792458.0,
            "speed_of_sound": 343.0,  # At 20°C in air
        }
    },
}


@tool
def convert_units(value: float, from_unit: str, to_unit: str, quantity: str = "auto") -> str:
    """Convert between different units of measurement.
    
    Args:
        value: Numerical value to convert
        from_unit: Source unit
        to_unit: Target unit
        quantity: Physical quantity (auto, length, mass, time, energy, etc.)
        
    Returns:
        Converted value with explanation
    """
    try:
        from_unit = from_unit.lower().strip()
        to_unit = to_unit.lower().strip()
        
        # Auto-detect quantity if not specified
        if quantity == "auto":
            quantity = _detect_quantity(from_unit, to_unit)
            if not quantity:
                return f"Could not auto-detect quantity for units '{from_unit}' and '{to_unit}'. Please specify the quantity."
        
        quantity = quantity.lower()
        
        if quantity not in UNIT_CONVERSIONS:
            available = ", ".join(UNIT_CONVERSIONS.keys())
            return f"Unknown quantity '{quantity}'. Available: {available}"
        
        # Special handling for temperature
        if quantity == "temperature":
            return _convert_temperature(value, from_unit, to_unit)
        
        # Standard conversion
        units_data = UNIT_CONVERSIONS[quantity]["units"]
        
        if from_unit not in units_data:
            available = ", ".join(units_data.keys())
            return f"Unknown {quantity} unit '{from_unit}'. Available: {available}"
        
        if to_unit not in units_data:
            available = ", ".join(units_data.keys())
            return f"Unknown {quantity} unit '{to_unit}'. Available: {available}"
        
        # Convert to base unit, then to target unit
        base_value = value * units_data[from_unit]
        result = base_value / units_data[to_unit]
        
        return f"{value} {from_unit} = {result:.10g} {to_unit}"
    
    except Exception as e:
        return f"Error converting units: {str(e)}"


def _detect_quantity(from_unit: str, to_unit: str) -> Optional[str]:
    """Auto-detect the physical quantity from unit names."""
    for quantity, data in UNIT_CONVERSIONS.items():
        units = data["units"]
        if from_unit in units and to_unit in units:
            return quantity
    return None


def _convert_temperature(value: float, from_unit: str, to_unit: str) -> str:
    """Convert temperature with proper offset handling."""
    temp_units = UNIT_CONVERSIONS["temperature"]["units"]
    
    if from_unit not in temp_units or to_unit not in temp_units:
        available = ", ".join(temp_units.keys())
        return f"Unknown temperature unit. Available: {available}"
    
    from_data = temp_units[from_unit]
    to_data = temp_units[to_unit]
    
    # Convert to Kelvin first
    kelvin_value = value * from_data["scale"] + from_data["offset"]
    
    # Convert from Kelvin to target
    result = (kelvin_value - to_data["offset"]) / to_data["scale"]
    
    return f"{value}°{from_unit.upper()} = {result:.6g}°{to_unit.upper()}"


@tool
def list_units(quantity: str = "all") -> str:
    """List available units for a given physical quantity.
    
    Args:
        quantity: Physical quantity (all, length, mass, time, energy, etc.)
        
    Returns:
        List of available units
    """
    try:
        if quantity.lower() == "all":
            result = "**Available Physical Quantities and Units:**\n\n"
            for qty, data in UNIT_CONVERSIONS.items():
                result += f"**{qty.title()}** (base: {data['base_unit']}):\n"
                units = list(data["units"].keys())
                # Group similar units
                result += f"  {', '.join(units[:10])}"
                if len(units) > 10:
                    result += f" ... (+{len(units)-10} more)"
                result += "\n\n"
            return result
        
        quantity = quantity.lower()
        if quantity not in UNIT_CONVERSIONS:
            available = ", ".join(UNIT_CONVERSIONS.keys())
            return f"Unknown quantity '{quantity}'. Available: {available}"
        
        data = UNIT_CONVERSIONS[quantity]
        units = list(data["units"].keys())
        
        result = f"**{quantity.title()} Units** (base: {data['base_unit']}):\n\n"
        
        # Group units by type for better readability
        if quantity == "length":
            result += "**Metric:** m, km, cm, mm, μm, nm, pm, fm, Å\n"
            result += "**Imperial:** in, ft, yd, mile\n"
            result += "**Astronomical:** au, ly, pc\n"
            result += "**Nuclear:** fermi, bohr_radius\n"
        elif quantity == "mass":
            result += "**Metric:** kg, g, mg, μg, tonne\n"
            result += "**Imperial:** lb, oz, stone, ton\n"
            result += "**Atomic:** u, amu, electron_mass, proton_mass\n"
            result += "**Astronomical:** solar_mass, earth_mass\n"
        elif quantity == "energy":
            result += "**SI:** J, kJ, MJ\n"
            result += "**Common:** cal, kcal, BTU, Wh, kWh\n"
            result += "**Atomic:** eV, keV, MeV, GeV, TeV\n"
            result += "**Nuclear:** rydberg, hartree\n"
        else:
            # For other quantities, just list all units
            for i in range(0, len(units), 8):
                result += f"  {', '.join(units[i:i+8])}\n"
        
        result += f"\n**Total:** {len(units)} units available"
        return result
    
    except Exception as e:
        return f"Error listing units: {str(e)}"


@tool
def unit_analysis(expression: str) -> str:
    """Analyze the units in a physics expression.
    
    Args:
        expression: Physics expression with units (e.g., "5 m/s * 10 s")
        
    Returns:
        Unit analysis result
    """
    try:
        # This is a simplified unit analysis
        # A full implementation would parse the expression and track units
        
        return f"""**Unit Analysis for:** {expression}

This is a simplified unit analysis tool. For complete analysis:

1. **Identify all quantities and their units**
2. **Apply dimensional analysis rules:**
   - Multiplication: units multiply
   - Division: units divide
   - Addition/Subtraction: units must match
   - Powers: units are raised to the power

3. **Check dimensional consistency**
4. **Simplify resulting units**

**Example:** 5 m/s × 10 s = 50 m
- Units: (m/s) × s = m ✓

**Common Unit Combinations:**
- Force: kg⋅m/s² = N
- Energy: kg⋅m²/s² = J
- Power: kg⋅m²/s³ = W
- Pressure: kg/(m⋅s²) = Pa

For detailed unit analysis, break down complex expressions into simpler parts."""
    
    except Exception as e:
        return f"Error in unit analysis: {str(e)}"


@tool
def physics_unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between specialized physics units with context.
    
    Args:
        value: Value to convert
        from_unit: Source unit
        to_unit: Target unit
        
    Returns:
        Conversion with physics context
    """
    try:
        # Use the main converter
        result = convert_units(value, from_unit, to_unit)
        
        # Add physics context for common conversions
        context = _get_physics_context(from_unit, to_unit, value)
        
        if context:
            result += f"\n\n**Physics Context:**\n{context}"
        
        return result
    
    except Exception as e:
        return f"Error in physics unit conversion: {str(e)}"


def _get_physics_context(from_unit: str, to_unit: str, value: float) -> str:
    """Provide physics context for unit conversions."""
    contexts = {
        ("ev", "j"): "Electron volts to joules - useful for atomic and molecular energy scales",
        ("j", "ev"): "Joules to electron volts - converting macroscopic to atomic energy scales",
        ("angstrom", "m"): "Angstroms to meters - typical atomic and molecular length scales",
        ("m", "angstrom"): "Meters to angstroms - converting to atomic length scales",
        ("c", "m/s"): "Speed of light - fundamental constant in relativity",
        ("ly", "m"): "Light years to meters - astronomical distance conversion",
        ("au", "m"): "Astronomical units to meters - solar system distance scale",
        ("solar_mass", "kg"): "Solar masses to kilograms - stellar mass scale",
        ("earth_mass", "kg"): "Earth masses to kilograms - planetary mass scale",
    }
    
    key = (from_unit.lower(), to_unit.lower())
    return contexts.get(key, "")


def get_unit_converter_tools() -> List:
    """Get all unit converter tools.
    
    Returns:
        List of unit converter tools
    """
    return [
        convert_units,
        list_units,
        unit_analysis,
        physics_unit_converter
    ] 