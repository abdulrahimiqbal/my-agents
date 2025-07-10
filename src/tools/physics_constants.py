"""Physics constants database for scientific calculations."""

from typing import List, Dict, Any, Optional
from langchain_core.tools import tool


# Comprehensive physics constants database
PHYSICS_CONSTANTS = {
    # Fundamental Constants
    "speed_of_light": {
        "symbol": "c",
        "value": 299792458,
        "unit": "m/s",
        "description": "Speed of light in vacuum",
        "uncertainty": "exact"
    },
    "planck_constant": {
        "symbol": "h",
        "value": 6.62607015e-34,
        "unit": "J⋅s",
        "description": "Planck constant",
        "uncertainty": "exact"
    },
    "reduced_planck_constant": {
        "symbol": "ℏ",
        "value": 1.054571817e-34,
        "unit": "J⋅s",
        "description": "Reduced Planck constant (h/2π)",
        "uncertainty": "exact"
    },
    "elementary_charge": {
        "symbol": "e",
        "value": 1.602176634e-19,
        "unit": "C",
        "description": "Elementary charge",
        "uncertainty": "exact"
    },
    "electron_mass": {
        "symbol": "mₑ",
        "value": 9.1093837015e-31,
        "unit": "kg",
        "description": "Electron rest mass",
        "uncertainty": "3.0e-40 kg"
    },
    "proton_mass": {
        "symbol": "mₚ",
        "value": 1.67262192369e-27,
        "unit": "kg",
        "description": "Proton rest mass",
        "uncertainty": "5.1e-37 kg"
    },
    "neutron_mass": {
        "symbol": "mₙ",
        "value": 1.67492749804e-27,
        "unit": "kg",
        "description": "Neutron rest mass",
        "uncertainty": "9.5e-37 kg"
    },
    "avogadro_constant": {
        "symbol": "Nₐ",
        "value": 6.02214076e23,
        "unit": "mol⁻¹",
        "description": "Avogadro constant",
        "uncertainty": "exact"
    },
    "boltzmann_constant": {
        "symbol": "k",
        "value": 1.380649e-23,
        "unit": "J/K",
        "description": "Boltzmann constant",
        "uncertainty": "exact"
    },
    "gas_constant": {
        "symbol": "R",
        "value": 8.314462618,
        "unit": "J/(mol⋅K)",
        "description": "Universal gas constant",
        "uncertainty": "exact"
    },
    "gravitational_constant": {
        "symbol": "G",
        "value": 6.67430e-11,
        "unit": "m³/(kg⋅s²)",
        "description": "Gravitational constant",
        "uncertainty": "1.5e-15 m³/(kg⋅s²)"
    },
    "standard_gravity": {
        "symbol": "g",
        "value": 9.80665,
        "unit": "m/s²",
        "description": "Standard acceleration due to gravity",
        "uncertainty": "exact"
    },
    "vacuum_permeability": {
        "symbol": "μ₀",
        "value": 1.25663706212e-6,
        "unit": "H/m",
        "description": "Vacuum permeability",
        "uncertainty": "1.9e-16 H/m"
    },
    "vacuum_permittivity": {
        "symbol": "ε₀",
        "value": 8.8541878128e-12,
        "unit": "F/m",
        "description": "Vacuum permittivity",
        "uncertainty": "1.3e-21 F/m"
    },
    "fine_structure_constant": {
        "symbol": "α",
        "value": 7.2973525693e-3,
        "unit": "dimensionless",
        "description": "Fine-structure constant",
        "uncertainty": "1.1e-12"
    },
    "stefan_boltzmann_constant": {
        "symbol": "σ",
        "value": 5.670374419e-8,
        "unit": "W/(m²⋅K⁴)",
        "description": "Stefan-Boltzmann constant",
        "uncertainty": "exact"
    },
    "wien_displacement_constant": {
        "symbol": "b",
        "value": 2.897771955e-3,
        "unit": "m⋅K",
        "description": "Wien wavelength displacement law constant",
        "uncertainty": "exact"
    },
    "rydberg_constant": {
        "symbol": "R∞",
        "value": 1.0973731568160e7,
        "unit": "m⁻¹",
        "description": "Rydberg constant",
        "uncertainty": "2.1e-5 m⁻¹"
    },
    "bohr_radius": {
        "symbol": "a₀",
        "value": 5.29177210903e-11,
        "unit": "m",
        "description": "Bohr radius",
        "uncertainty": "8.0e-21 m"
    },
    "atomic_mass_unit": {
        "symbol": "u",
        "value": 1.66053906660e-27,
        "unit": "kg",
        "description": "Atomic mass unit",
        "uncertainty": "5.0e-37 kg"
    },
    
    # Astronomical Constants
    "solar_mass": {
        "symbol": "M☉",
        "value": 1.98847e30,
        "unit": "kg",
        "description": "Solar mass",
        "uncertainty": "4.5e25 kg"
    },
    "earth_mass": {
        "symbol": "M⊕",
        "value": 5.9722e24,
        "unit": "kg",
        "description": "Earth mass",
        "uncertainty": "6e20 kg"
    },
    "astronomical_unit": {
        "symbol": "AU",
        "value": 1.495978707e11,
        "unit": "m",
        "description": "Astronomical unit",
        "uncertainty": "exact"
    },
    "parsec": {
        "symbol": "pc",
        "value": 3.0856775814913673e16,
        "unit": "m",
        "description": "Parsec",
        "uncertainty": "exact"
    },
    "light_year": {
        "symbol": "ly",
        "value": 9.4607304725808e15,
        "unit": "m",
        "description": "Light year",
        "uncertainty": "exact"
    },
    
    # Nuclear and Particle Physics
    "electron_volt": {
        "symbol": "eV",
        "value": 1.602176634e-19,
        "unit": "J",
        "description": "Electron volt",
        "uncertainty": "exact"
    },
    "classical_electron_radius": {
        "symbol": "rₑ",
        "value": 2.8179403262e-15,
        "unit": "m",
        "description": "Classical electron radius",
        "uncertainty": "1.3e-24 m"
    },
    "compton_wavelength": {
        "symbol": "λc",
        "value": 2.42631023867e-12,
        "unit": "m",
        "description": "Compton wavelength",
        "uncertainty": "7.3e-22 m"
    }
}


@tool
def get_physics_constant(name: str) -> str:
    """Get a physics constant by name.
    
    Args:
        name: Name of the constant (e.g., 'speed_of_light', 'planck_constant')
        
    Returns:
        Detailed information about the physics constant
    """
    try:
        # Allow partial matching and common aliases
        name_lower = name.lower().replace(" ", "_").replace("-", "_")
        
        # Common aliases
        aliases = {
            "c": "speed_of_light",
            "light_speed": "speed_of_light",
            "h": "planck_constant",
            "hbar": "reduced_planck_constant",
            "h_bar": "reduced_planck_constant",
            "e": "elementary_charge",
            "electron_charge": "elementary_charge",
            "me": "electron_mass",
            "mp": "proton_mass",
            "mn": "neutron_mass",
            "na": "avogadro_constant",
            "avogadro": "avogadro_constant",
            "k": "boltzmann_constant",
            "kb": "boltzmann_constant",
            "r": "gas_constant",
            "g": "gravitational_constant",
            "gravity": "standard_gravity",
            "g0": "standard_gravity",
            "mu0": "vacuum_permeability",
            "eps0": "vacuum_permittivity",
            "epsilon0": "vacuum_permittivity",
            "alpha": "fine_structure_constant",
            "sigma": "stefan_boltzmann_constant",
            "stefan": "stefan_boltzmann_constant",
            "wien": "wien_displacement_constant",
            "rydberg": "rydberg_constant",
            "bohr": "bohr_radius",
            "amu": "atomic_mass_unit",
            "ev": "electron_volt"
        }
        
        # Check aliases first
        if name_lower in aliases:
            name_lower = aliases[name_lower]
        
        # Search for the constant
        if name_lower in PHYSICS_CONSTANTS:
            const = PHYSICS_CONSTANTS[name_lower]
            result = f"""**{const['description']}**
Symbol: {const['symbol']}
Value: {const['value']:.10g} {const['unit']}
Uncertainty: {const['uncertainty']}"""
            return result
        
        # If not found, search for partial matches
        matches = [key for key in PHYSICS_CONSTANTS.keys() if name_lower in key]
        if matches:
            if len(matches) == 1:
                const = PHYSICS_CONSTANTS[matches[0]]
                result = f"""**{const['description']}**
Symbol: {const['symbol']}
Value: {const['value']:.10g} {const['unit']}
Uncertainty: {const['uncertainty']}"""
                return result
            else:
                return f"Multiple matches found: {', '.join(matches)}. Please be more specific."
        
        return f"Constant '{name}' not found. Use list_physics_constants() to see available constants."
    
    except Exception as e:
        return f"Error retrieving constant: {str(e)}"


@tool
def list_physics_constants(category: str = "all") -> str:
    """List available physics constants by category.
    
    Args:
        category: Category to filter (all, fundamental, astronomical, nuclear, electromagnetic)
        
    Returns:
        List of available physics constants
    """
    try:
        fundamental = [
            "speed_of_light", "planck_constant", "reduced_planck_constant",
            "elementary_charge", "gravitational_constant", "fine_structure_constant"
        ]
        
        particles = [
            "electron_mass", "proton_mass", "neutron_mass", "atomic_mass_unit",
            "classical_electron_radius", "compton_wavelength", "bohr_radius"
        ]
        
        thermodynamic = [
            "boltzmann_constant", "gas_constant", "avogadro_constant",
            "stefan_boltzmann_constant", "wien_displacement_constant"
        ]
        
        electromagnetic = [
            "vacuum_permeability", "vacuum_permittivity", "elementary_charge"
        ]
        
        astronomical = [
            "solar_mass", "earth_mass", "astronomical_unit", "parsec", "light_year"
        ]
        
        nuclear = [
            "electron_volt", "rydberg_constant", "standard_gravity"
        ]
        
        if category.lower() == "fundamental":
            constants_to_show = fundamental
        elif category.lower() == "particles":
            constants_to_show = particles
        elif category.lower() == "thermodynamic":
            constants_to_show = thermodynamic
        elif category.lower() == "electromagnetic":
            constants_to_show = electromagnetic
        elif category.lower() == "astronomical":
            constants_to_show = astronomical
        elif category.lower() == "nuclear":
            constants_to_show = nuclear
        else:
            constants_to_show = list(PHYSICS_CONSTANTS.keys())
        
        result = f"**Physics Constants - {category.title()}**\n\n"
        
        for const_name in constants_to_show:
            if const_name in PHYSICS_CONSTANTS:
                const = PHYSICS_CONSTANTS[const_name]
                result += f"• **{const_name}** ({const['symbol']}): {const['description']}\n"
        
        result += f"\nTotal: {len(constants_to_show)} constants"
        result += "\nUse get_physics_constant('name') for detailed information."
        
        return result
    
    except Exception as e:
        return f"Error listing constants: {str(e)}"


@tool
def calculate_derived_constant(formula: str, constants: str) -> str:
    """Calculate derived physics constants from fundamental ones.
    
    Args:
        formula: Mathematical formula (e.g., "c**2", "h*c", "e**2/(4*pi*eps0)")
        constants: Constants to use (e.g., "c, h, e, eps0")
        
    Returns:
        Calculated value with explanation
    """
    try:
        # Create a safe evaluation environment with physics constants
        safe_dict = {
            "pi": 3.141592653589793,
            "sqrt": lambda x: x**0.5,
            "exp": lambda x: 2.718281828459045**x,
            "log": lambda x: __import__('math').log(x),
            "sin": lambda x: __import__('math').sin(x),
            "cos": lambda x: __import__('math').cos(x),
            "tan": lambda x: __import__('math').tan(x),
        }
        
        # Add requested constants to the environment
        const_list = [c.strip() for c in constants.split(',')]
        used_constants = []
        
        for const_name in const_list:
            const_name_clean = const_name.lower().replace(" ", "_")
            
            # Handle aliases
            aliases = {
                "c": "speed_of_light",
                "h": "planck_constant",
                "hbar": "reduced_planck_constant",
                "e": "elementary_charge",
                "eps0": "vacuum_permittivity",
                "mu0": "vacuum_permeability",
                "k": "boltzmann_constant",
                "g": "gravitational_constant"
            }
            
            if const_name_clean in aliases:
                const_name_clean = aliases[const_name_clean]
            
            if const_name_clean in PHYSICS_CONSTANTS:
                const_data = PHYSICS_CONSTANTS[const_name_clean]
                safe_dict[const_name] = const_data["value"]
                used_constants.append(f"{const_name} = {const_data['value']:.6g} {const_data['unit']}")
        
        # Evaluate the formula
        result = eval(formula, {"__builtins__": {}}, safe_dict)
        
        response = f"""**Derived Constant Calculation**

Formula: {formula}

Constants used:
{chr(10).join(used_constants)}

Result: {result:.10g}

Note: Units depend on the formula and input constants."""
        
        return response
    
    except Exception as e:
        return f"Error calculating derived constant: {str(e)}"


def get_physics_constants_tools() -> List:
    """Get all physics constants tools.
    
    Returns:
        List of physics constants tools
    """
    return [
        get_physics_constant,
        list_physics_constants,
        calculate_derived_constant
    ] 