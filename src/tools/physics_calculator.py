"""Advanced physics calculator tools for scientific computations."""

import math
import cmath
import numpy as np
from typing import List, Union, Tuple, Optional
from langchain_core.tools import tool


@tool
def scientific_calculator(expression: str) -> str:
    """Evaluate scientific mathematical expressions with physics functions.
    
    Supports:
    - Basic operations: +, -, *, /, **, %
    - Scientific functions: sin, cos, tan, log, ln, exp, sqrt
    - Constants: pi, e
    - Complex numbers: use j for imaginary unit
    
    Args:
        expression: Mathematical expression to evaluate
        
    Returns:
        Result of the calculation with explanation
    """
    try:
        # Define safe functions and constants for eval
        safe_dict = {
            # Basic math functions
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
            'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh,
            'log': math.log10, 'ln': math.log, 'log10': math.log10,
            'exp': math.exp, 'sqrt': math.sqrt,
            'abs': abs, 'pow': pow,
            
            # Constants
            'pi': math.pi, 'e': math.e,
            
            # Complex math
            'j': 1j, 'complex': complex,
            
            # NumPy functions for arrays
            'np': np,
        }
        
        # Evaluate the expression
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        
        # Format the result appropriately
        if isinstance(result, complex):
            if result.imag == 0:
                formatted_result = f"{result.real:.6g}"
            else:
                formatted_result = f"{result.real:.6g} + {result.imag:.6g}j"
        elif isinstance(result, (int, float)):
            formatted_result = f"{result:.6g}"
        else:
            formatted_result = str(result)
        
        return f"Expression: {expression}\nResult: {formatted_result}"
        
    except Exception as e:
        return f"Error evaluating expression '{expression}': {str(e)}"


@tool
def vector_operations(operation: str, vector1: str, vector2: str = None) -> str:
    """Perform vector operations for physics calculations.
    
    Supported operations:
    - magnitude: |v|
    - normalize: v/|v|
    - dot: v1 · v2
    - cross: v1 × v2
    - add: v1 + v2
    - subtract: v1 - v2
    - scale: scalar * v1
    
    Args:
        operation: Type of operation (magnitude, normalize, dot, cross, add, subtract, scale)
        vector1: First vector as comma-separated values (e.g., "1,2,3")
        vector2: Second vector or scalar (for applicable operations)
        
    Returns:
        Result of the vector operation
    """
    try:
        # Parse the first vector
        v1 = np.array([float(x.strip()) for x in vector1.split(',')])
        
        if operation == "magnitude":
            result = np.linalg.norm(v1)
            return f"Magnitude of {vector1}: {result:.6g}"
        
        elif operation == "normalize":
            magnitude = np.linalg.norm(v1)
            if magnitude == 0:
                return "Cannot normalize zero vector"
            normalized = v1 / magnitude
            return f"Normalized vector: [{', '.join(f'{x:.6g}' for x in normalized)}]"
        
        elif operation in ["dot", "cross", "add", "subtract"]:
            if vector2 is None:
                return f"Operation '{operation}' requires a second vector"
            
            v2 = np.array([float(x.strip()) for x in vector2.split(',')])
            
            if operation == "dot":
                result = np.dot(v1, v2)
                return f"Dot product: {vector1} · {vector2} = {result:.6g}"
            
            elif operation == "cross":
                if len(v1) != 3 or len(v2) != 3:
                    return "Cross product requires 3D vectors"
                result = np.cross(v1, v2)
                return f"Cross product: {vector1} × {vector2} = [{', '.join(f'{x:.6g}' for x in result)}]"
            
            elif operation == "add":
                result = v1 + v2
                return f"Vector addition: {vector1} + {vector2} = [{', '.join(f'{x:.6g}' for x in result)}]"
            
            elif operation == "subtract":
                result = v1 - v2
                return f"Vector subtraction: {vector1} - {vector2} = [{', '.join(f'{x:.6g}' for x in result)}]"
        
        elif operation == "scale":
            if vector2 is None:
                return "Scale operation requires a scalar value"
            scalar = float(vector2)
            result = scalar * v1
            return f"Scaled vector: {scalar} * {vector1} = [{', '.join(f'{x:.6g}' for x in result)}]"
        
        else:
            return f"Unknown operation: {operation}. Supported: magnitude, normalize, dot, cross, add, subtract, scale"
    
    except Exception as e:
        return f"Error in vector operation: {str(e)}"


@tool
def solve_quadratic(a: float, b: float, c: float) -> str:
    """Solve quadratic equation ax² + bx + c = 0.
    
    Args:
        a: Coefficient of x²
        b: Coefficient of x
        c: Constant term
        
    Returns:
        Solutions of the quadratic equation
    """
    try:
        if a == 0:
            if b == 0:
                if c == 0:
                    return "Infinite solutions (0 = 0)"
                else:
                    return "No solution (inconsistent equation)"
            else:
                solution = -c / b
                return f"Linear equation solution: x = {solution:.6g}"
        
        # Calculate discriminant
        discriminant = b**2 - 4*a*c
        
        if discriminant > 0:
            x1 = (-b + math.sqrt(discriminant)) / (2*a)
            x2 = (-b - math.sqrt(discriminant)) / (2*a)
            return f"Two real solutions:\nx₁ = {x1:.6g}\nx₂ = {x2:.6g}"
        
        elif discriminant == 0:
            x = -b / (2*a)
            return f"One repeated solution:\nx = {x:.6g}"
        
        else:
            real_part = -b / (2*a)
            imag_part = math.sqrt(-discriminant) / (2*a)
            return f"Two complex solutions:\nx₁ = {real_part:.6g} + {imag_part:.6g}i\nx₂ = {real_part:.6g} - {imag_part:.6g}i"
    
    except Exception as e:
        return f"Error solving quadratic equation: {str(e)}"


@tool
def physics_functions(function: str, value: float, unit: str = "") -> str:
    """Calculate common physics functions and conversions.
    
    Supported functions:
    - kinetic_energy: KE = ½mv² (requires mass and velocity)
    - potential_energy: PE = mgh (requires mass, gravity, height)
    - frequency_to_wavelength: λ = c/f
    - wavelength_to_frequency: f = c/λ
    - energy_to_frequency: f = E/h
    - frequency_to_energy: E = hf
    - degrees_to_radians: rad = deg × π/180
    - radians_to_degrees: deg = rad × 180/π
    
    Args:
        function: Physics function to calculate
        value: Input value
        unit: Unit of the input value
        
    Returns:
        Calculated result with appropriate units
    """
    try:
        # Physics constants
        c = 299792458  # speed of light in m/s
        h = 6.62607015e-34  # Planck constant in J⋅s
        g = 9.80665  # standard gravity in m/s²
        
        if function == "frequency_to_wavelength":
            wavelength = c / value
            return f"Wavelength: λ = c/f = {c:.3e} m/s / {value:.6g} Hz = {wavelength:.6g} m"
        
        elif function == "wavelength_to_frequency":
            frequency = c / value
            return f"Frequency: f = c/λ = {c:.3e} m/s / {value:.6g} m = {frequency:.6g} Hz"
        
        elif function == "energy_to_frequency":
            frequency = value / h
            return f"Frequency: f = E/h = {value:.6g} J / {h:.6e} J⋅s = {frequency:.6g} Hz"
        
        elif function == "frequency_to_energy":
            energy = h * value
            return f"Energy: E = hf = {h:.6e} J⋅s × {value:.6g} Hz = {energy:.6g} J"
        
        elif function == "degrees_to_radians":
            radians = value * math.pi / 180
            return f"Angle: {value:.6g}° = {radians:.6g} rad"
        
        elif function == "radians_to_degrees":
            degrees = value * 180 / math.pi
            return f"Angle: {value:.6g} rad = {degrees:.6g}°"
        
        else:
            return f"Unknown function: {function}. Available: frequency_to_wavelength, wavelength_to_frequency, energy_to_frequency, frequency_to_energy, degrees_to_radians, radians_to_degrees"
    
    except Exception as e:
        return f"Error calculating physics function: {str(e)}"


@tool
def dimensional_analysis(equation: str, variables: str) -> str:
    """Perform dimensional analysis on physics equations.
    
    Args:
        equation: Physics equation (e.g., "F = ma")
        variables: Variable dimensions (e.g., "F:[M L T^-2], m:[M], a:[L T^-2]")
        
    Returns:
        Dimensional analysis result
    """
    try:
        # This is a simplified dimensional analysis tool
        # In a full implementation, you'd parse the equation and check dimensional consistency
        
        return f"""Dimensional Analysis for: {equation}
Variables: {variables}

Note: This is a simplified tool. For complete dimensional analysis:
1. Express each variable in terms of fundamental dimensions [M], [L], [T]
2. Substitute dimensions into the equation
3. Check that both sides have the same dimensions
4. Verify the equation is dimensionally consistent

Example for F = ma:
- F: [M L T⁻²] (force)
- m: [M] (mass)  
- a: [L T⁻²] (acceleration)
- Check: [M L T⁻²] = [M] × [L T⁻²] ✓ Consistent!
"""
    
    except Exception as e:
        return f"Error in dimensional analysis: {str(e)}"


def get_physics_calculator_tools() -> List:
    """Get all physics calculator tools.
    
    Returns:
        List of physics calculator tools
    """
    return [
        scientific_calculator,
        vector_operations,
        solve_quadratic,
        physics_functions,
        dimensional_analysis
    ] 