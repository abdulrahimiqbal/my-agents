"""
Calculator tools for mathematical operations.

Provides safe mathematical computation tools following LangChain Academy
tool creation patterns from Module 1.
"""

import math
import operator
from typing import List, Union
from langchain_core.tools import BaseTool, tool


@tool
def add(a: float, b: float) -> float:
    """
    Add two numbers together.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Sum of a and b
    """
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """
    Subtract second number from first number.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Difference of a and b (a - b)
    """
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """
    Multiply two numbers together.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Product of a and b
    """
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """
    Divide first number by second number.
    
    Args:
        a: Dividend
        b: Divisor
        
    Returns:
        Quotient of a divided by b
        
    Raises:
        ValueError: If b is zero
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


@tool
def power(base: float, exponent: float) -> float:
    """
    Raise base to the power of exponent.
    
    Args:
        base: Base number
        exponent: Exponent
        
    Returns:
        base raised to the power of exponent
    """
    return base ** exponent


@tool
def square_root(number: float) -> float:
    """
    Calculate square root of a number.
    
    Args:
        number: Number to find square root of
        
    Returns:
        Square root of the number
        
    Raises:
        ValueError: If number is negative
    """
    if number < 0:
        raise ValueError("Cannot calculate square root of negative number")
    return math.sqrt(number)


@tool
def calculate_expression(expression: str) -> Union[float, str]:
    """
    Safely evaluate a mathematical expression.
    
    Args:
        expression: Mathematical expression as string (e.g., "2 + 3 * 4")
        
    Returns:
        Result of the expression or error message
    """
    # Define safe operations
    safe_dict = {
        "__builtins__": {},
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "pi": math.pi,
        "e": math.e,
    }
    
    try:
        # Remove any potentially dangerous characters/functions
        if any(dangerous in expression.lower() for dangerous in 
               ['import', 'exec', 'eval', 'open', 'file', '__']):
            return "Error: Expression contains potentially dangerous operations"
        
        result = eval(expression, safe_dict, {})
        return float(result)
    except ZeroDivisionError:
        return "Error: Division by zero"
    except ValueError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Error: Invalid expression - {str(e)}"


def get_calculator_tools() -> List[BaseTool]:
    """
    Get all calculator tools.
    
    Returns:
        List of calculator tools
    """
    return [
        add,
        subtract,
        multiply,
        divide,
        power,
        square_root,
        calculate_expression,
    ] 