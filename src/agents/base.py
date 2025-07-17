"""
Base agent class for the physics research system
"""

class BaseAgent:
    """Base class for all agents in the system"""
    
    def __init__(self, name: str = "BaseAgent"):
        self.name = name
    
    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"
    
    def __repr__(self):
        return self.__str__()