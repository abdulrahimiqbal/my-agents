"""
Real Physics Expert Agent with LLM Integration
"""

import asyncio
import os
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .base import BaseAgent
from ..tools.physics_calculator import get_physics_calculator_tools
from ..tools.physics_research import get_physics_research_tools

# Load environment variables
load_dotenv()

# Simple settings fallback
def get_default_settings():
    return {
        "default_model": os.getenv("PHYSICS_AGENT_MODEL", "gpt-4o-mini"),
        "temperature": float(os.getenv("PHYSICS_AGENT_TEMPERATURE", "0.1"))
    }


class RealPhysicsExpertAgent(BaseAgent):
    """Real Physics Expert Agent with actual LLM integration."""
    
    def __init__(self, model_name: str = None, temperature: float = 0.1):
        super().__init__()
        
        settings = get_default_settings()
        self.model_name = model_name or settings["default_model"]
        self.temperature = temperature
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=2000
        )
        
        # Physics expert system prompt
        self.system_prompt = """You are an advanced physics expert AI with deep knowledge across all domains of physics including:

- Quantum Mechanics & Quantum Field Theory
- Classical Mechanics & Dynamics  
- Thermodynamics & Statistical Mechanics
- Electromagnetism & Optics
- Relativity (Special & General)
- Condensed Matter Physics
- Particle Physics & Cosmology
- Mathematical Physics

Your responses should be:
1. Scientifically accurate and rigorous
2. Appropriately detailed for the question complexity
3. Include relevant equations when helpful
4. Explain physical intuition behind concepts
5. Reference experimental evidence when relevant
6. Acknowledge limitations or uncertainties

Always structure your response with clear sections and use proper physics terminology."""

        # Create prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{query}")
        ])
        
        # Create chain
        self.chain = self.prompt_template | self.llm | StrOutputParser()
        
        # Load physics tools
        self.tools = []
        self.tools.extend(get_physics_calculator_tools())
        self.tools.extend(get_physics_research_tools())
    
    async def analyze_physics_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze a physics query with real LLM."""
        try:
            # Generate response using LLM
            response = await self.chain.ainvoke({"query": query})
            
            # Calculate confidence based on response quality
            confidence = self._calculate_confidence(response, query)
            
            return {
                "response": response,
                "confidence": confidence,
                "agent": "real_physics_expert",
                "model_used": self.model_name,
                "tools_available": len(self.tools),
                "context_used": context is not None
            }
            
        except Exception as e:
            return {
                "response": f"Error in physics analysis: {str(e)}",
                "confidence": 0.0,
                "agent": "real_physics_expert",
                "error": str(e)
            }
    
    def _calculate_confidence(self, response: str, query: str) -> float:
        """Calculate confidence score based on response quality."""
        confidence = 0.5  # Base confidence
        
        # Length-based confidence (longer responses often more detailed)
        if len(response) > 200:
            confidence += 0.1
        if len(response) > 500:
            confidence += 0.1
        
        # Physics terminology presence
        physics_terms = [
            "equation", "theory", "principle", "law", "force", "energy", 
            "momentum", "quantum", "relativity", "thermodynamics", "field"
        ]
        term_count = sum(1 for term in physics_terms if term.lower() in response.lower())
        confidence += min(term_count * 0.05, 0.2)
        
        # Mathematical content
        if any(symbol in response for symbol in ["=", "∝", "∫", "∂", "Σ"]):
            confidence += 0.1
        
        # Structured response
        if any(marker in response for marker in ["1.", "2.", "•", "-", "**"]):
            confidence += 0.05
        
        return min(confidence, 0.95)  # Cap at 95%
    
    async def process_with_tools(self, query: str) -> Dict[str, Any]:
        """Process query with available physics tools."""
        # This would integrate with actual tool calling
        # For now, return the basic analysis
        return await self.analyze_physics_query(query)
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information."""
        return {
            "name": "Real Physics Expert Agent",
            "type": "physics_expert",
            "model": self.model_name,
            "temperature": self.temperature,
            "capabilities": [
                "quantum_mechanics", "thermodynamics", "electromagnetism",
                "mechanics", "relativity", "general_physics"
            ],
            "tools_count": len(self.tools),
            "description": "Advanced physics expert with real LLM integration"
        }