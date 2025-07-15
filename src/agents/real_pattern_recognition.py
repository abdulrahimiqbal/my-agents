"""
Real Pattern Recognition Agent with LLM Integration
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

# Load environment variables
load_dotenv()

# Simple settings fallback
def get_default_settings():
    return {
        "default_model": os.getenv("PHYSICS_AGENT_MODEL", "gpt-4o-mini"),
        "temperature": float(os.getenv("PHYSICS_AGENT_TEMPERATURE", "0.3"))
    }


class RealPatternRecognitionAgent(BaseAgent):
    """Real Pattern Recognition Agent with LLM integration for discovering patterns in physics data."""
    
    def __init__(self, model_name: str = None, temperature: float = 0.3):
        super().__init__()
        
        settings = get_default_settings()
        self.model_name = model_name or settings["default_model"]
        self.temperature = temperature  # Medium temperature for balanced creativity and accuracy
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=1500
        )
        
        # Pattern recognition system prompt
        self.system_prompt = """You are an advanced pattern recognition AI specialized in:

1. **Data Pattern Analysis**: Identifying trends, correlations, and anomalies in physics data
2. **Mathematical Pattern Discovery**: Finding underlying mathematical relationships
3. **Physical Law Recognition**: Discovering fundamental physics principles from observations
4. **Symmetry Detection**: Identifying symmetries and conservation laws
5. **Scaling Relationships**: Finding power laws and scaling behaviors
6. **Emergent Phenomena**: Recognizing emergent properties from complex systems

Your approach should be:
- Systematic and methodical in pattern analysis
- Mathematically rigorous when describing patterns
- Connect patterns to known physics principles
- Identify both obvious and subtle patterns
- Suggest experimental validation methods
- Consider statistical significance

Format your response with:
🔍 **Pattern Analysis**: [Detailed pattern description]
📊 **Mathematical Relationship**: [Equations and formulas]
🔬 **Physics Connection**: [Link to known principles]
📈 **Significance**: [Statistical and physical importance]
🧪 **Validation**: [How to test the pattern]"""

        # Create prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Analyze patterns in: {query}")
        ])
        
        # Create chain
        self.chain = self.prompt_template | self.llm | StrOutputParser()
    
    async def analyze_patterns(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze patterns with real LLM."""
        try:
            # Generate pattern analysis using LLM
            response = await self.chain.ainvoke({"query": query})
            
            # Extract patterns
            patterns = self._extract_patterns(response)
            
            # Calculate confidence based on pattern quality
            confidence = self._calculate_confidence(response, patterns)
            
            return {
                "response": response,
                "patterns": patterns,
                "confidence": confidence,
                "agent": "real_pattern_recognition",
                "model_used": self.model_name,
                "pattern_count": len(patterns),
                "context_used": context is not None
            }
            
        except Exception as e:
            return {
                "response": f"Error in pattern analysis: {str(e)}",
                "patterns": [],
                "confidence": 0.0,
                "agent": "real_pattern_recognition",
                "error": str(e)
            }
    
    def _extract_patterns(self, response: str) -> List[Dict[str, str]]:
        """Extract structured patterns from response."""
        patterns = []
        
        # Simple extraction based on formatting
        sections = response.split("🔍 **Pattern Analysis**:")
        
        for i, section in enumerate(sections[1:], 1):  # Skip first empty section
            pattern_parts = {}
            
            # Extract different components
            if "📊 **Mathematical Relationship**:" in section:
                parts = section.split("📊 **Mathematical Relationship**:")
                pattern_parts["analysis"] = parts[0].strip()
                remaining = parts[1] if len(parts) > 1 else ""
                
                if "🔬 **Physics Connection**:" in remaining:
                    math_physics = remaining.split("🔬 **Physics Connection**:")
                    pattern_parts["mathematical"] = math_physics[0].strip()
                    pattern_parts["physics"] = math_physics[1].strip() if len(math_physics) > 1 else ""
            
            if pattern_parts:
                pattern_parts["id"] = f"pattern_{i}"
                patterns.append(pattern_parts)
        
        return patterns
    
    def _calculate_confidence(self, response: str, patterns: List[Dict]) -> float:
        """Calculate confidence based on pattern quality."""
        confidence = 0.4  # Base confidence
        
        # Number of patterns identified
        confidence += min(len(patterns) * 0.1, 0.2)
        
        # Mathematical content
        math_indicators = ["equation", "formula", "∝", "=", "∫", "∂", "Σ", "correlation"]
        math_count = sum(1 for indicator in math_indicators if indicator.lower() in response.lower())
        confidence += min(math_count * 0.05, 0.2)
        
        # Physics terminology
        physics_terms = ["symmetry", "conservation", "scaling", "power law", "correlation", "trend"]
        physics_count = sum(1 for term in physics_terms if term.lower() in response.lower())
        confidence += min(physics_count * 0.05, 0.15)
        
        # Statistical analysis indicators
        stats_terms = ["significant", "correlation", "regression", "variance", "distribution"]
        stats_count = sum(1 for term in stats_terms if term.lower() in response.lower())
        confidence += min(stats_count * 0.05, 0.1)
        
        return min(confidence, 0.9)  # Cap at 90%
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information."""
        return {
            "name": "Real Pattern Recognition Agent",
            "type": "pattern_recognition",
            "model": self.model_name,
            "temperature": self.temperature,
            "capabilities": [
                "data_pattern_analysis", "mathematical_relationships", 
                "symmetry_detection", "scaling_laws", "emergent_phenomena"
            ],
            "description": "Advanced pattern recognition with LLM for discovering physics patterns"
        }