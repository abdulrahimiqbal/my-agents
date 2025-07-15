"""
Real Hypothesis Generator Agent with LLM Integration
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
        "temperature": float(os.getenv("HYPOTHESIS_AGENT_TEMPERATURE", "0.7"))
    }


class RealHypothesisGeneratorAgent(BaseAgent):
    """Real Hypothesis Generator Agent with creative LLM integration."""
    
    def __init__(self, model_name: str = None, temperature: float = 0.7):
        super().__init__()
        
        settings = get_default_settings()
        self.model_name = model_name or settings["default_model"]
        self.temperature = temperature  # Higher temperature for creativity
        
        # Initialize LLM with higher temperature for creativity
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=1500
        )
        
        # Creative hypothesis generation system prompt
        self.system_prompt = """You are a creative physics hypothesis generator AI that specializes in:

1. **Novel Hypothesis Generation**: Creating innovative, testable hypotheses
2. **Cross-Domain Connections**: Finding unexpected links between physics domains
3. **Experimental Design**: Suggesting ways to test hypotheses
4. **Research Gap Identification**: Spotting unexplored areas
5. **Creative Problem Solving**: Approaching physics problems from new angles

Your approach should be:
- Creative but scientifically grounded
- Generate 2-3 distinct hypotheses per query
- Include testability criteria
- Consider interdisciplinary connections
- Acknowledge speculative nature when appropriate
- Suggest experimental approaches

Format your response with:
🔬 **Hypothesis 1**: [Creative hypothesis]
🧪 **Testability**: [How to test it]
🔗 **Connections**: [Links to other domains]

🔬 **Hypothesis 2**: [Alternative hypothesis]
🧪 **Testability**: [How to test it]
🔗 **Connections**: [Links to other domains]

💡 **Research Directions**: [Future investigation paths]"""

        # Create prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Generate creative hypotheses for: {query}")
        ])
        
        # Create chain
        self.chain = self.prompt_template | self.llm | StrOutputParser()
    
    async def generate_hypotheses(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate creative hypotheses with real LLM."""
        try:
            # Generate hypotheses using LLM
            response = await self.chain.ainvoke({"query": query})
            
            # Extract individual hypotheses
            hypotheses = self._extract_hypotheses(response)
            
            # Calculate confidence based on creativity and feasibility
            confidence = self._calculate_confidence(response, hypotheses)
            
            return {
                "response": response,
                "hypotheses": hypotheses,
                "confidence": confidence,
                "agent": "real_hypothesis_generator",
                "model_used": self.model_name,
                "creativity_score": self._calculate_creativity_score(response),
                "context_used": context is not None
            }
            
        except Exception as e:
            return {
                "response": f"Error in hypothesis generation: {str(e)}",
                "hypotheses": [],
                "confidence": 0.0,
                "agent": "real_hypothesis_generator",
                "error": str(e)
            }
    
    def _extract_hypotheses(self, response: str) -> List[Dict[str, str]]:
        """Extract structured hypotheses from response."""
        hypotheses = []
        
        # Simple extraction based on formatting
        sections = response.split("🔬 **Hypothesis")
        
        for i, section in enumerate(sections[1:], 1):  # Skip first empty section
            hypothesis_text = section.split("🧪 **Testability**:")
            testability_text = ""
            connections_text = ""
            
            if len(hypothesis_text) > 1:
                hypothesis = hypothesis_text[0].strip()
                remaining = hypothesis_text[1]
                
                if "🔗 **Connections**:" in remaining:
                    parts = remaining.split("🔗 **Connections**:")
                    testability_text = parts[0].strip()
                    connections_text = parts[1].strip() if len(parts) > 1 else ""
                else:
                    testability_text = remaining.strip()
                
                hypotheses.append({
                    "id": f"hypothesis_{i}",
                    "hypothesis": hypothesis,
                    "testability": testability_text,
                    "connections": connections_text
                })
        
        return hypotheses
    
    def _calculate_confidence(self, response: str, hypotheses: List[Dict]) -> float:
        """Calculate confidence based on hypothesis quality."""
        confidence = 0.4  # Base confidence for creative work
        
        # Number of hypotheses generated
        confidence += min(len(hypotheses) * 0.15, 0.3)
        
        # Presence of testability criteria
        if "testability" in response.lower() or "test" in response.lower():
            confidence += 0.1
        
        # Interdisciplinary connections
        if "connection" in response.lower() or "domain" in response.lower():
            confidence += 0.1
        
        # Creative indicators
        creative_words = ["novel", "innovative", "unexpected", "creative", "unique"]
        creative_count = sum(1 for word in creative_words if word in response.lower())
        confidence += min(creative_count * 0.05, 0.15)
        
        return min(confidence, 0.9)  # Cap at 90% for creative work
    
    def _calculate_creativity_score(self, response: str) -> float:
        """Calculate creativity score of the response."""
        creativity = 0.0
        
        # Unusual word combinations
        unusual_phrases = [
            "quantum", "emergent", "nonlinear", "topological", 
            "holographic", "fractal", "chaotic", "entangled"
        ]
        creativity += sum(0.1 for phrase in unusual_phrases if phrase in response.lower())
        
        # Cross-domain references
        domains = [
            "biology", "chemistry", "computer", "mathematics", 
            "philosophy", "engineering", "neuroscience"
        ]
        creativity += sum(0.15 for domain in domains if domain in response.lower())
        
        return min(creativity, 1.0)
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information."""
        return {
            "name": "Real Hypothesis Generator Agent",
            "type": "hypothesis_generator",
            "model": self.model_name,
            "temperature": self.temperature,
            "capabilities": [
                "creative_thinking", "research_gaps", "experimental_design",
                "interdisciplinary_connections", "hypothesis_generation"
            ],
            "description": "Creative hypothesis generator with high-temperature LLM for innovative thinking"
        }