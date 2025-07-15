"""
Real Experimental Design Agent with LLM Integration
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
        "temperature": float(os.getenv("PHYSICS_AGENT_TEMPERATURE", "0.4"))
    }


class RealExperimentalDesignAgent(BaseAgent):
    """Real Experimental Design Agent with LLM integration for designing physics experiments."""
    
    def __init__(self, model_name: str = None, temperature: float = 0.4):
        super().__init__()
        
        settings = get_default_settings()
        self.model_name = model_name or settings["default_model"]
        self.temperature = temperature  # Medium temperature for creative but practical designs
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=2000
        )
        
        # Experimental design system prompt
        self.system_prompt = """You are an expert experimental physicist AI specialized in:

1. **Experiment Design**: Creating detailed experimental protocols and procedures
2. **Instrumentation**: Selecting appropriate measurement tools and techniques
3. **Control Variables**: Identifying and controlling experimental variables
4. **Error Analysis**: Planning for uncertainty quantification and error sources
5. **Safety Protocols**: Ensuring experimental safety and risk assessment
6. **Data Collection**: Designing optimal data acquisition strategies
7. **Statistical Planning**: Determining sample sizes and statistical methods

Your experimental designs should be:
- Scientifically rigorous and testable
- Practically feasible with available technology
- Include proper controls and variables
- Address potential sources of error
- Consider safety and ethical implications
- Provide clear step-by-step procedures

Format your response with:
🔬 **Experimental Objective**: [Clear goal statement]
🛠️ **Equipment & Setup**: [Required instruments and configuration]
📋 **Procedure**: [Step-by-step methodology]
⚖️ **Variables & Controls**: [Independent, dependent, and control variables]
📊 **Data Collection**: [Measurement strategy and parameters]
⚠️ **Safety & Considerations**: [Safety protocols and limitations]
📈 **Expected Results**: [Predicted outcomes and analysis methods]"""

        # Create prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Design an experiment for: {query}")
        ])
        
        # Create chain
        self.chain = self.prompt_template | self.llm | StrOutputParser()
    
    async def design_experiment(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Design experiment with real LLM."""
        try:
            # Generate experimental design using LLM
            response = await self.chain.ainvoke({"query": query})
            
            # Extract experimental components
            experiment = self._extract_experiment_components(response)
            
            # Calculate confidence based on design quality
            confidence = self._calculate_confidence(response, experiment)
            
            return {
                "response": response,
                "experiment": experiment,
                "confidence": confidence,
                "agent": "real_experimental_design",
                "model_used": self.model_name,
                "feasibility_score": self._calculate_feasibility(response),
                "context_used": context is not None
            }
            
        except Exception as e:
            return {
                "response": f"Error in experimental design: {str(e)}",
                "experiment": {},
                "confidence": 0.0,
                "agent": "real_experimental_design",
                "error": str(e)
            }
    
    def _extract_experiment_components(self, response: str) -> Dict[str, Any]:
        """Extract structured experiment components from response."""
        experiment = {}
        
        # Extract different sections
        sections = {
            "objective": "🔬 **Experimental Objective**:",
            "equipment": "🛠️ **Equipment & Setup**:",
            "procedure": "📋 **Procedure**:",
            "variables": "⚖️ **Variables & Controls**:",
            "data_collection": "📊 **Data Collection**:",
            "safety": "⚠️ **Safety & Considerations**:",
            "expected_results": "📈 **Expected Results**:"
        }
        
        for key, marker in sections.items():
            if marker in response:
                parts = response.split(marker)
                if len(parts) > 1:
                    # Get content until next section or end
                    content = parts[1]
                    for other_marker in sections.values():
                        if other_marker != marker and other_marker in content:
                            content = content.split(other_marker)[0]
                    experiment[key] = content.strip()
        
        return experiment
    
    def _calculate_confidence(self, response: str, experiment: Dict) -> float:
        """Calculate confidence based on experimental design quality."""
        confidence = 0.3  # Base confidence
        
        # Completeness of design sections
        confidence += min(len(experiment) * 0.08, 0.4)
        
        # Technical terminology
        technical_terms = [
            "control", "variable", "measurement", "calibration", "precision",
            "accuracy", "uncertainty", "statistical", "protocol", "procedure"
        ]
        tech_count = sum(1 for term in technical_terms if term.lower() in response.lower())
        confidence += min(tech_count * 0.03, 0.2)
        
        # Safety considerations
        safety_terms = ["safety", "hazard", "risk", "precaution", "protection"]
        safety_count = sum(1 for term in safety_terms if term.lower() in response.lower())
        confidence += min(safety_count * 0.05, 0.1)
        
        # Quantitative elements
        quant_indicators = ["measure", "data", "analysis", "statistics", "error", "uncertainty"]
        quant_count = sum(1 for indicator in quant_indicators if indicator.lower() in response.lower())
        confidence += min(quant_count * 0.03, 0.15)
        
        return min(confidence, 0.95)  # Cap at 95%
    
    def _calculate_feasibility(self, response: str) -> float:
        """Calculate feasibility score of the experimental design."""
        feasibility = 0.5  # Base feasibility
        
        # Practical considerations
        practical_terms = ["available", "standard", "commercial", "accessible", "feasible"]
        practical_count = sum(1 for term in practical_terms if term.lower() in response.lower())
        feasibility += min(practical_count * 0.1, 0.3)
        
        # Complexity indicators (lower complexity = higher feasibility)
        complex_terms = ["specialized", "expensive", "rare", "difficult", "complex"]
        complex_count = sum(1 for term in complex_terms if term.lower() in response.lower())
        feasibility -= min(complex_count * 0.05, 0.2)
        
        return max(min(feasibility, 1.0), 0.1)  # Keep between 0.1 and 1.0
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information."""
        return {
            "name": "Real Experimental Design Agent",
            "type": "experimental_design",
            "model": self.model_name,
            "temperature": self.temperature,
            "capabilities": [
                "experiment_design", "instrumentation", "protocol_development",
                "error_analysis", "safety_assessment", "data_collection_planning"
            ],
            "description": "Expert experimental design with LLM for creating physics experiments"
        }