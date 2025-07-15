"""
Real Agent Manager - Coordinates actual LLM-powered agents
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import time
import logging

from .real_physics_expert import RealPhysicsExpertAgent
from .real_hypothesis_generator import RealHypothesisGeneratorAgent
from .types import TaskRequest, TaskResult, ConsensusResult, TaskType
from ..config.settings import get_settings


@dataclass
class AgentResponse:
    """Standardized agent response format."""
    agent_name: str
    response: str
    confidence: float
    execution_time: float
    metadata: Dict[str, Any]
    success: bool = True
    error: Optional[str] = None


class RealAgentManager:
    """Manages real LLM-powered agents for physics analysis."""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)
        
        # Initialize real agents
        self.agents = {
            "physics_expert": RealPhysicsExpertAgent(
                temperature=0.1  # Low temperature for accuracy
            ),
            "hypothesis_generator": RealHypothesisGeneratorAgent(
                temperature=0.7  # High temperature for creativity
            ),
            "mathematical_analyst": RealPhysicsExpertAgent(
                temperature=0.05  # Very low for mathematical precision
            )
        }
        
        self.agent_capabilities = {
            "physics_expert": [
                "quantum_mechanics", "thermodynamics", "electromagnetism",
                "mechanics", "relativity", "general_physics"
            ],
            "hypothesis_generator": [
                "creative_thinking", "research_gaps", "experimental_design",
                "interdisciplinary_connections"
            ],
            "mathematical_analyst": [
                "mathematical_modeling", "equation_derivation", 
                "numerical_analysis", "statistical_physics"
            ]
        }
    
    async def process_query_with_real_agents(
        self, 
        query: str, 
        selected_agents: List[str] = None,
        parallel: bool = True
    ) -> List[AgentResponse]:
        """Process query using real LLM-powered agents."""
        
        if selected_agents is None:
            selected_agents = ["physics_expert", "hypothesis_generator"]
        
        # Filter to available agents
        available_agents = [
            agent for agent in selected_agents 
            if agent in self.agents
        ]
        
        if not available_agents:
            raise ValueError(f"No available agents from: {selected_agents}")
        
        self.logger.info(f"Processing query with agents: {available_agents}")
        
        if parallel:
            # Process in parallel
            tasks = [
                self._process_with_agent(agent_name, query)
                for agent_name in available_agents
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Process sequentially
            responses = []
            for agent_name in available_agents:
                response = await self._process_with_agent(agent_name, query)
                responses.append(response)
        
        # Handle any exceptions
        valid_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                self.logger.error(f"Agent {available_agents[i]} failed: {response}")
                valid_responses.append(AgentResponse(
                    agent_name=available_agents[i],
                    response=f"Agent failed: {str(response)}",
                    confidence=0.0,
                    execution_time=0.0,
                    metadata={},
                    success=False,
                    error=str(response)
                ))
            else:
                valid_responses.append(response)
        
        return valid_responses
    
    async def _process_with_agent(self, agent_name: str, query: str) -> AgentResponse:
        """Process query with a specific agent."""
        start_time = time.time()
        
        try:
            agent = self.agents[agent_name]
            
            if agent_name == "physics_expert":
                result = await agent.analyze_physics_query(query)
            elif agent_name == "hypothesis_generator":
                result = await agent.generate_hypotheses(query)
            elif agent_name == "mathematical_analyst":
                # Use physics expert with mathematical focus
                math_query = f"Provide mathematical analysis for: {query}"
                result = await agent.analyze_physics_query(math_query)
            else:
                raise ValueError(f"Unknown agent: {agent_name}")
            
            execution_time = time.time() - start_time
            
            return AgentResponse(
                agent_name=agent_name,
                response=result.get("response", ""),
                confidence=result.get("confidence", 0.0),
                execution_time=execution_time,
                metadata=result,
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Error processing with {agent_name}: {e}")
            
            return AgentResponse(
                agent_name=agent_name,
                response=f"Error: {str(e)}",
                confidence=0.0,
                execution_time=execution_time,
                metadata={},
                success=False,
                error=str(e)
            )
    
    def build_consensus(self, responses: List[AgentResponse]) -> Dict[str, Any]:
        """Build consensus from multiple agent responses."""
        if not responses:
            return {
                "final_result": "No agent responses available",
                "confidence": 0.0,
                "agreement_level": 0.0,
                "participating_agents": [],
                "method_used": "none"
            }
        
        # Filter successful responses
        successful_responses = [r for r in responses if r.success]
        
        if not successful_responses:
            return {
                "final_result": "All agents failed to process query",
                "confidence": 0.0,
                "agreement_level": 0.0,
                "participating_agents": [r.agent_name for r in responses],
                "method_used": "error_handling"
            }
        
        # Calculate weighted consensus
        total_weight = sum(r.confidence for r in successful_responses)
        if total_weight == 0:
            weights = [1.0 / len(successful_responses)] * len(successful_responses)
        else:
            weights = [r.confidence / total_weight for r in successful_responses]
        
        # Build comprehensive response
        final_result = self._build_comprehensive_response(successful_responses)
        
        # Calculate overall confidence
        overall_confidence = sum(
            r.confidence * w for r, w in zip(successful_responses, weights)
        )
        
        # Calculate agreement level (simplified)
        agreement_level = min(overall_confidence + 0.1, 0.95)
        
        return {
            "final_result": final_result,
            "confidence": overall_confidence,
            "agreement_level": agreement_level,
            "participating_agents": [r.agent_name for r in successful_responses],
            "method_used": "weighted_consensus",
            "individual_results": [
                {
                    "agent": r.agent_name,
                    "response": r.response,
                    "confidence": r.confidence,
                    "execution_time": r.execution_time
                }
                for r in successful_responses
            ]
        }
    
    def _build_comprehensive_response(self, responses: List[AgentResponse]) -> str:
        """Build a comprehensive response from multiple agents."""
        result_parts = ["Multi-Agent Physics Analysis:\n"]
        
        for response in responses:
            if response.agent_name == "physics_expert":
                result_parts.append(f"🔬 **Physics Expert Analysis:**\n{response.response}\n")
            elif response.agent_name == "hypothesis_generator":
                result_parts.append(f"💡 **Creative Hypotheses:**\n{response.response}\n")
            elif response.agent_name == "mathematical_analyst":
                result_parts.append(f"📊 **Mathematical Analysis:**\n{response.response}\n")
            else:
                result_parts.append(f"🤖 **{response.agent_name.title()}:**\n{response.response}\n")
        
        # Add consensus summary
        result_parts.append(
            "🤝 **Consensus Summary:**\n"
            "The multi-agent system has analyzed your query from multiple perspectives, "
            "combining rigorous physics analysis with creative insights and mathematical rigor. "
            "This comprehensive approach ensures both accuracy and innovative thinking."
        )
        
        return "\n".join(result_parts)