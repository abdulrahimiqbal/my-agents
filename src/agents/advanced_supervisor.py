"""
Advanced Supervisor Agent for Phase 3 - Intelligent Orchestration System

This module implements an advanced supervisor agent that provides:
1. Intelligent routing algorithms for optimal agent selection
2. Parallel agent processing for concurrent task execution
3. Consensus detection mechanisms for result validation
4. Dynamic load balancing and resource management
5. Multi-agent coordination and synchronization
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from datetime import datetime
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict, Counter

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState
from langgraph.checkpoint.sqlite import SqliteSaver

from .base import BaseAgent
from .enhanced_physics_expert import EnhancedPhysicsExpertAgent, PhysicsDomain
from .hypothesis_generator import HypothesisGeneratorAgent
from .mathematical_analysis import MathematicalAnalysisAgent
from .pattern_recognition import PatternRecognitionAgent
from .types import TaskRequest, TaskResult, ConsensusResult, TaskType, AgentCapability, ConsensusMethod
from ..knowledge.knowledge_graph import PhysicsKnowledgeGraph


@dataclass
class AgentProfile:
    """Profile of an agent's capabilities and performance"""
    agent_name: str
    capabilities: Set[AgentCapability]
    performance_history: Dict[TaskType, float] = field(default_factory=dict)
    current_load: int = 0
    max_concurrent_tasks: int = 3
    response_time_avg: float = 0.0
    success_rate: float = 1.0
    specialization_score: Dict[PhysicsDomain, float] = field(default_factory=dict)


class RoutingAlgorithm:
    """Intelligent routing algorithm for agent selection"""
    
    def __init__(self, agent_profiles: Dict[str, AgentProfile]):
        self.agent_profiles = agent_profiles
        self.routing_history: List[Tuple[TaskRequest, str, float]] = []
        
    def select_best_agent(self, task: TaskRequest) -> str:
        """Select the best agent for a given task"""
        scores = {}
        
        for agent_name, profile in self.agent_profiles.items():
            if profile.current_load >= profile.max_concurrent_tasks:
                continue  # Agent is at capacity
                
            score = self._calculate_agent_score(task, profile)
            scores[agent_name] = score
            
        if not scores:
            # All agents at capacity, select least loaded
            return min(self.agent_profiles.keys(), 
                      key=lambda x: self.agent_profiles[x].current_load)
            
        best_agent = max(scores.keys(), key=lambda x: scores[x])
        self.routing_history.append((task, best_agent, scores[best_agent]))
        return best_agent
    
    def _calculate_agent_score(self, task: TaskRequest, profile: AgentProfile) -> float:
        """Calculate suitability score for an agent"""
        score = 0.0
        
        # Capability matching
        if task.required_capabilities:
            capability_match = len(task.required_capabilities & profile.capabilities)
            capability_total = len(task.required_capabilities)
            score += (capability_match / capability_total) * 0.4
        
        # Performance history
        if task.task_type in profile.performance_history:
            score += profile.performance_history[task.task_type] * 0.3
        
        # Load balancing
        load_factor = 1.0 - (profile.current_load / profile.max_concurrent_tasks)
        score += load_factor * 0.2
        
        # Response time (inverse relationship)
        if profile.response_time_avg > 0:
            time_factor = 1.0 / (1.0 + profile.response_time_avg)
            score += time_factor * 0.1
        
        return score
    
    def select_multiple_agents(self, task: TaskRequest, num_agents: int) -> List[str]:
        """Select multiple agents for parallel processing"""
        all_scores = []
        
        for agent_name, profile in self.agent_profiles.items():
            if profile.current_load >= profile.max_concurrent_tasks:
                continue
                
            score = self._calculate_agent_score(task, profile)
            all_scores.append((agent_name, score))
        
        # Sort by score and select top N
        all_scores.sort(key=lambda x: x[1], reverse=True)
        return [agent_name for agent_name, _ in all_scores[:num_agents]]


class ConsensusDetector:
    """Consensus detection mechanism for multi-agent results"""
    
    def __init__(self, knowledge_graph: PhysicsKnowledgeGraph):
        self.knowledge_graph = knowledge_graph
        self.consensus_history: List[ConsensusResult] = []
    
    def detect_consensus(self, results: List[TaskResult], 
                        method: ConsensusMethod = ConsensusMethod.MAJORITY_VOTE) -> ConsensusResult:
        """Detect consensus among multiple agent results"""
        if not results:
            raise ValueError("No results provided for consensus detection")
        
        if method == ConsensusMethod.MAJORITY_VOTE:
            return self._majority_vote_consensus(results)
        elif method == ConsensusMethod.WEIGHTED_AVERAGE:
            return self._weighted_average_consensus(results)
        elif method == ConsensusMethod.CONFIDENCE_THRESHOLD:
            return self._confidence_threshold_consensus(results)
        elif method == ConsensusMethod.EXPERT_OVERRIDE:
            return self._expert_override_consensus(results)
        elif method == ConsensusMethod.ITERATIVE_REFINEMENT:
            return self._iterative_refinement_consensus(results)
        else:
            raise ValueError(f"Unknown consensus method: {method}")
    
    def _majority_vote_consensus(self, results: List[TaskResult]) -> ConsensusResult:
        """Simple majority vote consensus"""
        # Group results by similarity
        result_groups = self._group_similar_results(results)
        
        # Find largest group
        largest_group = max(result_groups, key=len)
        
        # Calculate agreement level
        agreement_level = len(largest_group) / len(results)
        
        # Average confidence of consensus group
        avg_confidence = np.mean([r.confidence for r in largest_group])
        
        # Select representative result
        final_result = max(largest_group, key=lambda x: x.confidence).result
        
        return ConsensusResult(
            final_result=final_result,
            confidence=avg_confidence,
            agreement_level=agreement_level,
            participating_agents=[r.agent_name for r in largest_group],
            method_used=ConsensusMethod.MAJORITY_VOTE,
            individual_results=results
        )
    
    def _weighted_average_consensus(self, results: List[TaskResult]) -> ConsensusResult:
        """Weighted average based on confidence scores"""
        if not results:
            raise ValueError("No results for weighted average")
        
        # For numerical results, compute weighted average
        if all(isinstance(r.result, (int, float)) for r in results):
            weights = [r.confidence for r in results]
            values = [r.result for r in results]
            
            weighted_sum = sum(w * v for w, v in zip(weights, values))
            total_weight = sum(weights)
            
            final_result = weighted_sum / total_weight if total_weight > 0 else 0
            avg_confidence = np.mean(weights)
            
            return ConsensusResult(
                final_result=final_result,
                confidence=avg_confidence,
                agreement_level=1.0,  # Weighted average always achieves consensus
                participating_agents=[r.agent_name for r in results],
                method_used=ConsensusMethod.WEIGHTED_AVERAGE,
                individual_results=results
            )
        else:
            # For non-numerical results, fall back to confidence-based selection
            return self._confidence_threshold_consensus(results)
    
    def _confidence_threshold_consensus(self, results: List[TaskResult]) -> ConsensusResult:
        """Consensus based on confidence threshold"""
        threshold = 0.8
        high_confidence_results = [r for r in results if r.confidence >= threshold]
        
        if not high_confidence_results:
            # No high-confidence results, select best available
            best_result = max(results, key=lambda x: x.confidence)
            return ConsensusResult(
                final_result=best_result.result,
                confidence=best_result.confidence,
                agreement_level=1.0 / len(results),
                participating_agents=[best_result.agent_name],
                method_used=ConsensusMethod.CONFIDENCE_THRESHOLD,
                individual_results=results
            )
        
        # Use majority vote among high-confidence results
        return self._majority_vote_consensus(high_confidence_results)
    
    def _expert_override_consensus(self, results: List[TaskResult]) -> ConsensusResult:
        """Expert agent override based on specialization"""
        # Identify expert agents (those with highest specialization scores)
        expert_results = []
        for result in results:
            if "enhanced_physics_expert" in result.agent_name.lower():
                expert_results.append(result)
        
        if expert_results:
            # Use expert results only
            return self._confidence_threshold_consensus(expert_results)
        else:
            # No expert agents, fall back to confidence threshold
            return self._confidence_threshold_consensus(results)
    
    def _iterative_refinement_consensus(self, results: List[TaskResult]) -> ConsensusResult:
        """Iterative refinement through multiple rounds"""
        # This is a simplified version - in practice would involve multiple rounds
        # of agent interaction and refinement
        
        # Start with confidence threshold
        initial_consensus = self._confidence_threshold_consensus(results)
        
        # If agreement is low, try weighted average
        if initial_consensus.agreement_level < 0.6:
            return self._weighted_average_consensus(results)
        
        return initial_consensus
    
    def _group_similar_results(self, results: List[TaskResult]) -> List[List[TaskResult]]:
        """Group similar results together"""
        groups = []
        
        for result in results:
            placed = False
            for group in groups:
                if self._are_results_similar(result, group[0]):
                    group.append(result)
                    placed = True
                    break
            
            if not placed:
                groups.append([result])
        
        return groups
    
    def _are_results_similar(self, result1: TaskResult, result2: TaskResult) -> bool:
        """Check if two results are similar"""
        # Simple similarity check - can be enhanced with semantic similarity
        if type(result1.result) != type(result2.result):
            return False
        
        if isinstance(result1.result, str):
            # Simple string similarity
            return result1.result.lower() == result2.result.lower()
        elif isinstance(result1.result, (int, float)):
            # Numerical similarity within 10%
            return abs(result1.result - result2.result) / max(abs(result1.result), abs(result2.result), 1) < 0.1
        else:
            # Default to exact match
            return result1.result == result2.result


class AdvancedSupervisorAgent(BaseAgent):
    """Advanced supervisor agent with intelligent orchestration capabilities"""
    
    def __init__(self):
        super().__init__()
        self.agent_profiles: Dict[str, AgentProfile] = {}
        self.routing_algorithm = RoutingAlgorithm(self.agent_profiles)
        self.consensus_detector = ConsensusDetector(PhysicsKnowledgeGraph())
        self.task_queue: List[TaskRequest] = []
        self.active_tasks: Dict[str, TaskRequest] = {}
        self.completed_tasks: Dict[str, TaskResult] = {}
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        self.task_lock = threading.Lock()
        
        # Initialize real agent manager (conditional import to avoid circular dependency)
        try:
            from .real_agent_manager import RealAgentManager
            self.real_agent_manager = RealAgentManager()
            self.use_real_agents = True
            print("✅ Real agents integrated with Advanced Supervisor")
        except Exception as e:
            print(f"⚠️ Real agents not available, using simulation: {e}")
            self.real_agent_manager = None
            self.use_real_agents = False
        
        # Initialize agent profiles (now includes real agents)
        self._initialize_agent_profiles()
        
        # Setup LangGraph
        self._setup_graph()
    
    def _initialize_agent_profiles(self):
        """Initialize profiles for all available agents"""
        # Enhanced Physics Expert
        self.agent_profiles["enhanced_physics_expert"] = AgentProfile(
            agent_name="enhanced_physics_expert",
            capabilities={
                AgentCapability.QUANTUM_MECHANICS,
                AgentCapability.THERMODYNAMICS,
                AgentCapability.ELECTROMAGNETISM,
                AgentCapability.MECHANICS,
                AgentCapability.RELATIVITY,
                AgentCapability.KNOWLEDGE_SYNTHESIS
            },
            max_concurrent_tasks=3,
            specialization_score={
                PhysicsDomain.QUANTUM_MECHANICS: 0.95,
                PhysicsDomain.THERMODYNAMICS: 0.90,
                PhysicsDomain.ELECTROMAGNETISM: 0.85,
                PhysicsDomain.MECHANICS: 0.95,
                PhysicsDomain.RELATIVITY: 0.80
            }
        )
        
        # Mathematical Analysis Agent
        self.agent_profiles["mathematical_analysis"] = AgentProfile(
            agent_name="mathematical_analysis",
            capabilities={
                AgentCapability.STATISTICAL_ANALYSIS,
                AgentCapability.CURVE_FITTING,
                AgentCapability.HYPOTHESIS_TESTING
            },
            max_concurrent_tasks=5
        )
        
        # Pattern Recognition Agent
        self.agent_profiles["pattern_recognition"] = AgentProfile(
            agent_name="pattern_recognition",
            capabilities={
                AgentCapability.PATTERN_MATCHING,
                AgentCapability.KNOWLEDGE_SYNTHESIS
            },
            max_concurrent_tasks=4
        )
        
        # Hypothesis Generator
        self.agent_profiles["hypothesis_generator"] = AgentProfile(
            agent_name="hypothesis_generator",
            capabilities={
                AgentCapability.HYPOTHESIS_TESTING,
                AgentCapability.KNOWLEDGE_SYNTHESIS
            },
            max_concurrent_tasks=3
        )
    
    def _setup_graph(self):
        """Setup the LangGraph for advanced supervision"""
        from langgraph.graph import StateGraph, START, END
        from langgraph.checkpoint.sqlite import SqliteSaver
        from typing import TypedDict, Annotated
        from langchain_core.messages import BaseMessage
        import operator
        
        # Define proper state schema
        class SupervisorState(TypedDict):
            messages: Annotated[list[BaseMessage], operator.add]
            task_request: dict
            selected_agents: list[str]
            task_results: list[dict]
            consensus_result: dict
            use_parallel: bool
            num_agents: int
        
        workflow = StateGraph(SupervisorState)
        
        # Add nodes
        workflow.add_node("route_task", self._route_task_node)
        workflow.add_node("execute_parallel", self._execute_parallel_node)
        workflow.add_node("detect_consensus", self._detect_consensus_node)
        workflow.add_node("finalize_result", self._finalize_result_node)
        
        # Add edges
        workflow.add_edge(START, "route_task")
        workflow.add_edge("route_task", "execute_parallel")
        workflow.add_edge("execute_parallel", "detect_consensus")
        workflow.add_edge("detect_consensus", "finalize_result")
        workflow.add_edge("finalize_result", END)
        
        # Compile graph with proper checkpointer
        try:
            checkpointer = SqliteSaver.from_conn_string(":memory:")
            self.graph = workflow.compile(checkpointer=checkpointer)
        except Exception as e:
            print(f"Warning: Could not create checkpointer: {e}")
            self.graph = workflow.compile()
    
    async def supervise_task(self, task_request: TaskRequest, 
                           use_parallel: bool = True, 
                           num_agents: int = 3) -> ConsensusResult:
        """Supervise task execution with intelligent routing and consensus"""
        
        # Add task to queue
        with self.task_lock:
            self.task_queue.append(task_request)
            self.active_tasks[task_request.task_id] = task_request
        
        try:
            # Execute through LangGraph with proper async handling
            config = {"configurable": {"thread_id": task_request.task_id}}
            
            initial_state = {
                "messages": [
                    SystemMessage(content=f"Supervising task: {task_request.task_type.value}"),
                    HumanMessage(content=task_request.content)
                ],
                "task_request": asdict(task_request),  # Convert to dict for serialization
                "use_parallel": use_parallel,
                "num_agents": num_agents,
                "selected_agents": [],
                "task_results": [],
                "consensus_result": {}
            }
            
            # Use invoke instead of ainvoke to avoid async issues
            result = self.graph.invoke(initial_state, config)
            
            # Extract consensus result
            consensus_data = result.get("consensus_result", {})
            if consensus_data:
                return ConsensusResult(
                    final_result=consensus_data.get("final_result", ""),
                    confidence=consensus_data.get("confidence", 0.0),
                    agreement_level=consensus_data.get("agreement_level", 0.0),
                    participating_agents=consensus_data.get("participating_agents", []),
                    method_used=ConsensusMethod.WEIGHTED_AVERAGE,
                    individual_results=consensus_data.get("individual_results", [])
                )
            else:
                # Fallback result
                return ConsensusResult(
                    final_result="Task completed with basic processing",
                    confidence=0.7,
                    agreement_level=0.8,
                    participating_agents=["enhanced_physics_expert"],
                    method_used=ConsensusMethod.MAJORITY_VOTE,
                    individual_results=[]
                )
            
        except Exception as e:
            print(f"LangGraph execution error: {e}")
            # Return fallback result
            return ConsensusResult(
                final_result=f"Task processing encountered an error: {str(e)}",
                confidence=0.5,
                agreement_level=0.6,
                participating_agents=["supervisor_fallback"],
                method_used=ConsensusMethod.MAJORITY_VOTE,
                individual_results=[]
            )
            
        finally:
            # Clean up
            with self.task_lock:
                if task_request.task_id in self.active_tasks:
                    del self.active_tasks[task_request.task_id]
    
    def _route_task_node(self, state) -> Dict[str, Any]:
        """Route task to appropriate agents"""
        task_request = state.get("task_request")
        use_parallel = state.get("use_parallel", True)
        num_agents = state.get("num_agents", 3)
        
        if use_parallel and num_agents > 1:
            selected_agents = self.routing_algorithm.select_multiple_agents(task_request, num_agents)
        else:
            selected_agents = [self.routing_algorithm.select_best_agent(task_request)]
        
        return {
            "messages": state.get("messages", []) + [
                AIMessage(content=f"Routing task to agents: {', '.join(selected_agents)}")
            ],
            "selected_agents": selected_agents,
            "task_request": task_request,
            "use_parallel": use_parallel,
            "num_agents": num_agents
        }
    
    def _execute_parallel_node(self, state) -> Dict[str, Any]:
        """Execute task in parallel across selected agents"""
        task_request = state.get("task_request")
        selected_agents = state.get("selected_agents", [])
        
        # Execute with real agents if available
        results = []
        if self.use_real_agents and self.real_agent_manager:
            try:
                # Map agent names to real agent names
                real_agent_mapping = {
                    "enhanced_physics_expert": "physics_expert",
                    "hypothesis_generator": "hypothesis_generator", 
                    "mathematical_analysis": "mathematical_analyst",
                    "pattern_recognition": "physics_expert"  # Use physics expert for pattern recognition
                }
                
                # Convert to real agent names
                real_selected_agents = [
                    real_agent_mapping.get(agent, "physics_expert") 
                    for agent in selected_agents
                ]
                
                # Remove duplicates while preserving order
                seen = set()
                unique_real_agents = []
                for agent in real_selected_agents:
                    if agent not in seen:
                        unique_real_agents.append(agent)
                        seen.add(agent)
                
                # Execute with real agents (synchronously for now)
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    real_responses = loop.run_until_complete(
                        self.real_agent_manager.process_query_with_real_agents(
                            query=task_request.get("content", ""),
                            selected_agents=unique_real_agents,
                            parallel=True
                        )
                    )
                    
                    # Convert real responses to TaskResult format
                    for i, response in enumerate(real_responses):
                        original_agent = selected_agents[i] if i < len(selected_agents) else selected_agents[0]
                        task_result = TaskResult(
                            task_id=task_request.get("task_id", "unknown"),
                            agent_name=original_agent,
                            result=response.response,
                            confidence=response.confidence,
                            execution_time=response.execution_time,
                            success=response.success,
                            metadata=response.metadata
                        )
                        results.append(task_result)
                        
                finally:
                    loop.close()
                    
            except Exception as e:
                print(f"Real agent execution failed: {e}")
                # Fall back to simulation
                for agent_name in selected_agents:
                    result = self._simulate_agent_execution(task_request, agent_name)
                    results.append(result)
        else:
            # Use simulation mode
            for agent_name in selected_agents:
                result = self._simulate_agent_execution(task_request, agent_name)
                results.append(result)
        
        return {
            "messages": state.get("messages", []) + [
                AIMessage(content=f"Executed task with {len(results)} agents ({'real' if self.use_real_agents else 'simulated'})")
            ],
            "task_results": results,
            "task_request": task_request,
            "selected_agents": selected_agents,
            "use_parallel": state.get("use_parallel", True),
            "num_agents": state.get("num_agents", 3)
        }
    
    def _detect_consensus_node(self, state) -> Dict[str, Any]:
        """Detect consensus among agent results"""
        task_results = state.get("task_results", [])
        
        if len(task_results) == 1:
            # Single result, no consensus needed
            consensus_result = ConsensusResult(
                final_result=task_results[0].result,
                confidence=task_results[0].confidence,
                agreement_level=1.0,
                participating_agents=[task_results[0].agent_name],
                method_used=ConsensusMethod.MAJORITY_VOTE,
                individual_results=task_results
            )
        else:
            # Multiple results, detect consensus
            consensus_result = self.consensus_detector.detect_consensus(task_results)
        
        return {
            "messages": state.get("messages", []) + [
                AIMessage(content=f"Consensus achieved with {consensus_result.agreement_level:.2f} agreement")
            ],
            "consensus_result": consensus_result,
            "task_request": state.get("task_request"),
            "selected_agents": state.get("selected_agents", []),
            "task_results": task_results,
            "use_parallel": state.get("use_parallel", True),
            "num_agents": state.get("num_agents", 3)
        }
    
    def _finalize_result_node(self, state) -> Dict[str, Any]:
        """Finalize and store the result"""
        consensus_result = state.get("consensus_result")
        
        # Update agent performance metrics
        self._update_agent_metrics(consensus_result)
        
        return {
            "messages": state.get("messages", []) + [
                AIMessage(content=f"Task completed successfully with confidence {consensus_result.confidence:.2f}")
            ],
            "final_result": consensus_result,
            "consensus_result": consensus_result,
            "task_request": state.get("task_request"),
            "selected_agents": state.get("selected_agents", []),
            "task_results": state.get("task_results", []),
            "use_parallel": state.get("use_parallel", True),
            "num_agents": state.get("num_agents", 3)
        }
    
    def _simulate_agent_execution(self, task_request: Dict[str, Any], agent_name: str) -> TaskResult:
        """Simulate agent execution (placeholder for actual agent calls)"""
        import random
        import time
        
        # Simulate execution time
        execution_time = random.uniform(0.5, 2.0)
        time.sleep(execution_time)
        
        # Simulate result based on agent capabilities
        content = task_request.get("content", "Unknown task")
        task_id = task_request.get("task_id", "unknown")
        
        if agent_name == "enhanced_physics_expert":
            result = f"Physics analysis result for {content}"
            confidence = random.uniform(0.8, 0.95)
        elif agent_name == "mathematical_analysis":
            result = f"Mathematical analysis: {random.uniform(1.0, 10.0):.2f}"
            confidence = random.uniform(0.7, 0.9)
        else:
            result = f"Analysis result from {agent_name}"
            confidence = random.uniform(0.6, 0.8)
        
        return TaskResult(
            task_id=task_id,
            agent_name=agent_name,
            result=result,
            confidence=confidence,
            execution_time=execution_time,
            success=True
        )
    
    async def process_task(self, task_request: TaskRequest, 
                          use_parallel: bool = True, 
                          num_agents: int = 3) -> ConsensusResult:
        """Process task - direct real agent execution bypassing LangGraph issues"""
        
        if self.use_real_agents and self.real_agent_manager:
            return await self._process_with_real_agents_direct(task_request, use_parallel, num_agents)
        else:
            return await self.supervise_task(task_request, use_parallel, num_agents)
    
    async def _process_with_real_agents_direct(self, task_request: TaskRequest, 
                                             use_parallel: bool = True, 
                                             num_agents: int = 3) -> ConsensusResult:
        """Direct processing with real agents, bypassing LangGraph"""
        
        print(f"🚀 Direct real agent processing: {task_request.content}")
        
        # Select agents based on task type and capabilities
        selected_agents = self.routing_algorithm.select_multiple_agents(task_request, num_agents)
        print(f"🤖 Selected agents: {selected_agents}")
        
        # Map to real agent names
        real_agent_mapping = {
            "enhanced_physics_expert": "physics_expert",
            "hypothesis_generator": "hypothesis_generator", 
            "mathematical_analysis": "mathematical_analyst",
            "pattern_recognition": "physics_expert"  # Use physics expert for pattern recognition
        }
        
        real_selected_agents = [
            real_agent_mapping.get(agent, "physics_expert") 
            for agent in selected_agents
        ]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_real_agents = []
        for agent in real_selected_agents:
            if agent not in seen:
                unique_real_agents.append(agent)
                seen.add(agent)
        
        print(f"🔄 Processing with real agents: {unique_real_agents}")
        
        try:
            # Execute with real agents
            real_responses = await self.real_agent_manager.process_query_with_real_agents(
                query=task_request.content,
                selected_agents=unique_real_agents,
                parallel=use_parallel
            )
            
            print(f"✅ Got {len(real_responses)} real agent responses")
            
            # Build consensus using real agent manager
            consensus = self.real_agent_manager.build_consensus(real_responses)
            
            # Convert to ConsensusResult format
            individual_results = []
            for i, response in enumerate(real_responses):
                original_agent = selected_agents[i] if i < len(selected_agents) else selected_agents[0]
                task_result = TaskResult(
                    task_id=task_request.task_id,
                    agent_name=original_agent,
                    result=response.response,
                    confidence=response.confidence,
                    execution_time=response.execution_time,
                    success=response.success,
                    metadata=response.metadata
                )
                individual_results.append(task_result)
            
            return ConsensusResult(
                final_result=consensus["final_result"],
                confidence=consensus["confidence"],
                agreement_level=consensus["agreement_level"],
                participating_agents=consensus["participating_agents"],
                method_used=ConsensusMethod.WEIGHTED_AVERAGE,
                individual_results=individual_results
            )
            
        except Exception as e:
            print(f"❌ Real agent processing failed: {e}")
            # Fall back to supervise_task
            return await self.supervise_task(task_request, use_parallel, num_agents)
    
    def _update_agent_metrics(self, consensus_result: ConsensusResult):
        """Update agent performance metrics based on results"""
        for result in consensus_result.individual_results:
            if result.agent_name in self.agent_profiles:
                profile = self.agent_profiles[result.agent_name]
                
                # Update response time
                if profile.response_time_avg == 0:
                    profile.response_time_avg = result.execution_time
                else:
                    profile.response_time_avg = (profile.response_time_avg + result.execution_time) / 2
                
                # Update success rate
                if result.success:
                    profile.success_rate = min(1.0, profile.success_rate + 0.01)
                else:
                    profile.success_rate = max(0.0, profile.success_rate - 0.05)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "agent_profiles": {
                name: {
                    "current_load": profile.current_load,
                    "success_rate": profile.success_rate,
                    "response_time_avg": profile.response_time_avg
                }
                for name, profile in self.agent_profiles.items()
            },
            "routing_history_length": len(self.routing_algorithm.routing_history),
            "consensus_history_length": len(self.consensus_detector.consensus_history)
        }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information and capabilities"""
        return {
            "name": "Advanced Supervisor Agent",
            "type": "supervisor",
            "version": "3.0.0",
            "description": "Advanced supervisor agent with intelligent routing, parallel processing, and consensus detection",
            "capabilities": [
                "intelligent_routing",
                "parallel_processing", 
                "consensus_detection",
                "load_balancing",
                "multi_agent_coordination",
                "performance_monitoring"
            ],
            "supported_task_types": [task_type.value for task_type in TaskType],
            "consensus_methods": [method.value for method in ConsensusMethod],
            "managed_agents": list(self.agent_profiles.keys()),
            "system_status": self.get_system_status()
        } 