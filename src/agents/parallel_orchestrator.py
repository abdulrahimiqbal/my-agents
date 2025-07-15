"""
Parallel Agent Orchestration System - Phase 3

This module implements a sophisticated parallel agent orchestration system that enables:
1. Concurrent multi-agent processing for complex physics problems
2. Dynamic task decomposition and distribution
3. Real-time agent coordination and synchronization
4. Result aggregation and consensus building
5. Load balancing and resource optimization
6. Fault tolerance and error recovery
"""

import asyncio
import threading
import time
from typing import Dict, List, Optional, Any, Tuple, Set, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from collections import defaultdict, deque
import numpy as np
from queue import Queue, PriorityQueue
import uuid

from .base import BaseAgent
from .advanced_supervisor import AdvancedSupervisorAgent, TaskType, TaskRequest, TaskResult, AgentCapability
from .enhanced_physics_expert import EnhancedPhysicsExpertAgent, PhysicsDomain
from .hypothesis_generator import EnhancedHypothesisGeneratorAgent, CreativeReasoningPattern
from .mathematical_analysis import MathematicalAnalysisAgent
from .pattern_recognition import PatternRecognitionAgent
from ..knowledge.knowledge_graph import PhysicsKnowledgeGraph
from ..database.knowledge_api import KnowledgeAPI


class OrchestrationStrategy(Enum):
    """Strategies for orchestrating parallel agent execution."""
    PIPELINE = "pipeline"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"
    COLLABORATIVE = "collaborative"
    COMPETITIVE = "competitive"


class SynchronizationMode(Enum):
    """Modes for synchronizing parallel agents."""
    BARRIER = "barrier"
    MILESTONE = "milestone"
    STREAMING = "streaming"
    ASYNCHRONOUS = "asynchronous"
    CONSENSUS = "consensus"


class TaskPriority(Enum):
    """Priority levels for task execution."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class ParallelTask:
    """Represents a task that can be executed in parallel."""
    task_id: str
    task_type: TaskType
    description: str
    input_data: Dict[str, Any]
    priority: TaskPriority
    dependencies: List[str] = field(default_factory=list)
    required_capabilities: Set[AgentCapability] = field(default_factory=set)
    estimated_duration: float = 0.0
    max_retries: int = 3
    timeout: float = 300.0  # 5 minutes default
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ParallelResult:
    """Represents the result of a parallel task execution."""
    task_id: str
    agent_id: str
    result_data: Dict[str, Any]
    success: bool
    execution_time: float
    error_message: Optional[str] = None
    confidence_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    completed_at: datetime = field(default_factory=datetime.now)


@dataclass
class AgentWorker:
    """Represents an agent worker in the parallel orchestration system."""
    agent_id: str
    agent_instance: BaseAgent
    capabilities: Set[AgentCapability]
    current_task: Optional[str] = None
    is_busy: bool = False
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    last_activity: datetime = field(default_factory=datetime.now)
    error_count: int = 0
    success_count: int = 0


@dataclass
class OrchestrationSession:
    """Represents a parallel orchestration session."""
    session_id: str
    strategy: OrchestrationStrategy
    sync_mode: SynchronizationMode
    tasks: List[ParallelTask] = field(default_factory=list)
    results: List[ParallelResult] = field(default_factory=list)
    active_tasks: Dict[str, str] = field(default_factory=dict)  # task_id -> agent_id
    completed_tasks: Set[str] = field(default_factory=set)
    failed_tasks: Set[str] = field(default_factory=set)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ParallelAgentOrchestrator:
    """
    Advanced parallel agent orchestration system for complex physics problem solving.
    
    Features:
    - Concurrent multi-agent processing
    - Dynamic task decomposition and distribution
    - Real-time coordination and synchronization
    - Result aggregation and consensus building
    - Load balancing and resource optimization
    - Fault tolerance and error recovery
    """
    
    def __init__(self, 
                 max_workers: int = 8,
                 default_timeout: float = 300.0,
                 enable_load_balancing: bool = True,
                 enable_fault_tolerance: bool = True,
                 knowledge_graph: Optional[PhysicsKnowledgeGraph] = None):
        """
        Initialize the parallel agent orchestrator.
        
        Args:
            max_workers: Maximum number of concurrent worker threads
            default_timeout: Default timeout for task execution
            enable_load_balancing: Whether to enable dynamic load balancing
            enable_fault_tolerance: Whether to enable fault tolerance mechanisms
            knowledge_graph: Shared knowledge graph for agents
        """
        self.max_workers = max_workers
        self.default_timeout = default_timeout
        self.enable_load_balancing = enable_load_balancing
        self.enable_fault_tolerance = enable_fault_tolerance
        self.knowledge_graph = knowledge_graph or PhysicsKnowledgeGraph()
        
        # Initialize agent workers
        self.agent_workers: Dict[str, AgentWorker] = {}
        self.task_queue = PriorityQueue()
        self.result_queue = Queue()
        
        # Initialize orchestration components
        self.supervisor = AdvancedSupervisorAgent()
        self.sessions: Dict[str, OrchestrationSession] = {}
        self.active_sessions: Set[str] = set()
        
        # Initialize synchronization primitives
        self.sync_barriers: Dict[str, threading.Barrier] = {}
        self.sync_events: Dict[str, threading.Event] = {}
        self.sync_locks: Dict[str, threading.Lock] = {}
        
        # Initialize performance tracking
        self.performance_metrics = {
            "total_tasks_executed": 0,
            "total_execution_time": 0.0,
            "average_task_time": 0.0,
            "success_rate": 0.0,
            "throughput": 0.0,
            "resource_utilization": 0.0
        }
        
        # Initialize executor
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.running = False
        
        # Initialize knowledge API
        self.knowledge_api = KnowledgeAPI()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize default agents
        self._initialize_default_agents()
    
    def _initialize_default_agents(self):
        """Initialize default agent workers."""
        # Physics Expert Agent
        physics_expert = EnhancedPhysicsExpertAgent(
            specialization=PhysicsDomain.GENERAL_PHYSICS,
            knowledge_graph=self.knowledge_graph
        )
        self.register_agent_worker(
            agent_id="physics_expert",
            agent_instance=physics_expert,
            capabilities={
                AgentCapability.QUANTUM_MECHANICS,
                AgentCapability.THERMODYNAMICS,
                AgentCapability.ELECTROMAGNETISM,
                AgentCapability.MECHANICS,
                AgentCapability.RELATIVITY,
                AgentCapability.KNOWLEDGE_SYNTHESIS
            }
        )
        
        # Hypothesis Generator Agent
        hypothesis_generator = EnhancedHypothesisGeneratorAgent(
            creativity_level="high",
            exploration_scope="interdisciplinary",
            knowledge_graph=self.knowledge_graph
        )
        self.register_agent_worker(
            agent_id="hypothesis_generator",
            agent_instance=hypothesis_generator,
            capabilities={
                AgentCapability.CREATIVE_THINKING,
                AgentCapability.HYPOTHESIS_GENERATION,
                AgentCapability.INTERDISCIPLINARY_ANALYSIS
            }
        )
        
        # Mathematical Analysis Agent
        math_agent = MathematicalAnalysisAgent()
        self.register_agent_worker(
            agent_id="mathematical_analysis",
            agent_instance=math_agent,
            capabilities={
                AgentCapability.STATISTICAL_ANALYSIS,
                AgentCapability.CURVE_FITTING,
                AgentCapability.HYPOTHESIS_TESTING
            }
        )
        
        # Pattern Recognition Agent
        pattern_agent = PatternRecognitionAgent()
        self.register_agent_worker(
            agent_id="pattern_recognition",
            agent_instance=pattern_agent,
            capabilities={
                AgentCapability.PATTERN_RECOGNITION,
                AgentCapability.DATA_ANALYSIS,
                AgentCapability.RELATIONSHIP_DISCOVERY
            }
        )
    
    def register_agent_worker(self, 
                            agent_id: str, 
                            agent_instance: BaseAgent, 
                            capabilities: Set[AgentCapability]):
        """Register a new agent worker."""
        worker = AgentWorker(
            agent_id=agent_id,
            agent_instance=agent_instance,
            capabilities=capabilities,
            performance_metrics={
                "tasks_completed": 0,
                "average_execution_time": 0.0,
                "success_rate": 1.0,
                "load_factor": 0.0
            }
        )
        self.agent_workers[agent_id] = worker
        self.logger.info(f"Registered agent worker: {agent_id}")
    
    def create_orchestration_session(self, 
                                   strategy: OrchestrationStrategy = OrchestrationStrategy.ADAPTIVE,
                                   sync_mode: SynchronizationMode = SynchronizationMode.MILESTONE,
                                   session_metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new orchestration session."""
        session_id = str(uuid.uuid4())
        
        session = OrchestrationSession(
            session_id=session_id,
            strategy=strategy,
            sync_mode=sync_mode,
            metadata=session_metadata or {}
        )
        
        self.sessions[session_id] = session
        self.active_sessions.add(session_id)
        
        # Initialize synchronization primitives for this session
        if sync_mode == SynchronizationMode.BARRIER:
            self.sync_barriers[session_id] = threading.Barrier(len(self.agent_workers))
        elif sync_mode == SynchronizationMode.MILESTONE:
            self.sync_events[session_id] = threading.Event()
        
        self.sync_locks[session_id] = threading.Lock()
        
        self.logger.info(f"Created orchestration session: {session_id}")
        return session_id
    
    def add_parallel_task(self, 
                         session_id: str,
                         task_type: TaskType,
                         description: str,
                         input_data: Dict[str, Any],
                         priority: TaskPriority = TaskPriority.MEDIUM,
                         dependencies: Optional[List[str]] = None,
                         required_capabilities: Optional[Set[AgentCapability]] = None,
                         estimated_duration: float = 0.0,
                         timeout: Optional[float] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a task to the parallel orchestration session."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        task_id = str(uuid.uuid4())
        task = ParallelTask(
            task_id=task_id,
            task_type=task_type,
            description=description,
            input_data=input_data,
            priority=priority,
            dependencies=dependencies or [],
            required_capabilities=required_capabilities or set(),
            estimated_duration=estimated_duration,
            timeout=timeout or self.default_timeout,
            metadata=metadata or {}
        )
        
        self.sessions[session_id].tasks.append(task)
        self.logger.info(f"Added task {task_id} to session {session_id}")
        
        return task_id
    
    def execute_session(self, session_id: str) -> Dict[str, Any]:
        """Execute all tasks in a parallel orchestration session."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        self.logger.info(f"Starting execution of session {session_id} with {len(session.tasks)} tasks")
        
        # Decompose and distribute tasks based on strategy
        if session.strategy == OrchestrationStrategy.PIPELINE:
            return self._execute_pipeline_strategy(session)
        elif session.strategy == OrchestrationStrategy.PARALLEL:
            return self._execute_parallel_strategy(session)
        elif session.strategy == OrchestrationStrategy.HIERARCHICAL:
            return self._execute_hierarchical_strategy(session)
        elif session.strategy == OrchestrationStrategy.ADAPTIVE:
            return self._execute_adaptive_strategy(session)
        elif session.strategy == OrchestrationStrategy.COLLABORATIVE:
            return self._execute_collaborative_strategy(session)
        elif session.strategy == OrchestrationStrategy.COMPETITIVE:
            return self._execute_competitive_strategy(session)
        else:
            raise ValueError(f"Unknown orchestration strategy: {session.strategy}")
    
    def _execute_parallel_strategy(self, session: OrchestrationSession) -> Dict[str, Any]:
        """Execute tasks using parallel strategy."""
        futures = []
        task_futures = {}
        
        # Submit all independent tasks for parallel execution
        for task in session.tasks:
            if not task.dependencies:  # Independent tasks
                future = self.executor.submit(self._execute_task, session.session_id, task)
                futures.append(future)
                task_futures[task.task_id] = future
        
        # Wait for completion and handle dependent tasks
        completed_tasks = set()
        
        while len(completed_tasks) < len(session.tasks):
            # Check completed futures
            for future in as_completed(futures, timeout=1.0):
                if future not in task_futures.values():
                    continue
                
                # Find the task for this future
                task_id = None
                for tid, f in task_futures.items():
                    if f == future:
                        task_id = tid
                        break
                
                if task_id and task_id not in completed_tasks:
                    try:
                        result = future.result()
                        session.results.append(result)
                        completed_tasks.add(task_id)
                        session.completed_tasks.add(task_id)
                        
                        # Submit dependent tasks
                        self._submit_dependent_tasks(session, task_id, task_futures, futures)
                        
                    except Exception as e:
                        self.logger.error(f"Task {task_id} failed: {e}")
                        session.failed_tasks.add(task_id)
                        completed_tasks.add(task_id)
        
        # Aggregate results
        session.end_time = datetime.now()
        return self._aggregate_session_results(session)
    
    def _execute_adaptive_strategy(self, session: OrchestrationSession) -> Dict[str, Any]:
        """Execute tasks using adaptive strategy that adjusts based on performance."""
        # Start with parallel execution
        parallel_results = self._execute_parallel_strategy(session)
        
        # Analyze performance and adapt
        performance_metrics = self._analyze_session_performance(session)
        
        # If performance is poor, switch to hierarchical
        if performance_metrics.get("success_rate", 0) < 0.8:
            self.logger.info("Switching to hierarchical strategy due to poor performance")
            session.strategy = OrchestrationStrategy.HIERARCHICAL
            return self._execute_hierarchical_strategy(session)
        
        # If tasks are highly interdependent, switch to collaborative
        if performance_metrics.get("dependency_ratio", 0) > 0.6:
            self.logger.info("Switching to collaborative strategy due to high interdependency")
            session.strategy = OrchestrationStrategy.COLLABORATIVE
            return self._execute_collaborative_strategy(session)
        
        return parallel_results
    
    def _execute_collaborative_strategy(self, session: OrchestrationSession) -> Dict[str, Any]:
        """Execute tasks using collaborative strategy where agents work together."""
        # Group related tasks
        task_groups = self._group_related_tasks(session.tasks)
        
        # Execute each group collaboratively
        for group in task_groups:
            # Select best agents for this group
            selected_agents = self._select_collaborative_agents(group)
            
            # Execute tasks in collaborative mode
            group_results = self._execute_collaborative_group(session, group, selected_agents)
            session.results.extend(group_results)
        
        session.end_time = datetime.now()
        return self._aggregate_session_results(session)
    
    def _execute_competitive_strategy(self, session: OrchestrationSession) -> Dict[str, Any]:
        """Execute tasks using competitive strategy where multiple agents compete."""
        competitive_results = []
        
        for task in session.tasks:
            # Select multiple capable agents
            capable_agents = self._find_capable_agents(task.required_capabilities)
            
            if len(capable_agents) > 1:
                # Run task on multiple agents competitively
                futures = []
                for agent_id in capable_agents[:3]:  # Limit to top 3
                    future = self.executor.submit(self._execute_task, session.session_id, task, agent_id)
                    futures.append((future, agent_id))
                
                # Take the best result
                best_result = None
                best_score = -1
                
                for future, agent_id in futures:
                    try:
                        result = future.result(timeout=task.timeout)
                        if result.success and result.confidence_score > best_score:
                            best_result = result
                            best_score = result.confidence_score
                    except Exception as e:
                        self.logger.error(f"Competitive execution failed for {agent_id}: {e}")
                
                if best_result:
                    competitive_results.append(best_result)
                    session.completed_tasks.add(task.task_id)
                else:
                    session.failed_tasks.add(task.task_id)
            else:
                # Single agent execution
                result = self._execute_task(session.session_id, task)
                competitive_results.append(result)
        
        session.results.extend(competitive_results)
        session.end_time = datetime.now()
        return self._aggregate_session_results(session)
    
    def _execute_task(self, 
                     session_id: str, 
                     task: ParallelTask, 
                     preferred_agent_id: Optional[str] = None) -> ParallelResult:
        """Execute a single task."""
        start_time = time.time()
        
        # Select agent
        if preferred_agent_id and preferred_agent_id in self.agent_workers:
            agent_id = preferred_agent_id
        else:
            agent_id = self._select_best_agent(task)
        
        if not agent_id:
            return ParallelResult(
                task_id=task.task_id,
                agent_id="none",
                result_data={},
                success=False,
                execution_time=time.time() - start_time,
                error_message="No suitable agent found"
            )
        
        # Mark agent as busy
        worker = self.agent_workers[agent_id]
        worker.is_busy = True
        worker.current_task = task.task_id
        
        try:
            # Execute task based on type
            if task.task_type == TaskType.PHYSICS_ANALYSIS:
                result_data = self._execute_physics_analysis(worker.agent_instance, task)
            elif task.task_type == TaskType.HYPOTHESIS_GENERATION:
                result_data = self._execute_hypothesis_generation(worker.agent_instance, task)
            elif task.task_type == TaskType.MATHEMATICAL_COMPUTATION:
                result_data = self._execute_mathematical_computation(worker.agent_instance, task)
            elif task.task_type == TaskType.PATTERN_RECOGNITION:
                result_data = self._execute_pattern_recognition(worker.agent_instance, task)
            else:
                result_data = self._execute_generic_task(worker.agent_instance, task)
            
            # Create successful result
            execution_time = time.time() - start_time
            result = ParallelResult(
                task_id=task.task_id,
                agent_id=agent_id,
                result_data=result_data,
                success=True,
                execution_time=execution_time,
                confidence_score=result_data.get("confidence", 0.8)
            )
            
            # Update performance metrics
            worker.success_count += 1
            worker.performance_metrics["tasks_completed"] += 1
            worker.performance_metrics["average_execution_time"] = (
                (worker.performance_metrics["average_execution_time"] * (worker.success_count - 1) + execution_time) / 
                worker.success_count
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            worker.error_count += 1
            
            return ParallelResult(
                task_id=task.task_id,
                agent_id=agent_id,
                result_data={},
                success=False,
                execution_time=execution_time,
                error_message=str(e)
            )
            
        finally:
            # Mark agent as available
            worker.is_busy = False
            worker.current_task = None
            worker.last_activity = datetime.now()
    
    def _select_best_agent(self, task: ParallelTask) -> Optional[str]:
        """Select the best agent for a task."""
        # Find agents with required capabilities
        capable_agents = self._find_capable_agents(task.required_capabilities)
        
        if not capable_agents:
            return None
        
        # If load balancing is enabled, consider current load
        if self.enable_load_balancing:
            # Score agents based on availability and performance
            agent_scores = {}
            for agent_id in capable_agents:
                worker = self.agent_workers[agent_id]
                
                # Base score from performance
                base_score = worker.performance_metrics.get("success_rate", 0.5)
                
                # Penalty for being busy
                if worker.is_busy:
                    base_score *= 0.1
                
                # Bonus for recent activity (freshness)
                time_since_activity = (datetime.now() - worker.last_activity).total_seconds()
                freshness_bonus = max(0, 1 - time_since_activity / 3600)  # Decay over 1 hour
                
                agent_scores[agent_id] = base_score + freshness_bonus * 0.2
            
            # Select agent with highest score
            return max(agent_scores.items(), key=lambda x: x[1])[0]
        else:
            # Simple round-robin selection
            available_agents = [aid for aid in capable_agents if not self.agent_workers[aid].is_busy]
            return available_agents[0] if available_agents else capable_agents[0]
    
    def _find_capable_agents(self, required_capabilities: Set[AgentCapability]) -> List[str]:
        """Find agents with required capabilities."""
        capable_agents = []
        
        for agent_id, worker in self.agent_workers.items():
            if required_capabilities.issubset(worker.capabilities):
                capable_agents.append(agent_id)
        
        return capable_agents
    
    def _execute_physics_analysis(self, agent: BaseAgent, task: ParallelTask) -> Dict[str, Any]:
        """Execute physics analysis task."""
        if hasattr(agent, 'analyze_physics_problem'):
            return agent.analyze_physics_problem(
                problem=task.input_data.get("problem", ""),
                context=task.input_data.get("context", "")
            )
        else:
            return {"error": "Agent does not support physics analysis"}
    
    def _execute_hypothesis_generation(self, agent: BaseAgent, task: ParallelTask) -> Dict[str, Any]:
        """Execute hypothesis generation task."""
        if hasattr(agent, 'generate_creative_hypotheses'):
            return agent.generate_creative_hypotheses(
                topic=task.input_data.get("topic", ""),
                context=task.input_data.get("context", ""),
                num_hypotheses=task.input_data.get("num_hypotheses", 3)
            )
        else:
            return {"error": "Agent does not support hypothesis generation"}
    
    def _execute_mathematical_computation(self, agent: BaseAgent, task: ParallelTask) -> Dict[str, Any]:
        """Execute mathematical computation task."""
        if hasattr(agent, 'analyze_experimental_data'):
            return agent.analyze_experimental_data(
                data=task.input_data.get("data", {}),
                analysis_types=task.input_data.get("analysis_types", ["statistical"])
            )
        else:
            return {"error": "Agent does not support mathematical computation"}
    
    def _execute_pattern_recognition(self, agent: BaseAgent, task: ParallelTask) -> Dict[str, Any]:
        """Execute pattern recognition task."""
        if hasattr(agent, 'recognize_patterns'):
            return agent.recognize_patterns(
                datasets=task.input_data.get("datasets", []),
                pattern_types=task.input_data.get("pattern_types", [])
            )
        else:
            return {"error": "Agent does not support pattern recognition"}
    
    def _execute_generic_task(self, agent: BaseAgent, task: ParallelTask) -> Dict[str, Any]:
        """Execute generic task."""
        if hasattr(agent, 'chat'):
            return {
                "response": agent.chat(task.description),
                "confidence": 0.7
            }
        else:
            return {"error": "Agent does not support generic tasks"}
    
    def _submit_dependent_tasks(self, 
                              session: OrchestrationSession, 
                              completed_task_id: str,
                              task_futures: Dict[str, Future],
                              futures: List[Future]):
        """Submit tasks that depend on the completed task."""
        for task in session.tasks:
            if (completed_task_id in task.dependencies and 
                task.task_id not in session.completed_tasks and
                task.task_id not in session.failed_tasks and
                task.task_id not in task_futures):
                
                # Check if all dependencies are satisfied
                if all(dep in session.completed_tasks for dep in task.dependencies):
                    future = self.executor.submit(self._execute_task, session.session_id, task)
                    futures.append(future)
                    task_futures[task.task_id] = future
    
    def _group_related_tasks(self, tasks: List[ParallelTask]) -> List[List[ParallelTask]]:
        """Group related tasks for collaborative execution."""
        # Simple grouping by task type
        groups = defaultdict(list)
        for task in tasks:
            groups[task.task_type].append(task)
        
        return list(groups.values())
    
    def _select_collaborative_agents(self, task_group: List[ParallelTask]) -> List[str]:
        """Select agents for collaborative execution."""
        # Find agents that can handle tasks in this group
        all_capabilities = set()
        for task in task_group:
            all_capabilities.update(task.required_capabilities)
        
        capable_agents = self._find_capable_agents(all_capabilities)
        return capable_agents[:3]  # Limit to 3 agents per group
    
    def _execute_collaborative_group(self, 
                                   session: OrchestrationSession,
                                   task_group: List[ParallelTask],
                                   selected_agents: List[str]) -> List[ParallelResult]:
        """Execute a group of tasks collaboratively."""
        results = []
        
        for task in task_group:
            # Execute task with collaboration
            result = self._execute_task(session.session_id, task)
            results.append(result)
            
            # Add to completed tasks
            if result.success:
                session.completed_tasks.add(task.task_id)
            else:
                session.failed_tasks.add(task.task_id)
        
        return results
    
    def _aggregate_session_results(self, session: OrchestrationSession) -> Dict[str, Any]:
        """Aggregate results from a session."""
        successful_results = [r for r in session.results if r.success]
        failed_results = [r for r in session.results if not r.success]
        
        total_execution_time = sum(r.execution_time for r in session.results)
        average_confidence = np.mean([r.confidence_score for r in successful_results]) if successful_results else 0.0
        
        return {
            "session_id": session.session_id,
            "strategy": session.strategy.value,
            "sync_mode": session.sync_mode.value,
            "total_tasks": len(session.tasks),
            "successful_tasks": len(successful_results),
            "failed_tasks": len(failed_results),
            "success_rate": len(successful_results) / len(session.tasks) if session.tasks else 0.0,
            "total_execution_time": total_execution_time,
            "average_confidence": average_confidence,
            "results": [
                {
                    "task_id": r.task_id,
                    "agent_id": r.agent_id,
                    "success": r.success,
                    "execution_time": r.execution_time,
                    "confidence_score": r.confidence_score,
                    "result_data": r.result_data
                }
                for r in session.results
            ],
            "performance_metrics": self._analyze_session_performance(session),
            "start_time": session.start_time.isoformat(),
            "end_time": session.end_time.isoformat() if session.end_time else None
        }
    
    def _analyze_session_performance(self, session: OrchestrationSession) -> Dict[str, float]:
        """Analyze performance metrics for a session."""
        if not session.results:
            return {}
        
        successful_results = [r for r in session.results if r.success]
        
        return {
            "success_rate": len(successful_results) / len(session.results),
            "average_execution_time": np.mean([r.execution_time for r in session.results]),
            "throughput": len(session.results) / (session.end_time - session.start_time).total_seconds() if session.end_time else 0.0,
            "average_confidence": np.mean([r.confidence_score for r in successful_results]) if successful_results else 0.0,
            "dependency_ratio": sum(len(task.dependencies) for task in session.tasks) / len(session.tasks) if session.tasks else 0.0,
            "resource_utilization": len([w for w in self.agent_workers.values() if w.is_busy]) / len(self.agent_workers)
        }
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get the current status of a session."""
        if session_id not in self.sessions:
            return {"error": "Session not found"}
        
        session = self.sessions[session_id]
        
        return {
            "session_id": session_id,
            "strategy": session.strategy.value,
            "sync_mode": session.sync_mode.value,
            "total_tasks": len(session.tasks),
            "completed_tasks": len(session.completed_tasks),
            "failed_tasks": len(session.failed_tasks),
            "active_tasks": len(session.active_tasks),
            "is_running": session_id in self.active_sessions,
            "start_time": session.start_time.isoformat(),
            "end_time": session.end_time.isoformat() if session.end_time else None,
            "progress": len(session.completed_tasks) / len(session.tasks) if session.tasks else 0.0
        }
    
    def cancel_session(self, session_id: str) -> bool:
        """Cancel a running session."""
        if session_id not in self.sessions:
            return False
        
        self.active_sessions.discard(session_id)
        
        # Cancel any running tasks for this session
        session = self.sessions[session_id]
        for worker in self.agent_workers.values():
            if worker.current_task and worker.current_task in [t.task_id for t in session.tasks]:
                worker.is_busy = False
                worker.current_task = None
        
        session.end_time = datetime.now()
        self.logger.info(f"Cancelled session: {session_id}")
        return True
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get the current status of all agents."""
        return {
            agent_id: {
                "is_busy": worker.is_busy,
                "current_task": worker.current_task,
                "capabilities": [cap.value for cap in worker.capabilities],
                "performance_metrics": worker.performance_metrics,
                "success_count": worker.success_count,
                "error_count": worker.error_count,
                "last_activity": worker.last_activity.isoformat()
            }
            for agent_id, worker in self.agent_workers.items()
        }
    
    def shutdown(self):
        """Shutdown the orchestrator."""
        self.logger.info("Shutting down parallel orchestrator")
        
        # Cancel all active sessions
        for session_id in list(self.active_sessions):
            self.cancel_session(session_id)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.running = False 