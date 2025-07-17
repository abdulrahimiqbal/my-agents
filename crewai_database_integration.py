#!/usr/bin/env python3
"""
CrewAI Database & Evaluation Integration
Adapts existing database/evaluation components to work with CrewAI instead of LangGraph.
"""

import os
import json
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

# CrewAI imports
from crewai import Agent, Task, Crew


class TaskType(Enum):
    """Types of tasks for evaluation tracking."""
    PHYSICS_ANALYSIS = "physics_analysis"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    MATHEMATICAL_COMPUTATION = "mathematical_computation"
    EXPERIMENTAL_DESIGN = "experimental_design"
    SCIENTIFIC_VALIDATION = "scientific_validation"


class FeedbackType(Enum):
    """Types of feedback for learning."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    CORRECTION = "correction"


@dataclass
class InteractionRecord:
    """Record of agent interaction for evaluation."""
    interaction_id: str
    agent_id: str
    task_type: TaskType
    user_query: str
    agent_response: str
    feedback_type: Optional[FeedbackType] = None
    feedback_score: Optional[float] = None
    feedback_text: Optional[str] = None
    context: Dict[str, Any] = None
    timestamp: datetime = None
    success: bool = True
    response_time: float = 0.0
    user_satisfaction: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.context is None:
            self.context = {}


@dataclass
class LearningMetrics:
    """Metrics for tracking learning progress."""
    agent_id: str
    task_type: TaskType
    total_interactions: int = 0
    successful_interactions: int = 0
    average_feedback_score: float = 0.0
    average_response_time: float = 0.0
    improvement_rate: float = 0.0
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()


class CrewAIKnowledgeAPI:
    """
    Knowledge API adapted for CrewAI agents.
    Provides event logging, hypothesis tracking, and knowledge management.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize the CrewAI Knowledge API."""
        self.db_path = db_path or "./data/crewai_knowledge.db"
        
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Events table for logging agent activities
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload TEXT,  -- JSON
                    thread_id TEXT,
                    session_id TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Hypotheses table for scientific hypothesis tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS hypotheses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    confidence_score REAL DEFAULT 0.5,
                    validation_status TEXT DEFAULT 'pending',
                    created_by TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    evidence TEXT,  -- JSON
                    evaluations TEXT  -- JSON
                )
            """)
            
            # Knowledge entries for validated findings
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    physics_domain TEXT,
                    confidence_level REAL DEFAULT 0.8,
                    source_agents TEXT,  -- JSON
                    validation_count INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    tags TEXT  -- JSON
                )
            """)
            
            # Agent performance metrics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_metrics (
                    agent_id TEXT,
                    task_type TEXT,
                    total_tasks INTEGER DEFAULT 0,
                    successful_tasks INTEGER DEFAULT 0,
                    average_response_time REAL DEFAULT 0.0,
                    average_quality_score REAL DEFAULT 0.0,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (agent_id, task_type)
                )
            """)
            
            # Crew execution logs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS crew_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    crew_name TEXT NOT NULL,
                    input_query TEXT NOT NULL,
                    agents_involved TEXT,  -- JSON
                    execution_time REAL,
                    success BOOLEAN DEFAULT TRUE,
                    result_quality REAL,
                    output_summary TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_source ON agent_events(source)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON agent_events(event_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_hypotheses_status ON hypotheses(validation_status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_domain ON knowledge_entries(physics_domain)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_agent ON agent_metrics(agent_id)")
            
            conn.commit()
    
    def log_event(self, source: str, event_type: str, payload: Dict[str, Any] = None, 
                  thread_id: str = None, session_id: str = None):
        """Log an agent event."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO agent_events (source, event_type, payload, thread_id, session_id)
                VALUES (?, ?, ?, ?, ?)
            """, (
                source,
                event_type,
                json.dumps(payload) if payload else None,
                thread_id,
                session_id
            ))
            conn.commit()
    
    def record_hypothesis(self, title: str, description: str, created_by: str,
                         confidence_score: float = 0.5) -> int:
        """Record a new hypothesis."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO hypotheses (title, description, confidence_score, created_by)
                VALUES (?, ?, ?, ?)
            """, (title, description, confidence_score, created_by))
            hypothesis_id = cursor.lastrowid
            conn.commit()
            return hypothesis_id
    
    def add_knowledge_entry(self, title: str, content: str, physics_domain: str,
                           source_agents: List[str], confidence_level: float = 0.8,
                           tags: List[str] = None) -> int:
        """Add a validated knowledge entry."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO knowledge_entries 
                (title, content, physics_domain, source_agents, confidence_level, tags)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                title,
                content,
                physics_domain,
                json.dumps(source_agents),
                confidence_level,
                json.dumps(tags or [])
            ))
            entry_id = cursor.lastrowid
            conn.commit()
            return entry_id
    
    def update_agent_metrics(self, agent_id: str, task_type: str, 
                           execution_time: float, success: bool, quality_score: float = None):
        """Update agent performance metrics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get current metrics
            cursor.execute("""
                SELECT total_tasks, successful_tasks, average_response_time, average_quality_score
                FROM agent_metrics WHERE agent_id = ? AND task_type = ?
            """, (agent_id, task_type))
            
            result = cursor.fetchone()
            
            if result:
                total_tasks, successful_tasks, avg_time, avg_quality = result
                total_tasks += 1
                if success:
                    successful_tasks += 1
                
                # Update averages
                new_avg_time = (avg_time * (total_tasks - 1) + execution_time) / total_tasks
                if quality_score is not None:
                    new_avg_quality = (avg_quality * (total_tasks - 1) + quality_score) / total_tasks
                else:
                    new_avg_quality = avg_quality
                
                cursor.execute("""
                    UPDATE agent_metrics 
                    SET total_tasks = ?, successful_tasks = ?, 
                        average_response_time = ?, average_quality_score = ?,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE agent_id = ? AND task_type = ?
                """, (total_tasks, successful_tasks, new_avg_time, new_avg_quality, agent_id, task_type))
            else:
                # Insert new record
                cursor.execute("""
                    INSERT INTO agent_metrics 
                    (agent_id, task_type, total_tasks, successful_tasks, 
                     average_response_time, average_quality_score)
                    VALUES (?, ?, 1, ?, ?, ?)
                """, (agent_id, task_type, 1 if success else 0, execution_time, quality_score or 0.0))
            
            conn.commit()
    
    def log_crew_execution(self, crew_name: str, input_query: str, 
                          agents_involved: List[str], execution_time: float,
                          success: bool, result_quality: float = None,
                          output_summary: str = None):
        """Log a crew execution for analysis."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO crew_executions 
                (crew_name, input_query, agents_involved, execution_time, 
                 success, result_quality, output_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                crew_name,
                input_query,
                json.dumps(agents_involved),
                execution_time,
                success,
                result_quality,
                output_summary
            ))
            conn.commit()
    
    def get_agent_performance(self, agent_id: str = None) -> List[Dict[str, Any]]:
        """Get agent performance metrics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if agent_id:
                cursor.execute("""
                    SELECT * FROM agent_metrics WHERE agent_id = ?
                    ORDER BY last_updated DESC
                """, (agent_id,))
            else:
                cursor.execute("""
                    SELECT * FROM agent_metrics ORDER BY last_updated DESC
                """)
            
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_recent_events(self, limit: int = 100, source: str = None) -> List[Dict[str, Any]]:
        """Get recent agent events."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if source:
                cursor.execute("""
                    SELECT * FROM agent_events WHERE source = ? 
                    ORDER BY timestamp DESC LIMIT ?
                """, (source, limit))
            else:
                cursor.execute("""
                    SELECT * FROM agent_events ORDER BY timestamp DESC LIMIT ?
                """, (limit,))
            
            columns = [desc[0] for desc in cursor.description]
            events = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            # Parse JSON payloads
            for event in events:
                if event['payload']:
                    try:
                        event['payload'] = json.loads(event['payload'])
                    except:
                        pass
            
            return events
    
    def get_system_analytics(self) -> Dict[str, Any]:
        """Get overall system analytics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total events
            cursor.execute("SELECT COUNT(*) FROM agent_events")
            total_events = cursor.fetchone()[0]
            
            # Total hypotheses
            cursor.execute("SELECT COUNT(*) FROM hypotheses")
            total_hypotheses = cursor.fetchone()[0]
            
            # Total knowledge entries
            cursor.execute("SELECT COUNT(*) FROM knowledge_entries")
            total_knowledge = cursor.fetchone()[0]
            
            # Average crew execution time
            cursor.execute("SELECT AVG(execution_time) FROM crew_executions WHERE success = 1")
            avg_execution_time = cursor.fetchone()[0] or 0.0
            
            # Success rate
            cursor.execute("SELECT COUNT(*) FROM crew_executions WHERE success = 1")
            successful_executions = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM crew_executions")
            total_executions = cursor.fetchone()[0]
            
            success_rate = successful_executions / max(total_executions, 1)
            
            return {
                "total_events": total_events,
                "total_hypotheses": total_hypotheses,
                "total_knowledge_entries": total_knowledge,
                "average_execution_time": avg_execution_time,
                "success_rate": success_rate,
                "total_crew_executions": total_executions
            }


class CrewAIEvaluationFramework:
    """
    Evaluation framework specifically designed for CrewAI systems.
    Provides performance tracking, quality assessment, and learning analytics.
    """
    
    def __init__(self, knowledge_api: CrewAIKnowledgeAPI = None):
        """Initialize the evaluation framework."""
        self.knowledge_api = knowledge_api or CrewAIKnowledgeAPI()
        self.interaction_history: List[InteractionRecord] = []
        self.learning_metrics: Dict[str, LearningMetrics] = {}
    
    def start_evaluation_session(self, session_id: str):
        """Start a new evaluation session."""
        self.knowledge_api.log_event(
            source="evaluation_framework",
            event_type="session_start",
            payload={"session_id": session_id},
            session_id=session_id
        )
    
    def evaluate_crew_execution(self, crew: Crew, query: str, result: Any,
                               execution_time: float) -> Dict[str, Any]:
        """Evaluate a crew execution comprehensively."""
        start_time = time.time()
        
        # Extract agent information
        agent_names = [agent.role for agent in crew.agents] if hasattr(crew, 'agents') else []
        
        # Assess result quality
        quality_score = self._assess_result_quality(query, result)
        
        # Check if execution was successful
        success = result is not None and str(result).strip() != ""
        
        # Log the execution
        self.knowledge_api.log_crew_execution(
            crew_name="PhysicsLabFlow",
            input_query=query,
            agents_involved=agent_names,
            execution_time=execution_time,
            success=success,
            result_quality=quality_score,
            output_summary=str(result)[:500] if result else None
        )
        
        # Update agent metrics
        for agent_name in agent_names:
            self.knowledge_api.update_agent_metrics(
                agent_id=agent_name,
                task_type="physics_analysis",
                execution_time=execution_time / len(agent_names),  # Distribute time
                success=success,
                quality_score=quality_score
            )
        
        evaluation_time = time.time() - start_time
        
        return {
            "success": success,
            "quality_score": quality_score,
            "execution_time": execution_time,
            "agents_involved": agent_names,
            "evaluation_time": evaluation_time,
            "result_length": len(str(result)) if result else 0
        }
    
    def _assess_result_quality(self, query: str, result: Any) -> float:
        """Assess the quality of a result (simple heuristic)."""
        if not result:
            return 0.0
        
        result_str = str(result)
        
        # Quality indicators
        quality_indicators = [
            len(result_str) > 100,  # Sufficient length
            "physics" in result_str.lower(),  # Physics relevance
            any(term in result_str.lower() for term in ["analysis", "theory", "experiment"]),
            "." in result_str,  # Complete sentences
            len(result_str.split()) > 20,  # Sufficient detail
        ]
        
        # Negative indicators
        negative_indicators = [
            "error" in result_str.lower(),
            "failed" in result_str.lower(),
            len(result_str) < 50,  # Too short
        ]
        
        quality_score = sum(quality_indicators) / len(quality_indicators)
        penalty = sum(negative_indicators) * 0.2
        
        return max(0.0, min(1.0, quality_score - penalty))
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        metrics = self.knowledge_api.get_agent_performance()
        analytics = self.knowledge_api.get_system_analytics()
        
        return {
            "system_analytics": analytics,
            "agent_metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "evaluation_framework": "CrewAI-Compatible"
        }
    
    def record_user_feedback(self, interaction_id: str, feedback_score: float,
                           feedback_text: str = None):
        """Record user feedback for learning."""
        self.knowledge_api.log_event(
            source="user_feedback",
            event_type="feedback_received",
            payload={
                "interaction_id": interaction_id,
                "feedback_score": feedback_score,
                "feedback_text": feedback_text
            }
        )


# Factory functions for easy integration
def create_crewai_knowledge_api(db_path: str = None) -> CrewAIKnowledgeAPI:
    """Factory function to create CrewAI Knowledge API."""
    return CrewAIKnowledgeAPI(db_path)


def create_crewai_evaluation_framework(knowledge_api: CrewAIKnowledgeAPI = None) -> CrewAIEvaluationFramework:
    """Factory function to create CrewAI Evaluation Framework."""
    return CrewAIEvaluationFramework(knowledge_api)


if __name__ == "__main__":
    # Demo usage
    print("🔬 CrewAI Database & Evaluation Integration Demo")
    print("=" * 60)
    
    # Initialize components
    knowledge_api = create_crewai_knowledge_api()
    evaluation_framework = create_crewai_evaluation_framework(knowledge_api)
    
    # Demo logging
    knowledge_api.log_event(
        source="demo_agent",
        event_type="analysis_complete",
        payload={"query": "demo physics question", "confidence": 0.85}
    )
    
    # Demo metrics
    knowledge_api.update_agent_metrics(
        agent_id="physics_expert",
        task_type="physics_analysis",
        execution_time=2.5,
        success=True,
        quality_score=0.9
    )
    
    # Get analytics
    analytics = knowledge_api.get_system_analytics()
    print("📊 System Analytics:")
    for key, value in analytics.items():
        print(f"  - {key}: {value}")
    
    print("\n✅ CrewAI Database & Evaluation Integration Ready!") 