"""
Agent Learning System - Adaptive Behavior Based on Interaction History

This module implements an adaptive learning system that allows agents to:
1. Learn from interaction history and feedback
2. Adapt behavior based on performance metrics
3. Improve responses over time through reinforcement learning
4. Personalize interactions based on user preferences
5. Optimize task routing and agent selection
"""

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import logging

from ..database.knowledge_api import KnowledgeAPI
from ..memory.stores import MemoryStore
from .advanced_supervisor import TaskType, AgentCapability, TaskResult


class LearningStrategy(Enum):
    """Different learning strategies for agent adaptation."""
    REINFORCEMENT = "reinforcement"
    COLLABORATIVE_FILTERING = "collaborative_filtering"
    PERFORMANCE_BASED = "performance_based"
    USER_FEEDBACK = "user_feedback"
    CONTEXTUAL_BANDIT = "contextual_bandit"


class FeedbackType(Enum):
    """Types of feedback for learning."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    CORRECTION = "correction"
    PREFERENCE = "preference"


@dataclass
class InteractionRecord:
    """Record of agent interaction for learning."""
    interaction_id: str
    agent_id: str
    task_type: TaskType
    user_query: str
    agent_response: str
    feedback_type: Optional[FeedbackType] = None
    feedback_score: Optional[float] = None
    feedback_text: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    response_time: float = 0.0
    user_satisfaction: Optional[float] = None


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
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class AdaptationRule:
    """Rule for adapting agent behavior."""
    rule_id: str
    condition: str
    action: str
    priority: int = 1
    confidence: float = 0.5
    usage_count: int = 0
    success_rate: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)


class AgentLearningSystem:
    """
    Adaptive learning system for agent behavior optimization.
    
    Features:
    - Interaction history tracking
    - Performance-based adaptation
    - User feedback integration
    - Contextual learning
    - Behavioral pattern recognition
    """
    
    def __init__(self, 
                 db_path: Optional[str] = None,
                 learning_rate: float = 0.1,
                 adaptation_threshold: float = 0.7,
                 memory_window: int = 1000):
        """
        Initialize the agent learning system.
        
        Args:
            db_path: Path to learning database
            learning_rate: Rate of learning adaptation
            adaptation_threshold: Threshold for triggering adaptations
            memory_window: Number of recent interactions to consider
        """
        self.learning_rate = learning_rate
        self.adaptation_threshold = adaptation_threshold
        self.memory_window = memory_window
        
        # Initialize database
        self.db_path = db_path or "data/learning.db"
        self._init_database()
        
        # Initialize components
        self.knowledge_api = KnowledgeAPI()
        self.memory_store = MemoryStore()
        self.logger = logging.getLogger(__name__)
        
        # Learning state
        self.interaction_history: deque = deque(maxlen=memory_window)
        self.agent_metrics: Dict[str, LearningMetrics] = {}
        self.adaptation_rules: Dict[str, AdaptationRule] = {}
        self.user_preferences: Dict[str, Dict[str, Any]] = {}
        
        # Load existing data
        self._load_learning_data()
    
    def _init_database(self):
        """Initialize the learning database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Interaction records table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interaction_records (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    user_query TEXT NOT NULL,
                    agent_response TEXT NOT NULL,
                    feedback_type TEXT,
                    feedback_score REAL,
                    feedback_text TEXT,
                    context_json TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    success BOOLEAN DEFAULT TRUE,
                    response_time REAL DEFAULT 0.0,
                    user_satisfaction REAL
                )
            """)
            
            # Learning metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learning_metrics (
                    agent_id TEXT,
                    task_type TEXT,
                    total_interactions INTEGER DEFAULT 0,
                    successful_interactions INTEGER DEFAULT 0,
                    average_feedback_score REAL DEFAULT 0.0,
                    average_response_time REAL DEFAULT 0.0,
                    improvement_rate REAL DEFAULT 0.0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (agent_id, task_type)
                )
            """)
            
            # Adaptation rules table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS adaptation_rules (
                    rule_id TEXT PRIMARY KEY,
                    condition TEXT NOT NULL,
                    action TEXT NOT NULL,
                    priority INTEGER DEFAULT 1,
                    confidence REAL DEFAULT 0.5,
                    usage_count INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # User preferences table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT PRIMARY KEY,
                    preferences_json TEXT NOT NULL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_interactions_agent ON interaction_records(agent_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_interactions_task ON interaction_records(task_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON interaction_records(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_agent ON learning_metrics(agent_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rules_priority ON adaptation_rules(priority)")
            
            conn.commit()
    
    def _load_learning_data(self):
        """Load existing learning data from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Load recent interactions
                cursor.execute("""
                    SELECT * FROM interaction_records 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (self.memory_window,))
                
                for row in cursor.fetchall():
                    record = InteractionRecord(
                        interaction_id=row[0],
                        agent_id=row[1],
                        task_type=TaskType(row[2]),
                        user_query=row[3],
                        agent_response=row[4],
                        feedback_type=FeedbackType(row[5]) if row[5] else None,
                        feedback_score=row[6],
                        feedback_text=row[7],
                        context=json.loads(row[8]) if row[8] else {},
                        timestamp=datetime.fromisoformat(row[9]),
                        success=bool(row[10]),
                        response_time=row[11],
                        user_satisfaction=row[12]
                    )
                    self.interaction_history.append(record)
                
                # Load metrics
                cursor.execute("SELECT * FROM learning_metrics")
                for row in cursor.fetchall():
                    key = f"{row[0]}_{row[1]}"
                    self.agent_metrics[key] = LearningMetrics(
                        agent_id=row[0],
                        task_type=TaskType(row[1]),
                        total_interactions=row[2],
                        successful_interactions=row[3],
                        average_feedback_score=row[4],
                        average_response_time=row[5],
                        improvement_rate=row[6],
                        last_updated=datetime.fromisoformat(row[7])
                    )
                
                # Load adaptation rules
                cursor.execute("SELECT * FROM adaptation_rules")
                for row in cursor.fetchall():
                    self.adaptation_rules[row[0]] = AdaptationRule(
                        rule_id=row[0],
                        condition=row[1],
                        action=row[2],
                        priority=row[3],
                        confidence=row[4],
                        usage_count=row[5],
                        success_rate=row[6],
                        created_at=datetime.fromisoformat(row[7])
                    )
                
                # Load user preferences
                cursor.execute("SELECT * FROM user_preferences")
                for row in cursor.fetchall():
                    self.user_preferences[row[0]] = json.loads(row[1])
                    
        except Exception as e:
            self.logger.warning(f"Failed to load learning data: {e}")
    
    def record_interaction(self, interaction: InteractionRecord):
        """Record a new interaction for learning."""
        # Add to memory
        self.interaction_history.append(interaction)
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO interaction_records 
                (id, agent_id, task_type, user_query, agent_response, 
                 feedback_type, feedback_score, feedback_text, context_json,
                 timestamp, success, response_time, user_satisfaction)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                interaction.interaction_id,
                interaction.agent_id,
                interaction.task_type.value,
                interaction.user_query,
                interaction.agent_response,
                interaction.feedback_type.value if interaction.feedback_type else None,
                interaction.feedback_score,
                interaction.feedback_text,
                json.dumps(interaction.context),
                interaction.timestamp.isoformat(),
                interaction.success,
                interaction.response_time,
                interaction.user_satisfaction
            ))
            conn.commit()
        
        # Update metrics
        self._update_metrics(interaction)
        
        # Check for adaptation triggers
        self._check_adaptation_triggers(interaction)
    
    def _update_metrics(self, interaction: InteractionRecord):
        """Update learning metrics based on new interaction."""
        key = f"{interaction.agent_id}_{interaction.task_type.value}"
        
        if key not in self.agent_metrics:
            self.agent_metrics[key] = LearningMetrics(
                agent_id=interaction.agent_id,
                task_type=interaction.task_type
            )
        
        metrics = self.agent_metrics[key]
        
        # Update counters
        metrics.total_interactions += 1
        if interaction.success:
            metrics.successful_interactions += 1
        
        # Update averages
        if interaction.feedback_score is not None:
            old_avg = metrics.average_feedback_score
            n = metrics.total_interactions
            metrics.average_feedback_score = (old_avg * (n-1) + interaction.feedback_score) / n
        
        old_time_avg = metrics.average_response_time
        n = metrics.total_interactions
        metrics.average_response_time = (old_time_avg * (n-1) + interaction.response_time) / n
        
        # Calculate improvement rate
        if metrics.total_interactions > 10:
            recent_success = sum(1 for r in list(self.interaction_history)[-10:] 
                               if r.agent_id == interaction.agent_id and r.success)
            metrics.improvement_rate = recent_success / 10.0
        
        metrics.last_updated = datetime.now()
        
        # Save to database
        self._save_metrics(metrics)
    
    def _save_metrics(self, metrics: LearningMetrics):
        """Save metrics to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO learning_metrics 
                (agent_id, task_type, total_interactions, successful_interactions,
                 average_feedback_score, average_response_time, improvement_rate, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.agent_id,
                metrics.task_type.value,
                metrics.total_interactions,
                metrics.successful_interactions,
                metrics.average_feedback_score,
                metrics.average_response_time,
                metrics.improvement_rate,
                metrics.last_updated.isoformat()
            ))
            conn.commit()
    
    def _check_adaptation_triggers(self, interaction: InteractionRecord):
        """Check if adaptation should be triggered based on interaction."""
        key = f"{interaction.agent_id}_{interaction.task_type.value}"
        
        if key not in self.agent_metrics:
            return
        
        metrics = self.agent_metrics[key]
        
        # Trigger adaptation if performance is below threshold
        if (metrics.total_interactions > 5 and 
            metrics.successful_interactions / metrics.total_interactions < self.adaptation_threshold):
            self._trigger_adaptation(interaction.agent_id, interaction.task_type, "low_performance")
        
        # Trigger adaptation if feedback is consistently negative
        if (interaction.feedback_score is not None and 
            interaction.feedback_score < 0.3 and 
            metrics.average_feedback_score < 0.5):
            self._trigger_adaptation(interaction.agent_id, interaction.task_type, "negative_feedback")
    
    def _trigger_adaptation(self, agent_id: str, task_type: TaskType, reason: str):
        """Trigger adaptation for an agent."""
        self.logger.info(f"Triggering adaptation for {agent_id} on {task_type.value}: {reason}")
        
        # Create adaptation rule
        rule_id = f"adapt_{agent_id}_{task_type.value}_{int(datetime.now().timestamp())}"
        
        # Generate adaptation based on reason
        if reason == "low_performance":
            condition = f"agent_id == '{agent_id}' and task_type == '{task_type.value}'"
            action = "increase_caution_level"
        elif reason == "negative_feedback":
            condition = f"agent_id == '{agent_id}' and task_type == '{task_type.value}'"
            action = "adjust_response_style"
        else:
            condition = f"agent_id == '{agent_id}'"
            action = "general_improvement"
        
        rule = AdaptationRule(
            rule_id=rule_id,
            condition=condition,
            action=action,
            priority=2,
            confidence=0.6
        )
        
        self.adaptation_rules[rule_id] = rule
        self._save_adaptation_rule(rule)
    
    def _save_adaptation_rule(self, rule: AdaptationRule):
        """Save adaptation rule to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO adaptation_rules 
                (rule_id, condition, action, priority, confidence, usage_count, success_rate, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rule.rule_id,
                rule.condition,
                rule.action,
                rule.priority,
                rule.confidence,
                rule.usage_count,
                rule.success_rate,
                rule.created_at.isoformat()
            ))
            conn.commit()
    
    def get_agent_recommendations(self, agent_id: str, task_type: TaskType) -> Dict[str, Any]:
        """Get recommendations for agent behavior adaptation."""
        key = f"{agent_id}_{task_type.value}"
        
        recommendations = {
            "agent_id": agent_id,
            "task_type": task_type.value,
            "recommendations": [],
            "metrics": {}
        }
        
        # Add metrics if available
        if key in self.agent_metrics:
            metrics = self.agent_metrics[key]
            recommendations["metrics"] = {
                "total_interactions": metrics.total_interactions,
                "success_rate": metrics.successful_interactions / max(metrics.total_interactions, 1),
                "average_feedback_score": metrics.average_feedback_score,
                "average_response_time": metrics.average_response_time,
                "improvement_rate": metrics.improvement_rate
            }
        
        # Add relevant adaptation rules
        for rule in self.adaptation_rules.values():
            if agent_id in rule.condition and task_type.value in rule.condition:
                recommendations["recommendations"].append({
                    "rule_id": rule.rule_id,
                    "action": rule.action,
                    "confidence": rule.confidence,
                    "priority": rule.priority
                })
        
        return recommendations
    
    def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """Update user preferences for personalization."""
        self.user_preferences[user_id] = preferences
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO user_preferences (user_id, preferences_json, last_updated)
                VALUES (?, ?, ?)
            """, (user_id, json.dumps(preferences), datetime.now().isoformat()))
            conn.commit()
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get overall learning system statistics."""
        total_interactions = len(self.interaction_history)
        successful_interactions = sum(1 for r in self.interaction_history if r.success)
        
        avg_feedback = np.mean([r.feedback_score for r in self.interaction_history 
                               if r.feedback_score is not None]) if self.interaction_history else 0.0
        
        return {
            "total_interactions": total_interactions,
            "success_rate": successful_interactions / max(total_interactions, 1),
            "average_feedback_score": float(avg_feedback),
            "active_agents": len(set(r.agent_id for r in self.interaction_history)),
            "adaptation_rules": len(self.adaptation_rules),
            "user_preferences": len(self.user_preferences)
        } 