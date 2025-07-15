"""
API for empirical events and knowledge management.
Integrates with existing memory store architecture.
"""

import sqlite3
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path

from ..memory.stores import MemoryStore


class KnowledgeAPI:
    """
    API for managing empirical events, hypotheses, and knowledge.
    
    Features:
    - Event logging for all agent activities
    - Hypothesis lifecycle management
    - Knowledge promotion and validation
    - Advanced search and analytics
    """
    
    def __init__(self, memory_store: MemoryStore = None):
        """
        Initialize KnowledgeAPI.
        
        Args:
            memory_store: Optional existing memory store instance
        """
        if memory_store is None:
            from ..memory.stores import get_memory_store
            memory_store = get_memory_store()
        
        self.memory_store = memory_store
        self.db_path = memory_store.db_path
        
        # Ensure database is migrated
        self._ensure_migration()
    
    def _ensure_migration(self):
        """Ensure database has been migrated to support knowledge management."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if events table exists
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='events'
                """)
                
                if not cursor.fetchone():
                    # Run migration
                    from .migrations import run_migration
                    print("🔄 Running database migration for knowledge management...")
                    if not run_migration(self.db_path):
                        raise Exception("Database migration failed")
                    print("✅ Database migration completed")
                    
        except Exception as e:
            print(f"❌ Database migration check failed: {e}")
            raise
    
    # Event Logging Methods
    async def log_event(
        self,
        source: str,
        event_type: str,
        payload: Dict[str, Any],
        thread_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> int:
        """
        Log an empirical event in the system.
        
        Args:
            source: Agent or component that generated the event
            event_type: Type of event (calculation, hypothesis_proposed, etc.)
            payload: Event-specific data as dictionary
            thread_id: Optional conversation thread ID
            session_id: Optional collaboration session ID
            
        Returns:
            Event ID
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO events (source, event_type, payload_json, thread_id, session_id)
                    VALUES (?, ?, ?, ?, ?)
                """, (source, event_type, json.dumps(payload), thread_id, session_id))
                
                event_id = cursor.lastrowid
                conn.commit()
                
                return event_id
                
        except Exception as e:
            print(f"❌ Event logging failed: {e}")
            raise
    
    async def get_events_by_type(
        self,
        event_type: str,
        limit: int = 100,
        since: Optional[datetime] = None,
        source: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve events by type with optional filters.
        
        Args:
            event_type: Type of event to retrieve
            limit: Maximum number of events to return
            since: Optional datetime filter
            source: Optional source filter
            
        Returns:
            List of event dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = "SELECT * FROM events WHERE event_type = ?"
                params = [event_type]
                
                if since:
                    query += " AND timestamp >= ?"
                    params.append(since.isoformat())
                
                if source:
                    query += " AND source = ?"
                    params.append(source)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                events = cursor.fetchall()
                
                return [dict(event) for event in events]
                
        except Exception as e:
            print(f"❌ Event retrieval failed: {e}")
            return []
    
    async def get_events_by_session(
        self,
        session_id: str,
        event_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get all events for a specific session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = "SELECT * FROM events WHERE session_id = ?"
                params = [session_id]
                
                if event_types:
                    placeholders = ','.join(['?' for _ in event_types])
                    query += f" AND event_type IN ({placeholders})"
                    params.extend(event_types)
                
                query += " ORDER BY timestamp ASC"
                
                cursor.execute(query, params)
                events = cursor.fetchall()
                
                return [dict(event) for event in events]
                
        except Exception as e:
            print(f"❌ Session event retrieval failed: {e}")
            return []
    
    # Hypothesis Management Methods
    async def propose_hypothesis(
        self,
        statement: str,
        sympy_expr: Optional[str] = None,
        created_by: str = "system",
        thread_id: Optional[str] = None,
        session_id: Optional[str] = None,
        initial_confidence: float = 0.5,
        domain: Optional[str] = None
    ) -> int:
        """
        Propose a new hypothesis for evaluation.
        
        Args:
            statement: Natural language hypothesis statement
            sympy_expr: Optional mathematical expression
            created_by: Agent or user who created the hypothesis
            thread_id: Originating conversation thread
            session_id: Originating collaboration session
            initial_confidence: Starting confidence score
            domain: Physics domain (quantum, classical, etc.)
            
        Returns:
            Hypothesis ID
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO hypotheses (
                        statement, sympy_expr, confidence, created_by, 
                        thread_id, session_id, status
                    ) VALUES (?, ?, ?, ?, ?, ?, 'proposed')
                """, (statement, sympy_expr, initial_confidence, created_by, thread_id, session_id))
                
                hypothesis_id = cursor.lastrowid
                conn.commit()
                
                # Log the hypothesis proposal event
                await self.log_event(
                    source=created_by,
                    event_type="hypothesis_proposed",
                    payload={
                        "hypothesis_id": hypothesis_id,
                        "statement": statement,
                        "confidence": initial_confidence,
                        "domain": domain
                    },
                    thread_id=thread_id,
                    session_id=session_id
                )
                
                return hypothesis_id
                
        except Exception as e:
            print(f"❌ Hypothesis proposal failed: {e}")
            raise
    
    async def update_confidence(
        self,
        hypothesis_id: int,
        new_confidence: float,
        supporting_event_id: Optional[int] = None,
        reason: Optional[str] = None,
        updated_by: str = "system"
    ) -> bool:
        """
        Update hypothesis confidence based on new evidence.
        
        Args:
            hypothesis_id: ID of hypothesis to update
            new_confidence: New confidence score (0.0 to 1.0)
            supporting_event_id: Optional event that influenced confidence
            reason: Optional explanation for confidence change
            updated_by: Agent or user making the update
            
        Returns:
            Success status
        """
        try:
            if not (0.0 <= new_confidence <= 1.0):
                raise ValueError("Confidence must be between 0.0 and 1.0")
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get current confidence
                cursor.execute("SELECT confidence, support_ids_json FROM hypotheses WHERE id = ?", (hypothesis_id,))
                result = cursor.fetchone()
                
                if not result:
                    print(f"❌ Hypothesis {hypothesis_id} not found")
                    return False
                
                old_confidence, support_ids_json = result
                support_ids = json.loads(support_ids_json or '[]')
                
                # Add supporting event if provided
                if supporting_event_id and supporting_event_id not in support_ids:
                    support_ids.append(supporting_event_id)
                
                # Update hypothesis
                cursor.execute("""
                    UPDATE hypotheses 
                    SET confidence = ?, support_ids_json = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (new_confidence, json.dumps(support_ids), hypothesis_id))
                
                conn.commit()
                
                # Log the confidence update event
                await self.log_event(
                    source=updated_by,
                    event_type="confidence_updated",
                    payload={
                        "hypothesis_id": hypothesis_id,
                        "old_confidence": old_confidence,
                        "new_confidence": new_confidence,
                        "reason": reason,
                        "supporting_event_id": supporting_event_id
                    }
                )
                
                return True
                
        except Exception as e:
            print(f"❌ Confidence update failed: {e}")
            return False
    
    async def update_hypothesis_status(
        self,
        hypothesis_id: int,
        new_status: str,
        reason: Optional[str] = None,
        updated_by: str = "system"
    ) -> bool:
        """Update hypothesis status with validation."""
        valid_statuses = ['proposed', 'under_review', 'validated', 'refuted', 'promoted']
        
        if new_status not in valid_statuses:
            raise ValueError(f"Invalid status. Must be one of: {valid_statuses}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get current status
                cursor.execute("SELECT status FROM hypotheses WHERE id = ?", (hypothesis_id,))
                result = cursor.fetchone()
                
                if not result:
                    print(f"❌ Hypothesis {hypothesis_id} not found")
                    return False
                
                old_status = result[0]
                
                # Update status
                cursor.execute("""
                    UPDATE hypotheses 
                    SET status = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (new_status, hypothesis_id))
                
                conn.commit()
                
                # Log the status update event
                await self.log_event(
                    source=updated_by,
                    event_type="hypothesis_status_updated",
                    payload={
                        "hypothesis_id": hypothesis_id,
                        "old_status": old_status,
                        "new_status": new_status,
                        "reason": reason
                    }
                )
                
                return True
                
        except Exception as e:
            print(f"❌ Status update failed: {e}")
            return False
    
    async def promote_to_knowledge(
        self,
        hypothesis_id: int,
        validation_events: List[int],
        domain: Optional[str] = None,
        promoted_by: str = "system"
    ) -> int:
        """
        Promote validated hypothesis to knowledge base.
        
        Args:
            hypothesis_id: ID of hypothesis to promote
            validation_events: List of event IDs that validate the hypothesis
            domain: Physics domain classification
            promoted_by: Agent or user promoting the hypothesis
            
        Returns:
            Knowledge ID
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get hypothesis details
                cursor.execute("""
                    SELECT statement, sympy_expr, confidence, created_by, created_at
                    FROM hypotheses WHERE id = ?
                """, (hypothesis_id,))
                
                result = cursor.fetchone()
                if not result:
                    raise ValueError(f"Hypothesis {hypothesis_id} not found")
                
                statement, sympy_expr, confidence, created_by, created_at = result
                
                # Create provenance information
                provenance = {
                    "hypothesis_id": hypothesis_id,
                    "validation_events": validation_events,
                    "promoted_by": promoted_by,
                    "promoted_at": datetime.now().isoformat(),
                    "original_creator": created_by,
                    "original_confidence": confidence,
                    "validation_count": len(validation_events)
                }
                
                # Insert into knowledge table
                cursor.execute("""
                    INSERT INTO knowledge (statement, sympy_expr, provenance_json, domain, confidence)
                    VALUES (?, ?, ?, ?, ?)
                """, (statement, sympy_expr, json.dumps(provenance), domain, confidence))
                
                knowledge_id = cursor.lastrowid
                
                # Update hypothesis status to promoted
                cursor.execute("""
                    UPDATE hypotheses 
                    SET status = 'promoted', updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (hypothesis_id,))
                
                # Update knowledge FTS index
                cursor.execute("""
                    INSERT INTO knowledge_fts (statement, domain)
                    VALUES (?, ?)
                """, (statement, domain or ""))
                
                conn.commit()
                
                # Log the promotion event
                await self.log_event(
                    source=promoted_by,
                    event_type="hypothesis_promoted",
                    payload={
                        "hypothesis_id": hypothesis_id,
                        "knowledge_id": knowledge_id,
                        "domain": domain,
                        "validation_events": validation_events,
                        "final_confidence": confidence
                    }
                )
                
                return knowledge_id
                
        except Exception as e:
            print(f"❌ Knowledge promotion failed: {e}")
            raise
    
    # Retrieval Methods
    async def get_hypotheses_by_status(
        self,
        status: str,
        confidence_threshold: float = 0.0,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve hypotheses by status and confidence.
        
        Args:
            status: Hypothesis status to filter by
            confidence_threshold: Minimum confidence score
            limit: Maximum number of results
            
        Returns:
            List of hypothesis dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM hypotheses 
                    WHERE status = ? AND confidence >= ?
                    ORDER BY confidence DESC, created_at DESC
                    LIMIT ?
                """, (status, confidence_threshold, limit))
                
                hypotheses = cursor.fetchall()
                
                result = []
                for hyp in hypotheses:
                    hyp_dict = dict(hyp)
                    hyp_dict['support_ids'] = json.loads(hyp_dict['support_ids_json'] or '[]')
                    result.append(hyp_dict)
                
                return result
                
        except Exception as e:
            print(f"❌ Hypothesis retrieval failed: {e}")
            return []
    
    async def search_knowledge(
        self,
        query: str,
        domain: Optional[str] = None,
        limit: int = 10,
        confidence_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search knowledge base using full-text search.
        
        Args:
            query: Search query string
            domain: Optional domain filter
            limit: Maximum number of results
            confidence_threshold: Minimum confidence score
            
        Returns:
            List of knowledge dictionaries with relevance scores
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Use FTS for search
                fts_query = query.replace("'", "''")  # Escape quotes
                
                base_query = """
                    SELECT k.*, k.rowid as knowledge_id, 
                           fts.rank as relevance_score
                    FROM knowledge k
                    JOIN knowledge_fts fts ON k.rowid = fts.rowid
                    WHERE knowledge_fts MATCH ?
                    AND k.confidence >= ?
                """
                
                params = [fts_query, confidence_threshold]
                
                if domain:
                    base_query += " AND k.domain = ?"
                    params.append(domain)
                
                base_query += " ORDER BY fts.rank LIMIT ?"
                params.append(limit)
                
                cursor.execute(base_query, params)
                results = cursor.fetchall()
                
                knowledge_list = []
                for result in results:
                    knowledge_dict = dict(result)
                    knowledge_dict['provenance'] = json.loads(knowledge_dict['provenance_json'] or '{}')
                    knowledge_list.append(knowledge_dict)
                
                return knowledge_list
                
        except Exception as e:
            print(f"❌ Knowledge search failed: {e}")
            return []
    
    async def get_knowledge_by_domain(
        self,
        domain: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get all knowledge entries for a specific domain."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM knowledge 
                    WHERE domain = ?
                    ORDER BY confidence DESC, created_at DESC
                    LIMIT ?
                """, (domain, limit))
                
                results = cursor.fetchall()
                
                knowledge_list = []
                for result in results:
                    knowledge_dict = dict(result)
                    knowledge_dict['provenance'] = json.loads(knowledge_dict['provenance_json'] or '{}')
                    knowledge_list.append(knowledge_dict)
                
                return knowledge_list
                
        except Exception as e:
            print(f"❌ Domain knowledge retrieval failed: {e}")
            return []
    
    def get_all_knowledge(self) -> List[Dict[str, Any]]:
        """Get all knowledge entries."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, statement, sympy_expr, provenance_json, created_at, domain, confidence
                    FROM knowledge
                    ORDER BY created_at DESC
                """)
                
                knowledge_entries = []
                for row in cursor.fetchall():
                    entry = {
                        'id': row[0],
                        'statement': row[1],
                        'sympy_expr': row[2],
                        'provenance': json.loads(row[3]) if row[3] else {},
                        'created_at': row[4],
                        'domain': row[5],
                        'confidence': row[6],
                        'title': row[1][:100] + '...' if len(row[1]) > 100 else row[1],
                        'content': row[1],
                        'type': 'knowledge'
                    }
                    knowledge_entries.append(entry)
                
                return knowledge_entries
                
        except Exception as e:
            print(f"❌ Failed to get all knowledge: {e}")
            return []
    
    # Analytics Methods
    async def get_hypothesis_analytics(
        self,
        hypothesis_id: int
    ) -> Dict[str, Any]:
        """
        Get detailed analytics for a specific hypothesis.
        
        Args:
            hypothesis_id: ID of hypothesis to analyze
            
        Returns:
            Dictionary with analytics data
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Get hypothesis details
                cursor.execute("SELECT * FROM hypotheses WHERE id = ?", (hypothesis_id,))
                hypothesis = cursor.fetchone()
                
                if not hypothesis:
                    return {"error": "Hypothesis not found"}
                
                # Get supporting events
                support_ids = json.loads(hypothesis['support_ids_json'] or '[]')
                supporting_events = []
                
                if support_ids:
                    placeholders = ','.join(['?' for _ in support_ids])
                    cursor.execute(f"""
                        SELECT * FROM events 
                        WHERE id IN ({placeholders})
                        ORDER BY timestamp ASC
                    """, support_ids)
                    supporting_events = [dict(event) for event in cursor.fetchall()]
                
                # Get confidence history (from events)
                cursor.execute("""
                    SELECT timestamp, payload_json FROM events
                    WHERE event_type = 'confidence_updated'
                    AND JSON_EXTRACT(payload_json, '$.hypothesis_id') = ?
                    ORDER BY timestamp ASC
                """, (hypothesis_id,))
                
                confidence_history = []
                for event in cursor.fetchall():
                    payload = json.loads(event[0])
                    confidence_history.append({
                        "timestamp": event[1],
                        "confidence": payload.get('new_confidence'),
                        "reason": payload.get('reason')
                    })
                
                return {
                    "hypothesis": dict(hypothesis),
                    "supporting_events": supporting_events,
                    "confidence_history": confidence_history,
                    "support_count": len(supporting_events),
                    "age_days": (datetime.now() - datetime.fromisoformat(hypothesis['created_at'])).days
                }
                
        except Exception as e:
            print(f"❌ Hypothesis analytics failed: {e}")
            return {"error": str(e)}
    
    async def get_agent_performance_metrics(
        self,
        agent_name: str,
        time_window: int = 7  # days
    ) -> Dict[str, Any]:
        """
        Get performance metrics for a specific agent.
        
        Args:
            agent_name: Name of the agent
            time_window: Time window in days
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            since_date = datetime.now() - timedelta(days=time_window)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Get event counts by type
                cursor.execute("""
                    SELECT event_type, COUNT(*) as count
                    FROM events
                    WHERE source = ? AND timestamp >= ?
                    GROUP BY event_type
                """, (agent_name, since_date.isoformat()))
                
                event_counts = {row['event_type']: row['count'] for row in cursor.fetchall()}
                
                # Get hypotheses proposed
                cursor.execute("""
                    SELECT COUNT(*) as count, AVG(confidence) as avg_confidence
                    FROM hypotheses
                    WHERE created_by = ? AND created_at >= ?
                """, (agent_name, since_date.isoformat()))
                
                hyp_stats = cursor.fetchone()
                
                # Get knowledge contributed (promoted hypotheses)
                cursor.execute("""
                    SELECT COUNT(*) as count
                    FROM knowledge k
                    WHERE JSON_EXTRACT(k.provenance_json, '$.promoted_by') = ?
                    AND k.created_at >= ?
                """, (agent_name, since_date.isoformat()))
                
                knowledge_count = cursor.fetchone()['count']
                
                return {
                    "agent_name": agent_name,
                    "time_window_days": time_window,
                    "event_counts": event_counts,
                    "total_events": sum(event_counts.values()),
                    "hypotheses_proposed": hyp_stats['count'] or 0,
                    "avg_hypothesis_confidence": hyp_stats['avg_confidence'] or 0.0,
                    "knowledge_contributed": knowledge_count,
                    "activity_score": sum(event_counts.values()) / time_window
                }
                
        except Exception as e:
            print(f"❌ Agent performance metrics failed: {e}")
            return {"error": str(e)}
    
    async def get_system_analytics(self) -> Dict[str, Any]:
        """Get overall system analytics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Total counts
                cursor.execute("SELECT COUNT(*) as count FROM events")
                total_events = cursor.fetchone()['count']
                
                cursor.execute("SELECT COUNT(*) as count FROM hypotheses")
                total_hypotheses = cursor.fetchone()['count']
                
                cursor.execute("SELECT COUNT(*) as count FROM knowledge")
                total_knowledge = cursor.fetchone()['count']
                
                # Status distribution
                cursor.execute("""
                    SELECT status, COUNT(*) as count
                    FROM hypotheses
                    GROUP BY status
                """)
                status_distribution = {row['status']: row['count'] for row in cursor.fetchall()}
                
                # Domain distribution
                cursor.execute("""
                    SELECT domain, COUNT(*) as count
                    FROM knowledge
                    WHERE domain IS NOT NULL
                    GROUP BY domain
                """)
                domain_distribution = {row['domain']: row['count'] for row in cursor.fetchall()}
                
                # Promotion rate
                promoted_count = status_distribution.get('promoted', 0)
                promotion_rate = promoted_count / total_hypotheses if total_hypotheses > 0 else 0
                
                return {
                    "total_events": total_events,
                    "total_hypotheses": total_hypotheses,
                    "total_knowledge": total_knowledge,
                    "status_distribution": status_distribution,
                    "domain_distribution": domain_distribution,
                    "promotion_rate": promotion_rate,
                    "avg_confidence": await self._get_average_confidence()
                }
                
        except Exception as e:
            print(f"❌ System analytics failed: {e}")
            return {"error": str(e)}
    
    async def _get_average_confidence(self) -> float:
        """Get average confidence across all hypotheses."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT AVG(confidence) as avg_conf FROM hypotheses")
                result = cursor.fetchone()
                return result[0] if result[0] is not None else 0.0
                
        except Exception:
            return 0.0


# Convenience function for getting KnowledgeAPI instance
def get_knowledge_api() -> KnowledgeAPI:
    """Get or create the global KnowledgeAPI instance."""
    global _knowledge_api
    if '_knowledge_api' not in globals():
        _knowledge_api = KnowledgeAPI()
    return _knowledge_api 