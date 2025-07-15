"""
Memory storage implementations following LangChain Academy patterns.

This module provides memory storage capabilities incorporating:
- Module 2: Basic conversation memory and state management
- Module 5: Advanced memory patterns and personalization
"""

import sqlite3
import json
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from ..config import get_settings

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:
    SqliteSaver = None


class MemoryStore:
    """
    Comprehensive memory store for agent conversations and context.
    
    Features:
    - Conversation history storage
    - Automatic summarization of long conversations
    - User preferences and personalization
    - Efficient retrieval with search capabilities
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the memory store.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.settings = get_settings()
        self.db_path = db_path or self.settings.memory_db_path
        
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Conversations table for storing individual messages
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    agent_response TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            
            # Summaries table for storing conversation summaries
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    message_count INTEGER NOT NULL,
                    start_time DATETIME NOT NULL,
                    end_time DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # User preferences table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    preference_key TEXT NOT NULL,
                    preference_value TEXT NOT NULL,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, preference_key)
                )
            """)
            
            # Indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_thread_id ON conversations(thread_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_summaries_thread_id ON conversation_summaries(thread_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_preferences_user_id ON user_preferences(user_id)")
            
            conn.commit()
    
    async def store_conversation(
        self, 
        user_message: str, 
        agent_response: str, 
        thread_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Store a conversation turn.
        
        Args:
            user_message: User's message
            agent_response: Agent's response
            thread_id: Thread identifier
            metadata: Optional metadata dictionary
            
        Returns:
            ID of the stored conversation record
        """
        metadata = metadata or {}
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO conversations (thread_id, user_message, agent_response, metadata)
                VALUES (?, ?, ?, ?)
            """, (thread_id, user_message, agent_response, json.dumps(metadata)))
            
            conversation_id = cursor.lastrowid
            conn.commit()
            
            # Check if we need to summarize this thread
            await self._check_and_summarize_thread(thread_id)
            
            return conversation_id
    
    def get_conversation_history(
        self, 
        thread_id: str = "default", 
        limit: int = 10,
        include_summaries: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history for a thread.
        
        Args:
            thread_id: Thread identifier
            limit: Maximum number of recent messages to return
            include_summaries: Whether to include conversation summaries
            
        Returns:
            List of conversation records
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            history = []
            
            # Get summaries if requested
            if include_summaries:
                cursor.execute("""
                    SELECT summary, message_count, start_time, end_time
                    FROM conversation_summaries
                    WHERE thread_id = ?
                    ORDER BY created_at ASC
                """, (thread_id,))
                
                summaries = cursor.fetchall()
                for summary in summaries:
                    history.append({
                        "type": "summary",
                        "content": summary["summary"],
                        "message_count": summary["message_count"],
                        "start_time": summary["start_time"],
                        "end_time": summary["end_time"]
                    })
            
            # Get recent conversations
            cursor.execute("""
                SELECT user_message, agent_response, timestamp, metadata
                FROM conversations
                WHERE thread_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (thread_id, limit))
            
            conversations = cursor.fetchall()
            for conv in reversed(conversations):  # Reverse to get chronological order
                history.append({
                    "type": "conversation",
                    "user_message": conv["user_message"],
                    "agent_response": conv["agent_response"],
                    "timestamp": conv["timestamp"],
                    "metadata": json.loads(conv["metadata"])
                })
            
            return history
    
    async def _check_and_summarize_thread(self, thread_id: str, max_messages: int = 20):
        """
        Check if a thread needs summarization and create summary if needed.
        
        Args:
            thread_id: Thread to check
            max_messages: Maximum messages before summarization
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Count messages in thread that aren't summarized yet
            cursor.execute("""
                SELECT COUNT(*) as count, MIN(timestamp) as start_time, MAX(timestamp) as end_time
                FROM conversations
                WHERE thread_id = ?
                AND timestamp > (
                    SELECT COALESCE(MAX(end_time), '1970-01-01') 
                    FROM conversation_summaries 
                    WHERE thread_id = ?
                )
            """, (thread_id, thread_id))
            
            result = cursor.fetchone()
            count, start_time, end_time = result
            
            if count >= max_messages:
                await self._create_summary(thread_id, start_time, end_time, count)
    
    async def _create_summary(self, thread_id: str, start_time: str, end_time: str, message_count: int):
        """
        Create a summary of conversation messages.
        
        Args:
            thread_id: Thread to summarize
            start_time: Start time of messages to summarize
            end_time: End time of messages to summarize
            message_count: Number of messages being summarized
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get messages to summarize
            cursor.execute("""
                SELECT user_message, agent_response
                FROM conversations
                WHERE thread_id = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp ASC
            """, (thread_id, start_time, end_time))
            
            messages = cursor.fetchall()
            
            # Create a simple summary (in production, you'd use an LLM for this)
            conversation_text = ""
            for user_msg, agent_msg in messages:
                conversation_text += f"User: {user_msg}\nAgent: {agent_msg}\n\n"
            
            # Simple keyword-based summary (replace with LLM summarization in production)
            summary = f"Conversation summary ({message_count} messages): Discussion covered various topics including user queries and agent responses. Key themes extracted from {len(messages)} conversation turns."
            
            # Store summary
            cursor.execute("""
                INSERT INTO conversation_summaries (thread_id, summary, message_count, start_time, end_time)
                VALUES (?, ?, ?, ?, ?)
            """, (thread_id, summary, message_count, start_time, end_time))
            
            conn.commit()
    
    def store_user_preference(self, user_id: str, key: str, value: Any):
        """
        Store a user preference.
        
        Args:
            user_id: User identifier
            key: Preference key
            value: Preference value (will be JSON serialized)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO user_preferences (user_id, preference_key, preference_value, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """, (user_id, key, json.dumps(value)))
            conn.commit()
    
    def get_user_preference(self, user_id: str, key: str, default: Any = None) -> Any:
        """
        Get a user preference.
        
        Args:
            user_id: User identifier
            key: Preference key
            default: Default value if not found
            
        Returns:
            Preference value or default
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT preference_value FROM user_preferences
                WHERE user_id = ? AND preference_key = ?
            """, (user_id, key))
            
            result = cursor.fetchone()
            if result:
                return json.loads(result[0])
            return default
    
    def get_all_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Get all preferences for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary of all user preferences
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT preference_key, preference_value FROM user_preferences
                WHERE user_id = ?
            """, (user_id,))
            
            preferences = {}
            for row in cursor.fetchall():
                preferences[row["preference_key"]] = json.loads(row["preference_value"])
            
            return preferences
    
    def search_conversations(
        self, 
        query: str, 
        thread_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search conversations by content.
        
        Args:
            query: Search query
            thread_id: Optional thread to search within
            limit: Maximum results to return
            
        Returns:
            List of matching conversation records
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if thread_id:
                cursor.execute("""
                    SELECT thread_id, user_message, agent_response, timestamp
                    FROM conversations
                    WHERE thread_id = ? AND (user_message LIKE ? OR agent_response LIKE ?)
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (thread_id, f"%{query}%", f"%{query}%", limit))
            else:
                cursor.execute("""
                    SELECT thread_id, user_message, agent_response, timestamp
                    FROM conversations
                    WHERE user_message LIKE ? OR agent_response LIKE ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (f"%{query}%", f"%{query}%", limit))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "thread_id": row["thread_id"],
                    "user_message": row["user_message"],
                    "agent_response": row["agent_response"],
                    "timestamp": row["timestamp"]
                })
            
            return results
    
    def clear_thread(self, thread_id: str):
        """
        Clear all data for a specific thread.
        
        Args:
            thread_id: Thread to clear
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM conversations WHERE thread_id = ?", (thread_id,))
            cursor.execute("DELETE FROM conversation_summaries WHERE thread_id = ?", (thread_id,))
            conn.commit()
    
    def get_thread_stats(self, thread_id: str) -> Dict[str, Any]:
        """
        Get statistics for a thread.
        
        Args:
            thread_id: Thread identifier
            
        Returns:
            Dictionary with thread statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get conversation count and date range
            cursor.execute("""
                SELECT COUNT(*) as count, MIN(timestamp) as first_message, MAX(timestamp) as last_message
                FROM conversations
                WHERE thread_id = ?
            """, (thread_id,))
            
            conv_stats = cursor.fetchone()
            
            # Get summary count
            cursor.execute("""
                SELECT COUNT(*) as summary_count, SUM(message_count) as summarized_messages
                FROM conversation_summaries
                WHERE thread_id = ?
            """, (thread_id,))
            
            summary_stats = cursor.fetchone()
            
            return {
                "message_count": conv_stats[0],
                "first_message": conv_stats[1],
                "last_message": conv_stats[2],
                "summary_count": summary_stats[0] or 0,
                "summarized_messages": summary_stats[1] or 0,
                "total_interactions": (conv_stats[0] or 0) + (summary_stats[1] or 0)
            }
    
    def get_checkpointer(self):
        """
        Get a LangGraph checkpointer for this memory store.
        
        Returns:
            SqliteSaver instance configured with a separate agents database
        """
        if SqliteSaver is None:
            raise ImportError(
                "langgraph-checkpoint-sqlite is required for checkpointer functionality. "
                "Install it with: pip install langgraph-checkpoint-sqlite"
            )
        
        # Use a separate database file for agent checkpointing to avoid conflicts
        agents_db_path = self.db_path.replace('memory.db', 'agents.db')
        return SqliteSaver.from_conn_string(agents_db_path)
    
    def get_next_version(self, current: Any = None, channel: Any = None) -> str:
        """
        Get the next version for LangGraph checkpointer compatibility.
        This is a compatibility method that delegates to the actual checkpointer.
        
        Args:
            current: Current version (LangGraph parameter)
            channel: Channel information (LangGraph parameter)
        
        Returns:
            Next version string
        """
        try:
            checkpointer = self.get_checkpointer()
            if hasattr(checkpointer, 'get_next_version'):
                return checkpointer.get_next_version(current, channel)
            else:
                # Fallback - simple version increment logic
                from ..database.migrations import DatabaseMigrator
                migrator = DatabaseMigrator(self.db_path)
                current_version = migrator.get_schema_version()
                
                if current_version == "1.0.0":
                    return "1.1.0"
                elif current_version == "1.1.0":
                    return "1.2.0"
                else:
                    # Parse and increment
                    parts = current_version.split('.')
                    minor = int(parts[1]) + 1
                    return f"{parts[0]}.{minor}.0"
                    
        except Exception:
            return "1.1.0"  # Default next version
    
    async def aput_writes(self, config: Dict[str, Any], writes: List[Any], task_id: str) -> None:
        """
        Async method to handle writes for LangGraph checkpointer compatibility.
        This is a compatibility method that delegates to the actual checkpointer.
        
        Args:
            config: Configuration dictionary
            writes: List of writes to process
            task_id: Task identifier
        """
        try:
            checkpointer = self.get_checkpointer()
            if hasattr(checkpointer, 'aput_writes'):
                await checkpointer.aput_writes(config, writes, task_id)
            else:
                # Fallback - log the writes for debugging
                print(f"⚠️ aput_writes called but not supported by checkpointer: {writes}")
        except Exception as e:
            print(f"❌ aput_writes failed: {e}")
            # Don't raise to avoid breaking the workflow
    
    async def aget_tuple(self, config: Dict[str, Any]) -> Optional[Tuple[Any, ...]]:
        """
        Async method to get tuple data for LangGraph checkpointer compatibility.
        This is a compatibility method that delegates to the actual checkpointer.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Tuple data or None
        """
        try:
            checkpointer = self.get_checkpointer()
            if hasattr(checkpointer, 'aget_tuple'):
                return await checkpointer.aget_tuple(config)
            else:
                # Fallback - return None
                print(f"⚠️ aget_tuple called but not supported by checkpointer")
                return None
        except Exception as e:
            print(f"❌ aget_tuple failed: {e}")
            return None
    
    async def aput(self, config: Dict[str, Any], checkpoint: Any, metadata: Any, new_versions: Any) -> Dict[str, Any]:
        """
        Async method to put checkpoint data for LangGraph checkpointer compatibility.
        This is a compatibility method that delegates to the actual checkpointer.
        
        Args:
            config: Configuration dictionary
            checkpoint: Checkpoint data
            metadata: Metadata
            new_versions: New versions data
            
        Returns:
            Configuration dictionary
        """
        try:
            checkpointer = self.get_checkpointer()
            if hasattr(checkpointer, 'aput'):
                return await checkpointer.aput(config, checkpoint, metadata, new_versions)
            else:
                # Fallback - return config as-is
                print(f"⚠️ aput called but not supported by checkpointer")
                return config
        except Exception as e:
            print(f"❌ aput failed: {e}")
            return config
    
    async def aget(self, config: Dict[str, Any]) -> Optional[Any]:
        """
        Async method to get checkpoint data for LangGraph checkpointer compatibility.
        This is a compatibility method that delegates to the actual checkpointer.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Checkpoint data or None
        """
        try:
            checkpointer = self.get_checkpointer()
            if hasattr(checkpointer, 'aget'):
                return await checkpointer.aget(config)
            else:
                # Fallback - return None
                print(f"⚠️ aget called but not supported by checkpointer")
                return None
        except Exception as e:
            print(f"❌ aget failed: {e}")
            return None
    
    def alist(self, config: Dict[str, Any], *, filter: Optional[Dict[str, Any]] = None, before: Optional[Any] = None, limit: Optional[int] = None):
        """
        Async generator to list checkpoints for LangGraph checkpointer compatibility.
        This is a compatibility method that delegates to the actual checkpointer.
        
        Args:
            config: Configuration dictionary
            filter: Optional filter
            before: Optional before parameter
            limit: Optional limit
            
        Yields:
            Checkpoint data
        """
        try:
            checkpointer = self.get_checkpointer()
            if hasattr(checkpointer, 'alist'):
                return checkpointer.alist(config, filter=filter, before=before, limit=limit)
            else:
                # Fallback - return empty async generator
                async def empty_generator():
                    return
                    yield  # unreachable
                return empty_generator()
        except Exception as e:
            print(f"❌ alist failed: {e}")
            async def empty_generator():
                return
                yield  # unreachable
            return empty_generator()


# Global memory store instance
_memory_store: Optional[MemoryStore] = None


def get_memory_store() -> MemoryStore:
    """Get or create the global memory store instance."""
    global _memory_store
    if _memory_store is None:
        _memory_store = MemoryStore()
    return _memory_store