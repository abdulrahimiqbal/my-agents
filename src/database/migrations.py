"""
Database migration script for empirical events and knowledge management.
Ensures zero data loss and backward compatibility.
"""

import sqlite3
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

class DatabaseMigrator:
    """
    Database migration manager for adding empirical events and knowledge management.
    
    Features:
    - Automatic backup before migration
    - Rollback capability
    - Zero data loss guarantee
    - Schema version tracking
    """
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.backup_path = self.db_path.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.db')
        self.schema_version = "1.1.0"  # Version with knowledge management
        
    def migrate(self) -> bool:
        """
        Execute migration with rollback capability.
        
        Returns:
            True if migration successful, False otherwise
        """
        try:
            print(f"🔄 Starting database migration to version {self.schema_version}")
            
            # 1. Create backup
            if not self.create_backup():
                print("❌ Backup creation failed")
                return False
            
            # 2. Check existing schema
            current_version = self.get_schema_version()
            print(f"📊 Current schema version: {current_version}")
            
            if current_version == self.schema_version:
                print("✅ Database already at target version")
                return True
            
            # 3. Create new tables if absent
            if not self.create_new_tables():
                print("❌ Table creation failed")
                self.rollback()
                return False
            
            # 4. Migrate existing data
            if not self.migrate_existing_data():
                print("❌ Data migration failed")
                self.rollback()
                return False
            
            # 5. Update schema version
            self.update_schema_version()
            
            # 6. Verify migration success
            if not self.verify_migration():
                print("❌ Migration verification failed")
                self.rollback()
                return False
            
            print(f"✅ Migration to version {self.schema_version} completed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Migration failed with error: {e}")
            self.rollback()
            return False
    
    def create_backup(self) -> bool:
        """Create full database backup before migration."""
        try:
            if not self.db_path.exists():
                print("📝 Database doesn't exist yet, creating new one")
                return True
            
            print(f"💾 Creating backup: {self.backup_path}")
            shutil.copy2(self.db_path, self.backup_path)
            
            # Verify backup integrity
            if not self.verify_backup():
                print("❌ Backup verification failed")
                return False
            
            print("✅ Backup created successfully")
            return True
            
        except Exception as e:
            print(f"❌ Backup creation failed: {e}")
            return False
    
    def verify_backup(self) -> bool:
        """Verify backup integrity by checking table count."""
        try:
            with sqlite3.connect(self.db_path) as original_conn:
                original_cursor = original_conn.cursor()
                original_cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                original_count = original_cursor.fetchone()[0]
            
            with sqlite3.connect(self.backup_path) as backup_conn:
                backup_cursor = backup_conn.cursor()
                backup_cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                backup_count = backup_cursor.fetchone()[0]
            
            return original_count == backup_count
            
        except Exception as e:
            print(f"❌ Backup verification failed: {e}")
            return False
    
    def get_schema_version(self) -> str:
        """Get current schema version."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if schema_version table exists
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='schema_version'
                """)
                
                if not cursor.fetchone():
                    return "1.0.0"  # Original version
                
                cursor.execute("SELECT version FROM schema_version ORDER BY id DESC LIMIT 1")
                result = cursor.fetchone()
                return result[0] if result else "1.0.0"
                
        except Exception:
            return "1.0.0"
    
    def create_new_tables(self) -> bool:
        """Create new tables with indexes and constraints."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create schema_version table if it doesn't exist
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS schema_version (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        version TEXT NOT NULL,
                        applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create events table
                if not self.table_exists('events'):
                    print("📊 Creating events table...")
                    cursor.execute("""
                        CREATE TABLE events (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                            source TEXT NOT NULL,
                            event_type TEXT NOT NULL,
                            payload_json TEXT NOT NULL DEFAULT '{}',
                            thread_id TEXT,
                            session_id TEXT
                        )
                    """)
                    
                    # Create indexes for events
                    cursor.execute("CREATE INDEX idx_events_timestamp ON events(timestamp)")
                    cursor.execute("CREATE INDEX idx_events_source ON events(source)")
                    cursor.execute("CREATE INDEX idx_events_type ON events(event_type)")
                    cursor.execute("CREATE INDEX idx_events_thread_id ON events(thread_id)")
                    cursor.execute("CREATE INDEX idx_events_session_id ON events(session_id)")
                    
                    print("✅ Events table created")
                
                # Create hypotheses table
                if not self.table_exists('hypotheses'):
                    print("🧪 Creating hypotheses table...")
                    cursor.execute("""
                        CREATE TABLE hypotheses (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            statement TEXT NOT NULL,
                            sympy_expr TEXT,
                            support_ids_json TEXT DEFAULT '[]',
                            confidence REAL DEFAULT 0.5,
                            status TEXT DEFAULT 'proposed',
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                            created_by TEXT,
                            thread_id TEXT,
                            session_id TEXT,
                            CHECK (confidence >= 0.0 AND confidence <= 1.0),
                            CHECK (status IN ('proposed', 'under_review', 'validated', 'refuted', 'promoted'))
                        )
                    """)
                    
                    # Create indexes for hypotheses
                    cursor.execute("CREATE INDEX idx_hypotheses_status ON hypotheses(status)")
                    cursor.execute("CREATE INDEX idx_hypotheses_confidence ON hypotheses(confidence)")
                    cursor.execute("CREATE INDEX idx_hypotheses_created_at ON hypotheses(created_at)")
                    cursor.execute("CREATE INDEX idx_hypotheses_thread_id ON hypotheses(thread_id)")
                    
                    print("✅ Hypotheses table created")
                
                # Create knowledge table
                if not self.table_exists('knowledge'):
                    print("📚 Creating knowledge table...")
                    cursor.execute("""
                        CREATE TABLE knowledge (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            statement TEXT NOT NULL,
                            sympy_expr TEXT,
                            provenance_json TEXT DEFAULT '{}',
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                            domain TEXT,
                            confidence REAL DEFAULT 1.0
                        )
                    """)
                    
                    # Create indexes for knowledge
                    cursor.execute("CREATE INDEX idx_knowledge_domain ON knowledge(domain)")
                    cursor.execute("CREATE INDEX idx_knowledge_created_at ON knowledge(created_at)")
                    cursor.execute("CREATE INDEX idx_knowledge_confidence ON knowledge(confidence)")
                    
                    print("✅ Knowledge table created")
                
                # Create FTS table for knowledge search
                if not self.table_exists('knowledge_fts'):
                    print("🔍 Creating knowledge full-text search...")
                    cursor.execute("""
                        CREATE VIRTUAL TABLE knowledge_fts USING fts5(
                            statement, 
                            domain, 
                            content='knowledge', 
                            content_rowid='id'
                        )
                    """)
                    
                    print("✅ Knowledge FTS table created")
                
                # Create knowledge graph tables for Phase 3
                if not self.table_exists('concepts'):
                    print("🧠 Creating concepts table...")
                    cursor.execute("""
                        CREATE TABLE concepts (
                            id TEXT PRIMARY KEY,
                            name TEXT NOT NULL,
                            concept_type TEXT NOT NULL,
                            domain TEXT NOT NULL,
                            description TEXT,
                            mathematical_representation TEXT,
                            units TEXT,
                            typical_values TEXT,  -- JSON
                            historical_context TEXT,
                            applications TEXT,  -- JSON
                            related_experiments TEXT,  -- JSON
                            difficulty_level TEXT DEFAULT 'undergraduate',
                            confidence_score REAL DEFAULT 1.0,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            metadata TEXT  -- JSON
                        )
                    """)
                    
                    # Create indexes for concepts
                    cursor.execute("CREATE INDEX idx_concepts_domain ON concepts(domain)")
                    cursor.execute("CREATE INDEX idx_concepts_type ON concepts(concept_type)")
                    cursor.execute("CREATE INDEX idx_concepts_difficulty ON concepts(difficulty_level)")
                    cursor.execute("CREATE INDEX idx_concepts_confidence ON concepts(confidence_score)")
                    
                    print("✅ Concepts table created")
                
                if not self.table_exists('relationships'):
                    print("🔗 Creating relationships table...")
                    cursor.execute("""
                        CREATE TABLE relationships (
                            id TEXT PRIMARY KEY,
                            source_id TEXT NOT NULL,
                            target_id TEXT NOT NULL,
                            relationship_type TEXT NOT NULL,
                            strength REAL NOT NULL,
                            confidence REAL NOT NULL,
                            description TEXT,
                            evidence TEXT,  -- JSON
                            context TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            metadata TEXT,  -- JSON
                            FOREIGN KEY (source_id) REFERENCES concepts (id),
                            FOREIGN KEY (target_id) REFERENCES concepts (id)
                        )
                    """)
                    
                    # Create indexes for relationships
                    cursor.execute("CREATE INDEX idx_relationships_source ON relationships(source_id)")
                    cursor.execute("CREATE INDEX idx_relationships_target ON relationships(target_id)")
                    cursor.execute("CREATE INDEX idx_relationships_type ON relationships(relationship_type)")
                    cursor.execute("CREATE INDEX idx_relationships_strength ON relationships(strength)")
                    cursor.execute("CREATE INDEX idx_relationships_confidence ON relationships(confidence)")
                    
                    print("✅ Relationships table created")
                
                if not self.table_exists('graph_snapshots'):
                    print("📸 Creating graph snapshots table...")
                    cursor.execute("""
                        CREATE TABLE graph_snapshots (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            snapshot_hash TEXT UNIQUE,
                            graph_data BLOB,  -- Pickled NetworkX graph
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            description TEXT
                        )
                    """)
                    
                    # Create indexes for graph snapshots
                    cursor.execute("CREATE INDEX idx_graph_snapshots_hash ON graph_snapshots(snapshot_hash)")
                    cursor.execute("CREATE INDEX idx_graph_snapshots_created_at ON graph_snapshots(created_at)")
                    
                    print("✅ Graph snapshots table created")
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"❌ Table creation failed: {e}")
            return False
    
    def table_exists(self, table_name: str) -> bool:
        """Check if table already exists."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name=?
                """, (table_name,))
                return cursor.fetchone() is not None
                
        except Exception:
            return False
    
    def migrate_existing_data(self) -> bool:
        """Migrate existing conversation data to events table."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if conversations table exists
                if not self.table_exists('conversations'):
                    print("📝 No existing conversations to migrate")
                    return True
                
                print("🔄 Migrating existing conversations to events...")
                
                # Get all conversations
                cursor.execute("""
                    SELECT thread_id, user_message, agent_response, timestamp, metadata
                    FROM conversations
                    ORDER BY timestamp ASC
                """)
                
                conversations = cursor.fetchall()
                migrated_count = 0
                
                for conv in conversations:
                    thread_id, user_msg, agent_resp, timestamp, metadata = conv
                    
                    # Create user message event
                    cursor.execute("""
                        INSERT INTO events (timestamp, source, event_type, payload_json, thread_id)
                        VALUES (?, 'user', 'message', ?, ?)
                    """, (timestamp, json.dumps({
                        'message': user_msg,
                        'original_metadata': metadata
                    }), thread_id))
                    
                    # Create agent response event
                    cursor.execute("""
                        INSERT INTO events (timestamp, source, event_type, payload_json, thread_id)
                        VALUES (?, 'agent', 'response', ?, ?)
                    """, (timestamp, json.dumps({
                        'message': agent_resp,
                        'original_metadata': metadata
                    }), thread_id))
                    
                    migrated_count += 1
                
                conn.commit()
                print(f"✅ Migrated {migrated_count} conversations to events")
                return True
                
        except Exception as e:
            print(f"❌ Data migration failed: {e}")
            return False
    
    def update_schema_version(self):
        """Update schema version in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO schema_version (version)
                    VALUES (?)
                """, (self.schema_version,))
                conn.commit()
                
        except Exception as e:
            print(f"❌ Schema version update failed: {e}")
    
    def verify_migration(self) -> bool:
        """Verify migration success by checking table structure."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check all required tables exist
                required_tables = [
                    'events', 'hypotheses', 'knowledge', 'knowledge_fts',
                    'concepts', 'relationships', 'graph_snapshots'
                ]
                
                for table in required_tables:
                    cursor.execute("""
                        SELECT name FROM sqlite_master 
                        WHERE type='table' AND name=?
                    """, (table,))
                    
                    if not cursor.fetchone():
                        print(f"❌ Required table '{table}' not found")
                        return False
                
                # Check schema version
                current_version = self.get_schema_version()
                if current_version != self.schema_version:
                    print(f"❌ Schema version mismatch: expected {self.schema_version}, got {current_version}")
                    return False
                
                print("✅ Migration verification passed")
                return True
                
        except Exception as e:
            print(f"❌ Migration verification failed: {e}")
            return False
    
    def rollback(self) -> bool:
        """Rollback migration if needed."""
        try:
            if not self.backup_path.exists():
                print("❌ No backup found for rollback")
                return False
            
            print("🔄 Rolling back migration...")
            
            # Replace current database with backup
            if self.db_path.exists():
                self.db_path.unlink()
            
            shutil.copy2(self.backup_path, self.db_path)
            
            print("✅ Rollback completed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Rollback failed: {e}")
            return False
    
    def cleanup_backup(self, keep_backup: bool = True):
        """Clean up backup file after successful migration."""
        try:
            if not keep_backup and self.backup_path.exists():
                self.backup_path.unlink()
                print("🗑️ Backup file cleaned up")
            else:
                print(f"💾 Backup preserved at: {self.backup_path}")
                
        except Exception as e:
            print(f"⚠️ Backup cleanup failed: {e}")


def run_migration(db_path: str = None) -> bool:
    """
    Convenience function to run database migration.
    
    Args:
        db_path: Path to database file (uses default if None)
        
    Returns:
        True if migration successful
    """
    if db_path is None:
        from ..config import get_settings
        settings = get_settings()
        db_path = settings.memory_db_path
    
    migrator = DatabaseMigrator(db_path)
    success = migrator.migrate()
    
    if success:
        migrator.cleanup_backup(keep_backup=True)
    
    return success


if __name__ == "__main__":
    # Run migration when script is executed directly
    print("🚀 Running database migration...")
    success = run_migration()
    
    if success:
        print("🎉 Migration completed successfully!")
    else:
        print("💥 Migration failed!")
        exit(1) 