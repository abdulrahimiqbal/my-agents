"""
Real-Time Collaboration System - WebSocket-based Multi-User Physics Research

This module implements a real-time collaboration system that enables:
1. WebSocket-based real-time communication
2. Live agent status updates and monitoring
3. Concurrent user interaction support
4. Shared workspace management
5. Real-time result broadcasting
6. Collaborative session management
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from collections import defaultdict

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    WebSocketServerProtocol = Any

from ..agents.advanced_supervisor import AdvancedSupervisorAgent, TaskType, TaskRequest, TaskResult
from ..agents.parallel_orchestrator import ParallelAgentOrchestrator
from ..database.knowledge_api import KnowledgeAPI


class MessageType(Enum):
    """Types of real-time messages."""
    USER_JOIN = "user_join"
    USER_LEAVE = "user_leave"
    AGENT_STATUS = "agent_status"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    RESULT_UPDATE = "result_update"
    CHAT_MESSAGE = "chat_message"
    WORKSPACE_UPDATE = "workspace_update"
    SYSTEM_NOTIFICATION = "system_notification"
    ERROR = "error"


class UserRole(Enum):
    """User roles in collaboration."""
    OBSERVER = "observer"
    PARTICIPANT = "participant"
    MODERATOR = "moderator"
    ADMIN = "admin"


class AgentStatus(Enum):
    """Agent status for real-time monitoring."""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"


@dataclass
class CollaborationUser:
    """User in a collaboration session."""
    user_id: str
    username: str
    role: UserRole
    websocket: Optional[WebSocketServerProtocol] = None
    joined_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentStatusInfo:
    """Real-time agent status information."""
    agent_id: str
    status: AgentStatus
    current_task: Optional[str] = None
    progress: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollaborationMessage:
    """Real-time collaboration message."""
    message_id: str
    message_type: MessageType
    sender_id: str
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: Optional[str] = None


@dataclass
class CollaborationSession:
    """Collaboration session with multiple users."""
    session_id: str
    name: str
    created_by: str
    created_at: datetime = field(default_factory=datetime.now)
    users: Dict[str, CollaborationUser] = field(default_factory=dict)
    agent_statuses: Dict[str, AgentStatusInfo] = field(default_factory=dict)
    shared_workspace: Dict[str, Any] = field(default_factory=dict)
    active_tasks: Dict[str, TaskRequest] = field(default_factory=dict)
    message_history: List[CollaborationMessage] = field(default_factory=list)
    max_users: int = 10


class RealTimeCollaborationSystem:
    """
    Real-time collaboration system for multi-user physics research.
    
    Features:
    - WebSocket-based real-time communication
    - Live agent monitoring and status updates
    - Concurrent user interaction support
    - Shared workspace management
    - Real-time result broadcasting
    """
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 8765,
                 max_sessions: int = 100):
        """
        Initialize the real-time collaboration system.
        
        Args:
            host: WebSocket server host
            port: WebSocket server port
            max_sessions: Maximum number of concurrent sessions
        """
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets library is required for real-time collaboration")
        
        self.host = host
        self.port = port
        self.max_sessions = max_sessions
        
        # Initialize components
        self.supervisor = AdvancedSupervisorAgent()
        self.orchestrator = ParallelAgentOrchestrator()
        self.knowledge_api = KnowledgeAPI()
        self.logger = logging.getLogger(__name__)
        
        # Collaboration state
        self.sessions: Dict[str, CollaborationSession] = {}
        self.user_sessions: Dict[str, str] = {}  # user_id -> session_id
        self.connected_users: Dict[str, WebSocketServerProtocol] = {}
        
        # Message handlers
        self.message_handlers: Dict[MessageType, Callable] = {
            MessageType.USER_JOIN: self._handle_user_join,
            MessageType.USER_LEAVE: self._handle_user_leave,
            MessageType.CHAT_MESSAGE: self._handle_chat_message,
            MessageType.TASK_STARTED: self._handle_task_started,
            MessageType.WORKSPACE_UPDATE: self._handle_workspace_update,
        }
        
        # Agent monitoring
        self.agent_monitors: Dict[str, asyncio.Task] = {}
        self.is_running = False
    
    async def start_server(self):
        """Start the WebSocket server."""
        self.is_running = True
        self.logger.info(f"Starting collaboration server on {self.host}:{self.port}")
        
        # Start agent monitoring
        await self._start_agent_monitoring()
        
        # Start WebSocket server
        server = await websockets.serve(
            self._handle_websocket_connection,
            self.host,
            self.port
        )
        
        self.logger.info("Real-time collaboration server started")
        return server
    
    async def stop_server(self):
        """Stop the WebSocket server."""
        self.is_running = False
        
        # Stop agent monitoring
        for task in self.agent_monitors.values():
            task.cancel()
        
        # Close all connections
        for websocket in self.connected_users.values():
            await websocket.close()
        
        self.logger.info("Real-time collaboration server stopped")
    
    async def _handle_websocket_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new WebSocket connection."""
        user_id = None
        try:
            # Wait for authentication message
            auth_message = await websocket.recv()
            auth_data = json.loads(auth_message)
            
            user_id = auth_data.get("user_id")
            username = auth_data.get("username", f"User_{user_id}")
            session_id = auth_data.get("session_id")
            
            if not user_id:
                await self._send_error(websocket, "Authentication required")
                return
            
            # Register user
            self.connected_users[user_id] = websocket
            
            # Join or create session
            if session_id and session_id in self.sessions:
                await self._join_session(user_id, username, session_id, websocket)
            else:
                session_id = await self._create_session(user_id, username, websocket)
            
            # Handle messages
            await self._handle_user_messages(user_id, websocket)
            
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"User {user_id} disconnected")
        except Exception as e:
            self.logger.error(f"WebSocket error for user {user_id}: {e}")
            await self._send_error(websocket, str(e))
        finally:
            # Clean up
            if user_id:
                await self._handle_user_disconnect(user_id)
    
    async def _handle_user_messages(self, user_id: str, websocket: WebSocketServerProtocol):
        """Handle messages from a user."""
        async for message_data in websocket:
            try:
                message = json.loads(message_data)
                
                # Create collaboration message
                collab_message = CollaborationMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType(message["type"]),
                    sender_id=user_id,
                    content=message.get("content", {}),
                    session_id=self.user_sessions.get(user_id)
                )
                
                # Handle message
                await self._handle_message(collab_message)
                
            except Exception as e:
                self.logger.error(f"Error handling message from {user_id}: {e}")
                await self._send_error(websocket, f"Message handling error: {e}")
    
    async def _handle_message(self, message: CollaborationMessage):
        """Handle a collaboration message."""
        handler = self.message_handlers.get(message.message_type)
        if handler:
            await handler(message)
        else:
            self.logger.warning(f"No handler for message type: {message.message_type}")
    
    async def _handle_user_join(self, message: CollaborationMessage):
        """Handle user join message."""
        # This is handled in _join_session, but we can add additional logic here
        await self._broadcast_to_session(
            message.session_id,
            MessageType.SYSTEM_NOTIFICATION,
            {"message": f"User {message.sender_id} joined the session"}
        )
    
    async def _handle_user_leave(self, message: CollaborationMessage):
        """Handle user leave message."""
        await self._leave_session(message.sender_id)
    
    async def _handle_chat_message(self, message: CollaborationMessage):
        """Handle chat message."""
        # Broadcast chat message to all users in session
        await self._broadcast_to_session(
            message.session_id,
            MessageType.CHAT_MESSAGE,
            {
                "sender_id": message.sender_id,
                "content": message.content,
                "timestamp": message.timestamp.isoformat()
            }
        )
    
    async def _handle_task_started(self, message: CollaborationMessage):
        """Handle task started message."""
        session_id = message.session_id
        if session_id in self.sessions:
            session = self.sessions[session_id]
            
            # Create task request
            task_request = TaskRequest(
                task_id=str(uuid.uuid4()),
                task_type=TaskType(message.content["task_type"]),
                content=message.content["content"]
            )
            
            session.active_tasks[task_request.task_id] = task_request
            
            # Broadcast task started
            await self._broadcast_to_session(
                session_id,
                MessageType.TASK_STARTED,
                {
                    "task_id": task_request.task_id,
                    "task_type": task_request.task_type.value,
                    "content": task_request.content,
                    "started_by": message.sender_id
                }
            )
            
            # Start task execution (async)
            asyncio.create_task(self._execute_task(session_id, task_request))
    
    async def _handle_workspace_update(self, message: CollaborationMessage):
        """Handle workspace update message."""
        session_id = message.session_id
        if session_id in self.sessions:
            session = self.sessions[session_id]
            
            # Update workspace
            update_data = message.content
            for key, value in update_data.items():
                session.shared_workspace[key] = value
            
            # Broadcast update to other users
            await self._broadcast_to_session(
                session_id,
                MessageType.WORKSPACE_UPDATE,
                update_data,
                exclude_user=message.sender_id
            )
    
    async def _create_session(self, user_id: str, username: str, websocket: WebSocketServerProtocol) -> str:
        """Create a new collaboration session."""
        session_id = str(uuid.uuid4())
        
        session = CollaborationSession(
            session_id=session_id,
            name=f"Session_{session_id[:8]}",
            created_by=user_id
        )
        
        # Add user to session
        user = CollaborationUser(
            user_id=user_id,
            username=username,
            role=UserRole.ADMIN,
            websocket=websocket
        )
        
        session.users[user_id] = user
        self.sessions[session_id] = session
        self.user_sessions[user_id] = session_id
        
        # Send session info
        await self._send_message(
            websocket,
            MessageType.USER_JOIN,
            {
                "session_id": session_id,
                "role": user.role.value,
                "users": [{"user_id": u.user_id, "username": u.username, "role": u.role.value} 
                         for u in session.users.values()]
            }
        )
        
        self.logger.info(f"Created session {session_id} for user {user_id}")
        return session_id
    
    async def _join_session(self, user_id: str, username: str, session_id: str, websocket: WebSocketServerProtocol):
        """Join an existing collaboration session."""
        if session_id not in self.sessions:
            await self._send_error(websocket, "Session not found")
            return
        
        session = self.sessions[session_id]
        
        if len(session.users) >= session.max_users:
            await self._send_error(websocket, "Session is full")
            return
        
        # Add user to session
        user = CollaborationUser(
            user_id=user_id,
            username=username,
            role=UserRole.PARTICIPANT,
            websocket=websocket
        )
        
        session.users[user_id] = user
        self.user_sessions[user_id] = session_id
        
        # Send session info to new user
        await self._send_message(
            websocket,
            MessageType.USER_JOIN,
            {
                "session_id": session_id,
                "role": user.role.value,
                "users": [{"user_id": u.user_id, "username": u.username, "role": u.role.value} 
                         for u in session.users.values()],
                "workspace": session.shared_workspace,
                "active_tasks": list(session.active_tasks.keys())
            }
        )
        
        # Notify other users
        await self._broadcast_to_session(
            session_id,
            MessageType.SYSTEM_NOTIFICATION,
            {"message": f"{username} joined the session"},
            exclude_user=user_id
        )
        
        self.logger.info(f"User {user_id} joined session {session_id}")
    
    async def _leave_session(self, user_id: str):
        """Leave a collaboration session."""
        session_id = self.user_sessions.get(user_id)
        if not session_id or session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        user = session.users.pop(user_id, None)
        del self.user_sessions[user_id]
        
        if user:
            # Notify other users
            await self._broadcast_to_session(
                session_id,
                MessageType.SYSTEM_NOTIFICATION,
                {"message": f"{user.username} left the session"},
                exclude_user=user_id
            )
        
        # Clean up session if empty
        if not session.users:
            del self.sessions[session_id]
            self.logger.info(f"Deleted empty session {session_id}")
        
        self.logger.info(f"User {user_id} left session {session_id}")
    
    async def _handle_user_disconnect(self, user_id: str):
        """Handle user disconnection."""
        if user_id in self.connected_users:
            del self.connected_users[user_id]
        
        await self._leave_session(user_id)
    
    async def _execute_task(self, session_id: str, task_request: TaskRequest):
        """Execute a task and broadcast results."""
        try:
            # Update agent status
            await self._update_agent_status("supervisor", AgentStatus.BUSY, task_request.task_id)
            
            # Execute task (simplified - in real implementation, use orchestrator)
            result = TaskResult(
                task_id=task_request.task_id,
                agent_name="supervisor",
                result=f"Task {task_request.task_id} completed successfully",
                confidence=0.9,
                execution_time=2.0,
                success=True
            )
            
            # Broadcast completion
            await self._broadcast_to_session(
                session_id,
                MessageType.TASK_COMPLETED,
                {
                    "task_id": task_request.task_id,
                    "result": result.result,
                    "confidence": result.confidence,
                    "execution_time": result.execution_time,
                    "success": result.success
                }
            )
            
            # Update agent status
            await self._update_agent_status("supervisor", AgentStatus.IDLE)
            
        except Exception as e:
            self.logger.error(f"Task execution error: {e}")
            await self._broadcast_to_session(
                session_id,
                MessageType.ERROR,
                {"message": f"Task execution failed: {e}"}
            )
    
    async def _start_agent_monitoring(self):
        """Start monitoring agent statuses."""
        # Create monitoring tasks for different agents
        agents = ["supervisor", "physics_expert", "hypothesis_generator"]
        
        for agent_id in agents:
            self.agent_monitors[agent_id] = asyncio.create_task(
                self._monitor_agent(agent_id)
            )
    
    async def _monitor_agent(self, agent_id: str):
        """Monitor a specific agent's status."""
        while self.is_running:
            try:
                # Simulate agent monitoring (in real implementation, check actual agent status)
                status_info = AgentStatusInfo(
                    agent_id=agent_id,
                    status=AgentStatus.IDLE,
                    progress=0.0,
                    metrics={"cpu_usage": 0.1, "memory_usage": 0.2}
                )
                
                # Broadcast status to all sessions
                await self._broadcast_agent_status(status_info)
                
                # Wait before next check
                await asyncio.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Agent monitoring error for {agent_id}: {e}")
                await asyncio.sleep(10)
    
    async def _update_agent_status(self, agent_id: str, status: AgentStatus, current_task: Optional[str] = None):
        """Update and broadcast agent status."""
        status_info = AgentStatusInfo(
            agent_id=agent_id,
            status=status,
            current_task=current_task,
            progress=0.0 if status == AgentStatus.IDLE else 0.5
        )
        
        await self._broadcast_agent_status(status_info)
    
    async def _broadcast_agent_status(self, status_info: AgentStatusInfo):
        """Broadcast agent status to all sessions."""
        for session in self.sessions.values():
            session.agent_statuses[status_info.agent_id] = status_info
            
            await self._broadcast_to_session(
                session.session_id,
                MessageType.AGENT_STATUS,
                {
                    "agent_id": status_info.agent_id,
                    "status": status_info.status.value,
                    "current_task": status_info.current_task,
                    "progress": status_info.progress,
                    "last_update": status_info.last_update.isoformat(),
                    "metrics": status_info.metrics
                }
            )
    
    async def _broadcast_to_session(self, session_id: str, message_type: MessageType, 
                                  content: Dict[str, Any], exclude_user: Optional[str] = None):
        """Broadcast message to all users in a session."""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        
        for user_id, user in session.users.items():
            if exclude_user and user_id == exclude_user:
                continue
            
            if user.websocket:
                await self._send_message(user.websocket, message_type, content)
    
    async def _send_message(self, websocket: WebSocketServerProtocol, 
                          message_type: MessageType, content: Dict[str, Any]):
        """Send message to a specific websocket."""
        try:
            message = {
                "type": message_type.value,
                "content": content,
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send(json.dumps(message))
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
    
    async def _send_error(self, websocket: WebSocketServerProtocol, error_message: str):
        """Send error message to websocket."""
        await self._send_message(websocket, MessageType.ERROR, {"message": error_message})
    
    def get_collaboration_statistics(self) -> Dict[str, Any]:
        """Get collaboration system statistics."""
        return {
            "active_sessions": len(self.sessions),
            "connected_users": len(self.connected_users),
            "total_users": sum(len(session.users) for session in self.sessions.values()),
            "active_tasks": sum(len(session.active_tasks) for session in self.sessions.values()),
            "server_status": "running" if self.is_running else "stopped"
        } 