"""
Real-Time Collaboration System

This package provides real-time collaboration capabilities for multi-user physics research,
including WebSocket-based communication, live agent monitoring, and shared workspace management.
"""

try:
    from .realtime_system import (
        RealTimeCollaborationSystem,
        MessageType,
        UserRole,
        AgentStatus,
        CollaborationUser,
        CollaborationSession,
        CollaborationMessage,
        AgentStatusInfo
    )
    COLLABORATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Real-time collaboration not available: {e}")
    COLLABORATION_AVAILABLE = False

__all__ = []

if COLLABORATION_AVAILABLE:
    __all__.extend([
        "RealTimeCollaborationSystem",
        "MessageType",
        "UserRole", 
        "AgentStatus",
        "CollaborationUser",
        "CollaborationSession",
        "CollaborationMessage",
        "AgentStatusInfo"
    ]) 