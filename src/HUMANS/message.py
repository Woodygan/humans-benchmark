from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal, Callable
import json

@dataclass
class Message:
    """Message in a conversation"""
    role: Literal["user", "assistant", "system", "tool"]
    text_input: Optional[str] = None
    audio_path: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


@dataclass
class ModelResponse:
    """Response from a model"""
    text: str
    audio_path: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)