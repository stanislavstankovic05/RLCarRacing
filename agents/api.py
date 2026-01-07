from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class AgentSpec:
    needs_pixels: bool
    discrete_wrapper: bool
    action_set_name: Optional[str] = None
    resize: int = 84
    frame_stack: int = 4


@dataclass
class LoadedAgent:
    name: str
    spec: AgentSpec
    policy: Any  
