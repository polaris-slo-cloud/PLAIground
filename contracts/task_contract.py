from dataclasses import dataclass, field
from typing import Dict, Any, Union, TypedDict, Optional
from enum import Enum


class Capability(Enum):
    LLM = "llm"
    VISION = "vision"
    CLASSIFICATION = "classification"
    OBJECT_DETECTION = "object_detection"
    EMBEDDING = "embedding"
    SPEECH_TO_TEXT = "speech_to_text"
    TOOL = "tool"

class PromptConfig(TypedDict, total=False):
    prompt: str
    temperature: Optional[float]
    max_tokens: Optional[int]

class ParameterConfig(TypedDict):
    labels: Optional[list]
    threshold: Optional[float]

@dataclass
class TaskConfig:
    """Task-specific configuration."""

    task_type: str
    parameters: Union[PromptConfig, ParameterConfig, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'TaskConfig':
        task_type = config.pop('task_type', 'default')
        return cls(task_type=task_type, parameters=config)


@dataclass
class TaskContract:
    """Task Contract combining capability and configuration."""

    capability: str
    config: TaskConfig

    @classmethod
    def create(cls, capability: str, task_config: Dict[str, Any]) -> 'TaskContract':
        return cls(
            capability=capability,
            config=TaskConfig.from_dict(task_config.copy())
        )
