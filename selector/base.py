"""Base selector interface and implementation dataclass."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from .profile import ModelProfile
from .slo import SLOConstraints


@dataclass
class Implementation:
    """A model implementation that can be selected.

    Represents one way to fulfill a task - a specific
    provider, model, and endpoint combination.
    """
    provider: str
    model: str
    endpoint: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    profile: Optional[ModelProfile] = None

    def __str__(self) -> str:
        return f"{self.provider}/{self.model}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for connector pool."""
        d = {
            'provider': self.provider,
            'model': self.model,
        }
        if self.endpoint:
            d['endpoint'] = self.endpoint
        if self.config:
            d.update(self.config)
        return d


class Selector(ABC):
    """Abstract base class for model selectors.

    Selectors choose which implementation to use for a task
    based on various strategies (greedy, adaptive, pareto).
    """

    def __init__(self, name: str, slo_constraints: Optional[SLOConstraints] = None):
        self.name = name
        self.slo_constraints = slo_constraints or SLOConstraints()
        self._selections: Dict[str, Implementation] = {}

    @abstractmethod
    def select(
        self,
        task_id: str,
        implementations: List[Implementation]
    ) -> Implementation:
        """Select the best implementation for a task.

        Args:
            task_id: Unique identifier for the task
            implementations: Available implementation options

        Returns:
            Selected implementation
        """
        pass

    def update_metrics(
        self,
        task_id: str,
        latency_ms: float,
        cost: float,
        success: bool
    ) -> None:
        """Report execution metrics for potential adaptation.

        Override in adaptive selectors to enable runtime switching.

        Args:
            task_id: Task that was executed
            latency_ms: Execution latency
            cost: Execution cost
            success: Whether execution succeeded
        """
        pass

    def get_current(self, task_id: str) -> Optional[Implementation]:
        """Get currently selected implementation for a task."""
        return self._selections.get(task_id)

    def reset(self) -> None:
        """Reset selector state."""
        self._selections.clear()
