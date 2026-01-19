"""Greedy selector - picks best implementation by single metric."""

from typing import List, Optional
from enum import Enum

from .base import Selector, Implementation
from .slo import SLOConstraints


class Strategy(Enum):
    """Selection strategy for greedy selector."""
    COST = "cost"          # Minimize cost
    LATENCY = "latency"    # Minimize latency
    QUALITY = "quality"    # Maximize accuracy


class GreedySelector(Selector):
    """Greedy selector that optimizes for a single metric.

    Simple but effective - picks the best implementation
    based on cost, latency, or quality.
    """

    def __init__(
        self,
        strategy: Strategy = Strategy.COST,
        slo_constraints: Optional[SLOConstraints] = None
    ):
        super().__init__(name=f"greedy-{strategy.value}", slo_constraints=slo_constraints)
        self.strategy = strategy

    def select(
        self,
        task_id: str,
        implementations: List[Implementation]
    ) -> Implementation:
        """Select best implementation based on strategy."""
        if not implementations:
            raise ValueError(f"No implementations available for {task_id}")

        # Return cached selection if exists
        if task_id in self._selections:
            return self._selections[task_id]

        # Filter by SLO constraints first
        viable = self._filter_by_slo(implementations)
        if not viable:
            print(f"[{self.name}] No SLO-compliant implementations, using all")
            viable = implementations

        # Select based on strategy
        if self.strategy == Strategy.COST:
            selected = self._select_by_cost(viable)
        elif self.strategy == Strategy.LATENCY:
            selected = self._select_by_latency(viable)
        else:  # QUALITY
            selected = self._select_by_quality(viable)

        # Cache and return
        self._selections[task_id] = selected
        self._log_selection(task_id, selected, implementations)
        return selected

    def _filter_by_slo(self, implementations: List[Implementation]) -> List[Implementation]:
        """Filter implementations that meet SLO constraints."""
        viable = []
        for impl in implementations:
            if impl.profile is None:
                viable.append(impl)  # No profile = assume viable
                continue

            # Check latency constraint
            if impl.profile.p95_latency_ms and impl.profile.p95_latency_ms > self.slo_constraints.max_p95_latency_ms:
                continue

            # Check accuracy constraint
            if impl.profile.accuracy and impl.profile.accuracy < self.slo_constraints.min_accuracy:
                continue

            viable.append(impl)

        return viable

    def _select_by_cost(self, implementations: List[Implementation]) -> Implementation:
        """Select cheapest implementation."""
        return min(
            implementations,
            key=lambda impl: impl.profile.cost_per_call if impl.profile else float('inf')
        )

    def _select_by_latency(self, implementations: List[Implementation]) -> Implementation:
        """Select fastest implementation."""
        return min(
            implementations,
            key=lambda impl: impl.profile.avg_latency_ms if impl.profile else float('inf')
        )

    def _select_by_quality(self, implementations: List[Implementation]) -> Implementation:
        """Select highest quality implementation."""
        return max(
            implementations,
            key=lambda impl: impl.profile.accuracy if impl.profile and impl.profile.accuracy else 0
        )

    def _log_selection(
        self,
        task_id: str,
        selected: Implementation,
        all_impls: List[Implementation]
    ):
        """Log selection decision."""
        if self.strategy == Strategy.COST:
            metric = f"${selected.profile.cost_per_call:.4f}" if selected.profile else "unknown"
            print(f"[{self.name}] {task_id}: Selected {selected} (lowest cost: {metric})")
        elif self.strategy == Strategy.LATENCY:
            metric = f"{selected.profile.avg_latency_ms:.0f}ms" if selected.profile else "unknown"
            print(f"[{self.name}] {task_id}: Selected {selected} (lowest latency: {metric})")
        else:
            metric = f"{selected.profile.accuracy:.1%}" if selected.profile and selected.profile.accuracy else "unknown"
            print(f"[{self.name}] {task_id}: Selected {selected} (highest quality: {metric})")

    def get_summary(self) -> dict:
        return {
            "strategy": self.strategy.value,
            "selections": {k: str(v) for k, v in self._selections.items()},
            "total_switches": 0,
            "slo_compliant": True
        }
