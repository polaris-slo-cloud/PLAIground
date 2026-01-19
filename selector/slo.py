"""SLO (Service Level Objective) constraints for model selection."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SLOConstraints:
    """Service Level Objectives for model selection.

    Defines the constraints that must be satisfied when
    selecting and adapting model implementations.
    """
    min_accuracy: float = 0.8
    max_p95_latency_ms: float = 2000.0
    max_total_cost: float = 0.10

    def is_satisfied(
        self,
        accuracy: float,
        latency_ms: float,
        total_cost: float
    ) -> bool:
        """Check if current metrics satisfy all SLO constraints."""
        return (
            accuracy >= self.min_accuracy and
            latency_ms <= self.max_p95_latency_ms and
            total_cost <= self.max_total_cost
        )

    def get_violations(
        self,
        accuracy: float,
        latency_ms: float,
        total_cost: float
    ) -> list:
        """Get list of SLO violations."""
        violations = []
        if accuracy < self.min_accuracy:
            violations.append(f"accuracy {accuracy:.2%} < {self.min_accuracy:.2%}")
        if latency_ms > self.max_p95_latency_ms:
            violations.append(f"latency {latency_ms:.0f}ms > {self.max_p95_latency_ms:.0f}ms")
        if total_cost > self.max_total_cost:
            violations.append(f"cost ${total_cost:.4f} > ${self.max_total_cost:.4f}")
        return violations
