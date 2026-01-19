"""Model profile for selection decisions."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelProfile:
    """Performance profile for a model implementation.

    Used by selectors to make informed decisions without
    requiring live benchmarking.
    """
    avg_latency_ms: float = 1000.0
    cost_per_call: float = 0.001
    accuracy: Optional[float] = None

    # Additional metrics (optional)
    p95_latency_ms: Optional[float] = None
    throughput_qps: Optional[float] = None

    def __post_init__(self):
        if self.p95_latency_ms is None:
            # Estimate P95 as 1.5x average
            self.p95_latency_ms = self.avg_latency_ms * 1.5
