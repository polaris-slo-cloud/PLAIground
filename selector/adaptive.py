"""Adaptive selector (Pixie) - runtime model adaptation based on metrics."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

from .base import Selector, Implementation
from .slo import SLOConstraints
from .greedy import GreedySelector, Strategy


@dataclass
class SlackMetrics:
    """Remaining budget calculations for adaptive decisions."""
    latency_per_request: float    # Remaining latency budget per request
    cost_per_request: float       # Remaining cost budget per request
    accuracy_gap: float           # Current accuracy vs target
    progress: float               # Fraction of batch completed
    requests_remaining: int

    def is_critical(self) -> bool:
        """Check if we're in critical budget territory."""
        return self.latency_per_request < 0 or self.cost_per_request < 0

    def has_excess_budget(self) -> bool:
        """Check if we have significant excess budget."""
        return self.latency_per_request > 3000 and self.cost_per_request > 0.05


@dataclass
class SwitchEvent:
    """Record of a model switching decision."""
    request_num: int
    reason: str
    old_impl: str
    new_impl: str


class AdaptiveSelector(Selector):
    """Adaptive selector that adjusts selections based on runtime metrics.

    Also known as "Pixie" - monitors execution and adapts model
    selection to maintain SLO compliance.

    Key features:
    - Tracks cumulative metrics (latency, cost, accuracy)
    - Calculates remaining budget per request
    - Switches models when SLO violations detected
    - Supports multiple adaptation strategies
    """

    def __init__(
        self,
        slo_constraints: Optional[SLOConstraints] = None,
        total_requests: int = 100,
        initial_strategy: Strategy = Strategy.QUALITY,
        adaptation_interval: int = 10,
        safety_margin: float = 0.1
    ):
        super().__init__(name="adaptive", slo_constraints=slo_constraints)
        self.total_requests = total_requests
        self.initial_strategy = initial_strategy
        self.adaptation_interval = adaptation_interval
        self.safety_margin = safety_margin

        # Cumulative metrics
        self.requests_processed = 0
        self.total_latency_ms = 0.0
        self.total_cost = 0.0
        self.successful_requests = 0

        # Available implementations per task
        self._available: Dict[str, List[Implementation]] = {}

        # Switch history
        self.switch_events: List[SwitchEvent] = []
        self._last_switch = 0

        # Initial greedy selector for first selection
        self._greedy = GreedySelector(
            strategy=initial_strategy,
            slo_constraints=slo_constraints
        )

    def select(
        self,
        task_id: str,
        implementations: List[Implementation]
    ) -> Implementation:
        """Select implementation, adapting if needed."""

        # Store available implementations
        self._available[task_id] = implementations

        # First selection: use greedy strategy
        if task_id not in self._selections:
            selected = self._initial_selection(task_id, implementations)
            self._selections[task_id] = selected
            return selected

        # Return current selection (may have been adapted)
        return self._selections[task_id]

    def _initial_selection(
        self,
        task_id: str,
        implementations: List[Implementation]
    ) -> Implementation:
        """Make initial selection based on strategy and SLO constraints."""
        # Filter by SLO constraints
        viable = self._filter_slo_compliant(implementations)

        if not viable:
            viable = implementations

        # Select based on initial strategy
        if self.initial_strategy == Strategy.QUALITY:
            selected = max(viable, key=lambda i: i.profile.accuracy if i.profile and i.profile.accuracy else 0)
        elif self.initial_strategy == Strategy.COST:
            selected = min(viable, key=lambda i: i.profile.cost_per_call if i.profile else float('inf'))
        else:  # LATENCY
            selected = min(viable, key=lambda i: i.profile.avg_latency_ms if i.profile else float('inf'))

        return selected

    def _filter_slo_compliant(self, implementations: List[Implementation]) -> List[Implementation]:
        """Filter implementations that are estimated to meet SLO."""
        per_request_latency = self.slo_constraints.max_p95_latency_ms
        per_request_cost = self.slo_constraints.max_total_cost / self.total_requests

        viable = []
        for impl in implementations:
            if not impl.profile:
                viable.append(impl)
                continue

            latency_ok = impl.profile.avg_latency_ms <= per_request_latency * (1 - self.safety_margin)
            cost_ok = impl.profile.cost_per_call <= per_request_cost * (1 - self.safety_margin)
            accuracy_ok = (impl.profile.accuracy is None or
                          impl.profile.accuracy >= self.slo_constraints.min_accuracy * 0.95)

            if latency_ok and cost_ok and accuracy_ok:
                viable.append(impl)

        return viable

    def update_metrics(
        self,
        task_id: str,
        latency_ms: float,
        cost: float,
        success: bool
    ) -> None:
        """Update cumulative metrics and potentially adapt."""
        self.requests_processed += 1
        self.total_latency_ms += latency_ms
        self.total_cost += cost
        if success:
            self.successful_requests += 1

        # Check if we should adapt
        if self.requests_processed % self.adaptation_interval == 0:
            self._consider_adaptation()

    def _consider_adaptation(self):
        """Check and perform adaptation if needed."""
        slack = self._calculate_slack()
        if not slack:
            return

        should_switch, reason = self._should_switch(slack)
        if should_switch:
            self._adapt(slack, reason)

    def _calculate_slack(self) -> Optional[SlackMetrics]:
        """Calculate remaining budget per request."""
        remaining = self.total_requests - self.requests_processed
        if remaining == 0:
            return None

        # Latency slack
        latency_budget = self.slo_constraints.max_p95_latency_ms * self.total_requests
        latency_remaining = latency_budget - self.total_latency_ms
        latency_per_request = latency_remaining / remaining

        # Cost slack
        cost_remaining = self.slo_constraints.max_total_cost - self.total_cost
        cost_per_request = cost_remaining / remaining

        # Accuracy gap
        current_accuracy = self.successful_requests / self.requests_processed if self.requests_processed > 0 else 0
        accuracy_gap = current_accuracy - self.slo_constraints.min_accuracy

        return SlackMetrics(
            latency_per_request=latency_per_request,
            cost_per_request=cost_per_request,
            accuracy_gap=accuracy_gap,
            progress=self.requests_processed / self.total_requests,
            requests_remaining=remaining
        )

    def _should_switch(self, slack: SlackMetrics) -> Tuple[bool, str]:
        """Determine if we should switch models."""
        # Don't switch too frequently
        if self.requests_processed - self._last_switch < self.adaptation_interval:
            return False, ""

        # Don't switch near the end
        if slack.progress > 0.95:
            return False, ""

        # Check for budget issues
        current_latency = self._estimate_current_latency()
        current_cost = self._estimate_current_cost()

        safe_latency = slack.latency_per_request * (1 - self.safety_margin)
        safe_cost = slack.cost_per_request * (1 - self.safety_margin)

        if current_latency > safe_latency:
            return True, f"Latency overspend: {current_latency:.0f}ms > {safe_latency:.0f}ms"

        if current_cost > safe_cost:
            return True, f"Cost overspend: ${current_cost:.4f} > ${safe_cost:.4f}"

        # Check if we can upgrade for accuracy
        if slack.accuracy_gap < -0.02 and slack.has_excess_budget():
            return True, f"Low accuracy ({slack.accuracy_gap:.1%}) with excess budget"

        return False, ""

    def _adapt(self, slack: SlackMetrics, reason: str):
        """Adapt model selection based on slack."""

        for task_id, implementations in self._available.items():
            old_impl = self._selections.get(task_id)

            if slack.is_critical():
                # Emergency: pick cheapest
                new_impl = self._find_cheapest(implementations)
            elif slack.has_excess_budget() and slack.accuracy_gap < 0:
                # Upgrade: pick best quality within budget
                new_impl = self._find_best_quality(implementations, slack)
            elif "overspend" in reason.lower():
                # Downgrade: find cheaper alternative
                new_impl = self._find_constrained(implementations, slack)
            else:
                new_impl = old_impl

            if new_impl and new_impl != old_impl:
                self._selections[task_id] = new_impl
                self.switch_events.append(SwitchEvent(
                    request_num=self.requests_processed,
                    reason=reason,
                    old_impl=str(old_impl) if old_impl else "none",
                    new_impl=str(new_impl)
                ))

        self._last_switch = self.requests_processed

    def _find_cheapest(self, implementations: List[Implementation]) -> Implementation:
        """Find cheapest implementation."""
        return min(
            implementations,
            key=lambda i: i.profile.cost_per_call if i.profile else float('inf')
        )

    def _find_best_quality(
        self,
        implementations: List[Implementation],
        slack: SlackMetrics
    ) -> Implementation:
        """Find best quality within budget."""
        viable = [
            impl for impl in implementations
            if impl.profile and
               impl.profile.avg_latency_ms <= slack.latency_per_request and
               impl.profile.cost_per_call <= slack.cost_per_request
        ]

        if not viable:
            return self._find_cheapest(implementations)

        return max(viable, key=lambda i: i.profile.accuracy if i.profile and i.profile.accuracy else 0)

    def _find_constrained(
        self,
        implementations: List[Implementation],
        slack: SlackMetrics
    ) -> Implementation:
        """Find best implementation within tighter constraints."""
        scored = []
        for impl in implementations:
            if not impl.profile:
                continue

            latency_score = max(0, 1 - impl.profile.avg_latency_ms / slack.latency_per_request) if slack.latency_per_request > 0 else 0
            cost_score = max(0, 1 - impl.profile.cost_per_call / slack.cost_per_request) if slack.cost_per_request > 0 else 0
            accuracy = impl.profile.accuracy or 0.8

            score = 0.3 * accuracy + 0.4 * cost_score + 0.3 * latency_score
            scored.append((score, impl))

        if not scored:
            return implementations[0]

        return max(scored, key=lambda x: x[0])[1]

    def _estimate_current_latency(self) -> float:
        """Estimate latency of current configuration."""
        total = 0
        for impl in self._selections.values():
            if impl.profile:
                total += impl.profile.avg_latency_ms
        return total or 1000

    def _estimate_current_cost(self) -> float:
        """Estimate cost of current configuration."""
        total = 0
        for impl in self._selections.values():
            if impl.profile:
                total += impl.profile.cost_per_call
        return total or 0.001

    def get_summary(self) -> Dict:
        """Get summary of adaptive behavior."""
        return {
            'total_switches': len(self.switch_events),
            'switch_points': [e.request_num for e in self.switch_events],
            'final_accuracy': self.successful_requests / self.requests_processed if self.requests_processed > 0 else 0,
            'total_cost': self.total_cost,
            'avg_latency': self.total_latency_ms / self.requests_processed if self.requests_processed > 0 else 0,
            'slo_compliant': self._check_slo_compliance()
        }

    def _check_slo_compliance(self) -> bool:
        """Check if current metrics meet SLO."""
        if self.requests_processed == 0:
            return True

        return self.slo_constraints.is_satisfied(
            accuracy=self.successful_requests / self.requests_processed,
            latency_ms=self.total_latency_ms / self.requests_processed,
            total_cost=self.total_cost
        )

    def reset(self) -> None:
        """Reset selector state for new batch."""
        super().reset()
        self.requests_processed = 0
        self.total_latency_ms = 0.0
        self.total_cost = 0.0
        self.successful_requests = 0
        self.switch_events.clear()
        self._last_switch = 0
        self._available.clear()
