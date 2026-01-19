"""Workflow decorators."""

from typing import Type, Optional, Dict, Any


def selector(
    selector_class: Type,
    slo: Optional[Any] = None,
    **kwargs
):
    """Decorator to specify selector for a workflow class.

    Args:
        selector_class: Selector class (GreedySelector, AdaptiveSelector)
        slo: SLOConstraints instance
        **kwargs: Additional selector configuration

    Example:
        @selector(AdaptiveSelector, slo=SLOConstraints(min_accuracy=0.85))
        class MyWorkflow(Workflow):
            ...
    """
    def decorator(cls: Type) -> Type:
        cls._selector_class = selector_class
        cls._selector_slo = slo
        cls._selector_kwargs = kwargs
        return cls

    return decorator
