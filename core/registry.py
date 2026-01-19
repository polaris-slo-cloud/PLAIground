"""Registry for Compoundable Models with selector support."""

from typing import Dict, List, Optional, TYPE_CHECKING

from threading import Lock

from .model import CompoundableModel
from connectors.pool import ConnectorPool

if TYPE_CHECKING:
    from selector import Selector, Implementation
    from config import ConfigLoader


class CompoundableModelRegistry:
    """Global registry for Compoundable Models (System Contract).

    Supports two modes:
    1. Manual binding: registry.bind(name, provider, model_id)
    2. Auto-selection: registry.configure(config_loader, selector)
       Then selector picks implementation at execution time.
    """

    _lock = Lock()
    _models: Dict[str, CompoundableModel] = {}
    _by_capability: Dict[str, List[str]] = {}

    # Selector support
    _selector: Optional['Selector'] = None
    _config_loader: Optional['ConfigLoader'] = None
    _implementations: Dict[str, List['Implementation']] = {}

    @classmethod
    def register(cls, model: CompoundableModel) -> None:
        with cls._lock:
            cls._models[model.name] = model

            if model.task_contract:
                cap = model.task_contract.capability
                if cap not in cls._by_capability:
                    cls._by_capability[cap] = []
                cls._by_capability[cap].append(model.name)

    @classmethod
    def get(cls, name: str) -> Optional[CompoundableModel]:
        return cls._models.get(name)

    @classmethod
    def configure(
        cls,
        config_loader: 'ConfigLoader',
        selector: 'Selector'
    ) -> None:
        """Configure registry with implementations and selector.

        This enables automatic model selection via Pixie/selectors.

        Args:
            config_loader: Loaded ConfigLoader with implementations
            selector: Selector instance (Greedy, Adaptive, etc.)
        """
        with cls._lock:
            cls._config_loader = config_loader
            cls._selector = selector
            cls._implementations = config_loader.get_all_implementations()
            print(f"[registry] Configured with {selector.name} selector")
            for task_id, impls in cls._implementations.items():
                print(f"  {task_id}: {len(impls)} implementations")

    @classmethod
    def bind(cls, name: str, provider: str, model_id: str, **config) -> bool:
        """Manually bind a registered model to an implementation.

        This is the explicit binding mode - use configure() for auto-selection.
        """
        model = cls.get(name)
        if model is None:
            return False

        connection_keys = {'api_key', 'base_url', 'timeout', 'endpoint'}
        connection_config = {k: v for k, v in config.items() if k in connection_keys}
        execution_config = {k: v for k, v in config.items() if k not in connection_keys}

        connector = ConnectorPool.get(provider, **connection_config)
        model.bind(connector, model_id, **execution_config)
        return True

    @classmethod
    def auto_bind(cls, name: str) -> bool:
        """Automatically bind using selector.

        Uses the configured selector to pick the best implementation
        from the loaded config.

        Args:
            name: Model name (also used as task_id for lookup)

        Returns:
            True if successfully bound
        """
        if cls._selector is None or cls._implementations is None:
            print(f"[registry] No selector configured, cannot auto-bind {name}")
            return False

        model = cls.get(name)
        if model is None:
            return False

        # Get implementations for this task
        implementations = cls._implementations.get(name, [])
        if not implementations:
            print(f"[registry] No implementations found for {name}")
            return False

        # Use selector to pick implementation
        selected = cls._selector.select(name, implementations)

        # Get connector and bind
        connection_config = {}
        if selected.endpoint:
            connection_config['endpoint'] = selected.endpoint
        connection_config.update(selected.config)

        connector = ConnectorPool.get(selected.provider, **connection_config)
        model.bind(connector, selected.model, **selected.config)

        print(f"[registry] Auto-bound {name} -> {selected}")
        return True

    @classmethod
    def get_selector(cls) -> Optional['Selector']:
        """Get the configured selector."""
        return cls._selector

    @classmethod
    def report_metrics(
        cls,
        name: str,
        latency_ms: float,
        cost: float,
        success: bool
    ) -> None:
        """Report execution metrics to selector for adaptation.

        Args:
            name: Task/model name
            latency_ms: Execution latency
            cost: Execution cost
            success: Whether execution succeeded
        """
        if cls._selector:
            cls._selector.update_metrics(name, latency_ms, cost, success)

    @classmethod
    def list_all(cls) -> List[str]:
        return list(cls._models.keys())

    @classmethod
    def list_by_capability(cls, capability: str) -> List[CompoundableModel]:
        names = cls._by_capability.get(capability, [])
        return [cls._models[n] for n in names if n in cls._models]

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._models.clear()
            cls._by_capability.clear()
            cls._selector = None
            cls._config_loader = None
            cls._implementations.clear()
