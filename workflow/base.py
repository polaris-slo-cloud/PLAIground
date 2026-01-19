"""Declarative Workflow class for compound AI systems."""

import time
from typing import Dict, Any, Optional, List, Type

from dotenv import load_dotenv
load_dotenv()

from core.model import CompoundableModel
from selector import Selector, SLOConstraints, AdaptiveSelector
from selector.base import Implementation
from config import load_implementations, ConfigLoader
from connectors.pool import ConnectorPool


class DeclarativeWorkflow:
    """Base class for declarative workflows.

    Workflows group related compoundable models and define
    how they interact. The @selector decorator specifies
    which selector (Pixie) to use for model selection.

    Example:
        @selector(AdaptiveSelector, slo=SLOConstraints(...))
        class MyWorkflow(Workflow):

            @compoundable_model(capability="object_detection")
            def detect(self, data: ImageInput) -> Detection:
                pass

            def run(self, image):
                return self.detect(image)
    """

    # Set by @selector decorator
    _selector_class: Optional[Type[Selector]] = None
    _selector_slo: Optional[SLOConstraints] = None
    _selector_kwargs: Dict[str, Any] = {}

    def __init__(self, config_path: Optional[str] = None):
        """Initialize workflow with config and selector.

        Args:
            config_path: Path to implementations.yaml (optional)
        """

        # Instance-level storage
        self._models: Dict[str, CompoundableModel] = {}
        self._selector: Optional[Selector] = None
        self._config: Optional[ConfigLoader] = None

        # Load configuration
        self._config = load_implementations(config_path)
        # Create selector
        self._create_selector()

        # Discover and bind models
        self._discover_models()

        self._bind_models()



    def _create_selector(self) -> None:
        """Create selector instance from class decorator."""
        selector_class = getattr(self.__class__, '_selector_class', None)

        if selector_class is None:
            # Default to AdaptiveSelector
            selector_class = AdaptiveSelector

        # Get SLO from decorator or config
        slo = getattr(self.__class__, '_selector_slo', None)
        if slo is None:
            slo = self._config.get_slo_constraints()

        # Get additional kwargs
        kwargs = getattr(self.__class__, '_selector_kwargs', {})

        # Create selector
        self._selector = selector_class(slo_constraints=slo, **kwargs)

    def _discover_models(self) -> None:
        """Discover compoundable models defined on this class."""
        for name in dir(self.__class__):
            if name.startswith('_'):
                continue

            attr = getattr(self.__class__, name)

            # Check if it's a compoundable model marker
            if hasattr(attr, '_compoundable_model_info'):
                info = attr._compoundable_model_info
                model = self._create_model(name, info)
                self._models[name] = model

    def _create_model(self, name: str, info: Dict[str, Any]) -> CompoundableModel:
        """Create CompoundableModel from method info."""
        from contracts.task_contract import TaskContract

        task_contract = TaskContract.create(
            capability=info['capability'],
            task_config=info.get('task_config', {})
        )

        return CompoundableModel(
            name=name,
            input_contract=info.get('input_contract'),
            output_contract=info.get('output_contract'),
            task_contract=task_contract
        )

    def _bind_models(self) -> None:
        """Bind all models using selector."""
        implementations = self._config.get_all_implementations()

        for name, model in self._models.items():
            impls = implementations.get(name, [])

            if not impls:
                continue

            # Use selector to pick implementation
            selected = self._selector.select(name, impls)

            # Create connector and bind
            connection_config = {}
            if selected.endpoint:
                connection_config['endpoint'] = selected.endpoint
            connection_config.update(selected.config)

            connector = ConnectorPool.get(selected.provider, **connection_config)
            model.bind(connector, selected.model, **selected.config)


    def _execute_model(self, name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a model and report metrics.

        This is called by the method wrappers created by @compoundable_model.
        """

        model = self._models.get(name)
        if model is None:
            raise RuntimeError(f"Model '{name}' not found in workflow")

        current_model_id = model.system_contract.model_id if model.system_contract else None

        # Check current selection vs bound model
        impl = self._selector.get_current(name)

        if impl and current_model_id != impl.model:
            connection_config = {}
            if impl.endpoint:
                connection_config['endpoint'] = impl.endpoint

            connection_config.update(impl.config)
            connector = ConnectorPool.get(impl.provider, **connection_config)

            model.bind(connector, impl.model, **impl.config)

        if not model.is_bound():
            raise RuntimeError(f"Model '{name}' not bound")

        # Execute with timing
        start = time.time()
        result = model.execute(input_data)
        latency_ms = (time.time() - start) * 1000

        # Get cost from profile (if available)
        cost = impl.profile.cost_per_call if impl and impl.profile else 0.001

        # Determine success
        success = result.get('output') is not None or result.get('has_detection', False)

        # Report to selector for adaptation
        self._selector.update_metrics(name, latency_ms, cost, success)

        return result

    def get_selector(self) -> Optional[Selector]:
        """Get the workflow's selector."""
        return self._selector

    def get_summary(self) -> Dict[str, Any]:
        """Get workflow execution summary."""
        if self._selector:
            return self._selector.get_summary()
        return {}
