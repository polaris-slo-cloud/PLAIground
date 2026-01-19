from typing import Dict, Any, Optional, Callable, Union
from functools import wraps
import inspect

from contracts.data_contract import DataContract, extract_contracts
from contracts.task_contract import TaskContract
from .model import CompoundableModel


def compoundable_model(
    capability: str,
    task_config: Optional[Dict[str, Any]] = None,
    auto_register: bool = True
) -> Callable:
    """Decorator to define a Compoundable Model.

    Works in two modes:
    1. Standalone function: Creates and registers a CompoundableModel
    2. Method in DeclarativeWorkflow: Marks method with metadata for discovery

    Args:
        capability: Model capability (llm, object_detection, classification, etc.)
        task_config: Task Configuration containing prompt, parameters, etc.
        auto_register: Whether to auto-register in global registry (standalone only)

    Examples:
        # Standalone function
        @compoundable_model(capability="llm")
        def generate_text(data: TextInput) -> TextOutput:
            pass

        # Method in workflow class
        class MyWorkflow(DeclarativeWorkflow):
            @compoundable_model(capability="object_detection")
            def detect(self, data: ImageInput) -> Detection:
                pass
    """

    def decorator(func: Callable) -> Union[CompoundableModel, Callable]:
        # Check if this is a method (first param is 'self')
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        is_method = params and params[0] == 'self'

        if is_method:
            # Method mode: mark with metadata, don't create model yet
            input_contract, output_contract = extract_contracts(func, skip_self=True)

            @wraps(func)
            def method_wrapper(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
                """Wrapper that delegates to workflow's model executor."""
                return self._execute_model(func.__name__, input_data)

            # Store metadata for DeclarativeWorkflow to discover
            method_wrapper._compoundable_model_info = {
                'capability': capability,
                'task_config': task_config or {},
                'input_contract': input_contract,
                'output_contract': output_contract,
            }

            return method_wrapper
        else:
            # Function mode: create and register model (original behavior)
            input_contract, output_contract = extract_contracts(func)

            task_contract = TaskContract.create(
                capability=capability,
                task_config=task_config or {}
            )

            model = CompoundableModel(
                name=func.__name__,
                input_contract=input_contract,
                output_contract=output_contract,
                task_contract=task_contract,
            )

            if auto_register:
                from .registry import CompoundableModelRegistry
                CompoundableModelRegistry.register(model)

            return model

    return decorator
