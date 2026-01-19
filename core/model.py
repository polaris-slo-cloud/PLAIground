from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from .capability import CapabilityRegistry
from contracts.system_contract import SystemContract
from connectors.base import BackendConnector

if TYPE_CHECKING:
    from contracts.data_contract import DataContract
    from contracts.task_contract import TaskContract


@dataclass
class CompoundableModel:
    """Compoundable Model encapsulating the Three Contracts."""

    name: str
    input_contract: Optional['DataContract'] = None
    output_contract: Optional['DataContract'] = None
    task_contract: Optional['TaskContract'] = None
    system_contract: Optional[SystemContract] = field(default=None, repr=False)

    def bind(self, connector: BackendConnector, model_id: str, **config):
        """Bind this model to a specific implementation."""
        self.system_contract = SystemContract(
            provider='BASE', # Current state
            model_id=model_id,
            connector=connector,
            execution_config=config
        )

    def is_bound(self) -> bool:
        return self.system_contract is not None

    def _get_capability(self) -> str:
        """Get capability from Task Contract."""
        if self.task_contract:
            return self.task_contract.capability
        return "llm"

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:

        if not self.is_bound():
            raise RuntimeError(f"Model '{self.name}' not bound. Call registry.bind() first.")


        if self.input_contract and not self.input_contract.validate(input_data):
            raise ValueError(f"Input validation failed for '{self.name}'")

        capability = self._get_capability()
        handler = CapabilityRegistry.get_handler(capability)

        task_config = self.task_contract.config.parameters if self.task_contract else {}
        effective_config = self.system_contract.get_effective_config(task_config)

        output = handler.execute(
            self.system_contract,
            effective_config,
            input_data
        )

        if self.output_contract and not self.output_contract.validate(output):
            raise ValueError(f"Output validation failed for '{self.name}'")

        return output
