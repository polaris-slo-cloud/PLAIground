from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from connectors.base import BackendConnector

@dataclass
class SystemContract:

    provider: str
    model_id: str
    connector: BackendConnector
    execution_config: Dict[str, Any] = field(default_factory=dict)

    def get_effective_config(self, task_parameters: Dict[str, Any]) -> Dict[str, Any]:
        return {**task_parameters, **self.execution_config}