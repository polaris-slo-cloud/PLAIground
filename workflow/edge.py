from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable


@dataclass
class DataMapping:
    """Maps data between modules."""
    source_field: str
    target_field: str
    transform: Optional[Callable] = None

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.source_field not in data:
            return {}
        value = data[self.source_field]
        if self.transform:
            value = self.transform(value)
        return {self.target_field: value}


@dataclass
class Edge:
    """Connection between workflow nodes."""
    source: str
    target: str
    mappings: List[DataMapping] = field(default_factory=list)
    condition: Optional[str] = None

    def evaluate_condition(self, context: Dict[str, Any]) -> bool:
        if not self.condition:
            return True
        try:
            return eval(self.condition, {"__builtins__": {}}, context)
        except:
            return True
