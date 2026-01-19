from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Type


@dataclass
class DataContract:
    """Data Contract specifying input/output schema for Compoundable Models."""

    schema: Dict[str, Type]
    required: Optional[List[str]] = None

    def __post_init__(self):
        if self.required is None:
            self.required = list(self.schema.keys())

    def validate(self, data: Dict[str, Any]) -> bool:
        for field_name in self.required:
            if field_name not in data:
                return False
        for field_name, expected_type in self.schema.items():
            if field_name in data and not isinstance(data[field_name], expected_type):
                if expected_type == str and data[field_name] is not None:
                    continue
                return False
        return True

    def __call__(self, schema: Dict[str, Type]) -> 'DataContract':
        return DataContract(schema=schema)


def extract_contracts(func, skip_self: bool = False) -> tuple:
    """Extract DataContract from function annotations.

    Args:
        func: Function to extract contracts from
        skip_self: If True, skip 'self' parameter (for methods)

    Returns:
        Tuple of (input_contract, output_contract)
    """
    annotations = getattr(func, '__annotations__', {})

    input_contract = None
    output_contract = None

    for name, annotation in annotations.items():
        if skip_self and name == 'self':
            continue
        if name == 'return':
            if isinstance(annotation, DataContract):
                output_contract = annotation
        elif isinstance(annotation, DataContract):
            input_contract = annotation

    return input_contract, output_contract
