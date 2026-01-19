from typing import Dict, Any
from .workflow import Workflow


class WorkflowExecutor:
    """Executes workflows with optional hooks."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def execute(self, workflow: Workflow, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if self.verbose:
            print(f"Executing workflow: {workflow.name}")
            print(f"Models: {list(workflow.models.keys())}")

        result = workflow.execute(input_data, verbose=self.verbose)

        if self.verbose:
            print(f"Completed. Final output: {str(result.get('final_output', ''))[:100]}...")

        return result
