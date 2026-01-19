from typing import List, Optional
from .workflow import Workflow
from .edge import Edge, DataMapping
from core.model import CompoundableModel


class WorkflowBuilder:
    """Fluent API for building workflows."""

    def __init__(self, name: str):
        self.workflow = Workflow(name)

    def add(self, node_id: str, model: CompoundableModel) -> 'WorkflowBuilder':
        self.workflow.add_model(node_id, model)
        return self

    def connect(self, source: str, target: str,
                condition: str = None) -> 'WorkflowBuilder':
        self.workflow.add_edge(source, target, condition=condition)
        return self

    def pipeline(self, *node_ids: str) -> 'WorkflowBuilder':
        for i in range(len(node_ids) - 1):
            self.connect(node_ids[i], node_ids[i + 1])
        return self

    def route(self, source: str) -> 'RouterBuilder':
        return RouterBuilder(self, source)

    def build(self) -> Workflow:
        if "input" not in self.workflow.graph:
            self.workflow.graph.add_node("input")
        return self.workflow


class RouterBuilder:
    """Builder for conditional routing."""

    def __init__(self, builder: WorkflowBuilder, source: str):
        self.builder = builder
        self.source = source

    def when(self, condition: str, target: str) -> 'RouterBuilder':
        self.builder.workflow.add_edge(self.source, target, condition=condition)
        return self

    def default(self, target: str) -> WorkflowBuilder:
        self.builder.workflow.add_edge(self.source, target)
        return self.builder
