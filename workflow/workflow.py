import networkx as nx
from typing import Dict, Any, List, Optional
from .edge import Edge, DataMapping
from core.model import CompoundableModel


class Workflow:
    """DAG-based workflow of Compoundable Models."""

    def __init__(self, name: str):
        self.name = name
        self.graph = nx.DiGraph()
        self.models: Dict[str, CompoundableModel] = {}
        self.edges: List[Edge] = []

    def add_model(self, node_id: str, model: CompoundableModel):
        self.models[node_id] = model
        self.graph.add_node(node_id)

    def add_edge(self, source: str, target: str,
                 mappings: List[DataMapping] = None,
                 condition: str = None):
        edge = Edge(source, target, mappings or [], condition)
        self.edges.append(edge)
        self.graph.add_edge(source, target)

    def get_execution_order(self) -> List[str]:
        return list(nx.topological_sort(self.graph))

    def execute(self, input_data: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
        context = input_data.copy()
        results = {}
        execution_path = []
        routed_nodes = set()

        for node_id in self.get_execution_order():
            if node_id == "input" or node_id not in self.models:
                continue

            model = self.models[node_id]
            if not model.is_bound():
                raise RuntimeError(f"Model '{node_id}' not bound")

            incoming = [e for e in self.edges if e.target == node_id]
            should_execute = self._should_execute_node(node_id, incoming, context, routed_nodes, verbose)

            if not should_execute:
                continue

            module_input = self._prepare_input(node_id, context, results)
            result = model.execute(module_input)
            results[node_id] = result
            execution_path.append(node_id)

            context[f'{node_id}_output'] = result.get('output', result)
            context['output'] = result.get('output', result)

            for edge in self.edges:
                if edge.source == node_id:
                    routed_nodes.add(edge.source)

        return {
            'results': results,
            'final_output': context.get('output'),
            'context': context,
            'execution_path': execution_path
        }

    def _should_execute_node(self, node_id: str, incoming: List[Edge],
                             context: Dict, routed_nodes: set, verbose: bool = False) -> bool:
        if not incoming:
            return True

        conditional_edges = [e for e in incoming if e.condition]
        default_edges = [e for e in incoming if not e.condition]

        for edge in conditional_edges:
            if edge.evaluate_condition(context):
                return True

        if default_edges:
            for edge in default_edges:
                source_node = edge.source
                siblings = [e for e in self.edges if e.source == source_node and e.target != node_id]
                for sibling in siblings:
                    if sibling.condition and sibling.evaluate_condition(context):
                        return False
            return True

        return False

    def _prepare_input(self, node_id: str, context: Dict, results: Dict) -> Dict[str, Any]:
        incoming = [e for e in self.edges if e.target == node_id]

        if not incoming:
            return context.copy()

        module_input = {}
        for edge in incoming:
            if edge.source == "input":
                module_input.update(context)
            elif edge.source in results:
                if edge.condition:
                    module_input['prompt'] = context.get('prompt', '')
                else:
                    source_out = results[edge.source]
                    if edge.mappings:
                        for m in edge.mappings:
                            module_input.update(m.apply(source_out))
                    else:
                        module_input.update(source_out)

        if 'prompt' not in module_input and 'prompt' in context:
            module_input['prompt'] = context['prompt']

        return module_input
