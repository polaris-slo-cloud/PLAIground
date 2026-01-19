from typing import Dict, Any
from core.decorator import compoundable_model
from workflow import DeclarativeWorkflow, selector
from selector import AdaptiveSelector, GreedySelector, SLOConstraints
from selector.greedy import Strategy


def create_arc_workflow(
    selector_type: str = "adaptive",
    slo: SLOConstraints = None,
    total_requests: int = 100
):
    slo = slo or SLOConstraints(
        min_accuracy=0.8,
        max_p95_latency_ms=2000,
        max_total_cost=0.10
    )

    kwargs = {}
    if selector_type == "adaptive":
        sel_class = AdaptiveSelector
        kwargs["total_requests"] = total_requests
    elif selector_type == "greedy-quality":
        sel_class = GreedySelector
        kwargs["strategy"] = Strategy.QUALITY
    elif selector_type == "greedy-cost":
        sel_class = GreedySelector
        kwargs["strategy"] = Strategy.COST
    elif selector_type == "greedy-latency":
        sel_class = GreedySelector
        kwargs["strategy"] = Strategy.LATENCY
    else:
        raise ValueError(f"Unknown selector: {selector_type}")

    @selector(sel_class, slo=slo, **kwargs)
    class ARCRouterWorkflow(DeclarativeWorkflow):

        @compoundable_model(capability="classification")
        def classify_difficulty(self, data: Dict[str, Any]) -> Dict[str, Any]:
            pass

        @compoundable_model(
            capability="llm",
            task_config={
                "prompt": "Answer this easy ARC science question.\nQuestion: {question}\nRespond with only the letter (A, B, C, or D)."
            }
        )
        def simple_solver(self, data: Dict[str, Any]) -> Dict[str, Any]:
            pass

        @compoundable_model(
            capability="llm",
            task_config={
                "prompt": "Answer this challenging ARC science question.\nQuestion: {question}\nRespond with only the letter (A, B, C, or D)."
            }
        )
        def complex_solver(self, data: Dict[str, Any]) -> Dict[str, Any]:
            pass

        def run(self, question: str) -> Dict[str, Any]:
            classification = self.classify_difficulty({"prompt": question})
            difficulty = classification.get("output", "hard")

            if difficulty == "easy":
                result = self.simple_solver({"question": question})
                solver_used = "simple_solver"
            else:
                result = self.complex_solver({"question": question})
                solver_used = "complex_solver"

            return {
                "difficulty": difficulty,
                "solver": solver_used,
                "answer": result.get("output", ""),
                "model_used": str(self.get_selector().get_current(solver_used))
            }

    return ARCRouterWorkflow()
