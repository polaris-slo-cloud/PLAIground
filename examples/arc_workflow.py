"""ARC Router Workflow - Declarative Style.

Routes questions through classifier to appropriate solver:
- Easy questions -> simple_solver (GPT-3.5)
- Hard questions -> complex_solver (GPT-4)

Run with:
    python -m v2.examples.arc_workflow
"""

from typing import Dict, Any

from core.decorator import compoundable_model
from workflow import DeclarativeWorkflow, selector
from selector import AdaptiveSelector, SLOConstraints


@selector(
    AdaptiveSelector,
    slo=SLOConstraints(
        min_accuracy=0.80,
        max_p95_latency_ms=1000,
        max_total_cost=0.01
    )
)
class ARCRouterWorkflow(DeclarativeWorkflow):

    @compoundable_model(capability="classification")
    def classify_difficulty(self, data: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @compoundable_model(
        capability="llm",
        task_config={
            "prompt": (
                "Answer this easy ARC science question.\n"
                "Question: {question}\n"
                "Respond with only the letter (A, B, C, or D)."
            )
        }
    )
    def simple_solver(self, data: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @compoundable_model(
        capability="llm",
        task_config={
            "prompt": (
                "Answer this challenging ARC science question.\n"
                "Question: {question}\n"
                "Respond with only the letter (A, B, C, or D)."
            )
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
            "question": question,
            "difficulty": difficulty,
            "solver": solver_used,
            "answer": result.get("output", "")
        }


if __name__ == "__main__":
    print("ARC Router Workflow (Declarative)\n" + "=" * 50)

    workflow = ARCRouterWorkflow()

    questions = [
        "What is 2 + 2? A) 3 B) 4 C) 5 D) 6",
        "What process causes tectonic plates to move? A) Convection B) Radiation C) Conduction D) Insulation",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        result = workflow.run(q)
        print(f"Difficulty: {result['difficulty']}")
        print(f"Solver: {result['solver']}")
        print(f"Answer: {result['answer']}")

    print("\n" + "=" * 50)
    summary = workflow.get_summary()
    print(f"Total requests: {workflow.get_selector().requests_processed}")
    print(f"SLO compliant: {summary['slo_compliant']}")
