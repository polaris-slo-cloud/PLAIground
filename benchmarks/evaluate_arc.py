#!/usr/bin/env python3
import sys
import time
import json
import random
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

from datasets import load_dataset
from tqdm import tqdm

from .arc_workflow import create_arc_workflow
from selector import SLOConstraints


@dataclass
class QuestionResult:
    question_id: str
    difficulty: str
    correct_answer: str
    predicted_answer: str
    is_correct: bool
    latency_ms: float
    model_used: str


def load_arc_data(max_samples: int = None) -> List[Dict]:
    arc_easy = load_dataset("ai2_arc", "ARC-Easy", split="validation")
    arc_challenge = load_dataset("ai2_arc", "ARC-Challenge", split="validation")

    data = []
    for idx, item in enumerate(arc_easy):
        data.append({
            "id": f"easy_{idx}",
            "difficulty": "easy",
            "question": item["question"],
            "choices": item["choices"],
            "answer": item["answerKey"]
        })
    for idx, item in enumerate(arc_challenge):
        data.append({
            "id": f"hard_{idx}",
            "difficulty": "hard",
            "question": item["question"],
            "choices": item["choices"],
            "answer": item["answerKey"]
        })

    if max_samples:
        random.shuffle(data)
        data = data[:max_samples]

    return data


def format_question(q: Dict) -> str:
    text = f"{q['question']}\n"
    for i, choice in enumerate(q['choices']['text']):
        label = q['choices']['label'][i]
        text += f"{label}) {choice}\n"
    return text.strip()


def extract_answer(output: str) -> str:
    if not output:
        return ""
    output = output.strip().upper()
    # Check for standalone letter or letter followed by ) or .
    import re
    match = re.search(r'\b([ABCD])\b', output)
    if match:
        return match.group(1)
    # Fallback: first character if it's a letter
    if output and output[0] in 'ABCD':
        return output[0]
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector", default="adaptive",
                       choices=["adaptive", "greedy-cost", "greedy-quality", "greedy-latency"])
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-accuracy", type=float, default=0.8)
    parser.add_argument("--max-latency", type=float, default=2000)
    parser.add_argument("--max-cost", type=float, default=0.10)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)

    slo = SLOConstraints(
        min_accuracy=args.min_accuracy,
        max_p95_latency_ms=args.max_latency,
        max_total_cost=args.max_cost
    )

    print(f"Loading ARC dataset...")
    data = load_arc_data(args.samples)
    print(f"Loaded {len(data)} questions")

    print(f"Creating workflow with {args.selector} selector...")
    workflow = create_arc_workflow(args.selector, slo, len(data))

    print(f"\nRunning evaluation...")
    results = []
    model_usage = {}
    solver_usage = {"simple_solver": {}, "complex_solver": {}}

    for i, q in enumerate(tqdm(data)):
        formatted = format_question(q)

        start = time.time()
        result = workflow.run(formatted)
        latency_ms = (time.time() - start) * 1000

        predicted = extract_answer(result["answer"])
        is_correct = predicted == q["answer"]
        model_used = result.get("model_used", "unknown")
        solver_used = result.get("solver", "unknown")

        model_usage[model_used] = model_usage.get(model_used, 0) + 1
        if solver_used in solver_usage:
            solver_usage[solver_used][model_used] = solver_usage[solver_used].get(model_used, 0) + 1

        results.append(QuestionResult(
            question_id=q["id"],
            difficulty=q["difficulty"],
            correct_answer=q["answer"],
            predicted_answer=predicted,
            is_correct=is_correct,
            latency_ms=latency_ms,
            model_used=model_used
        ))

        if args.verbose and (i + 1) % 10 == 0:
            correct = sum(1 for r in results if r.is_correct)
            print(f"  [{i+1}/{len(data)}] acc={correct/(i+1):.1%}, last_model={model_used}")

    correct_total = sum(1 for r in results if r.is_correct)
    correct_easy = sum(1 for r in results if r.is_correct and r.difficulty == "easy")
    correct_hard = sum(1 for r in results if r.is_correct and r.difficulty == "hard")
    total_easy = sum(1 for r in results if r.difficulty == "easy")
    total_hard = sum(1 for r in results if r.difficulty == "hard")
    latencies = [r.latency_ms for r in results if r.latency_ms > 0]

    print(f"\n{'='*60}")
    print(f"RESULTS: {args.selector}")
    print(f"{'='*60}")
    acc_easy = f"{correct_easy/total_easy:.1%}" if total_easy else "N/A"
    acc_hard = f"{correct_hard/total_hard:.1%}" if total_hard else "N/A"
    print(f"Accuracy: {correct_total/len(results):.1%} (Easy: {acc_easy}, Hard: {acc_hard})")
    print(f"Latency: avg={np.mean(latencies):.0f}ms, p95={np.percentile(latencies, 95):.0f}ms")
    print(f"\nModel Usage by Solver:")
    for solver, models in solver_usage.items():
        if models:
            print(f"  {solver}:")
            for model, count in sorted(models.items(), key=lambda x: -x[1]):
                print(f"    {model}: {count}")

    summary = workflow.get_summary()
    print(f"\nSelector Summary:")
    print(f"  Total switches: {summary.get('total_switches', 0)}")
    print(f"  SLO compliant: {summary.get('slo_compliant', 'N/A')}")

    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "selector": args.selector,
                "samples": args.samples,
                "seed": args.seed,
                "slo": {"min_accuracy": args.min_accuracy, "max_latency": args.max_latency, "max_cost": args.max_cost}
            },
            "results": {
                "accuracy": correct_total / len(results),
                "accuracy_easy": correct_easy / total_easy if total_easy else 0,
                "accuracy_hard": correct_hard / total_hard if total_hard else 0,
                "avg_latency_ms": float(np.mean(latencies)),
                "p95_latency_ms": float(np.percentile(latencies, 95)),
                "model_usage": model_usage
            },
            "all_results": [asdict(r) for r in results]
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
