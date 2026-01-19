import json
from pathlib import Path
from typing import Dict, List
from selector.base import Implementation
from selector.profile import ModelProfile


def load_profiles(profile_path: str = None) -> Dict[str, List[Implementation]]:
    if profile_path is None:
        profile_path = Path(__file__).parent.parent.parent / "data/benchmarks/arc_full_profiles.json"

    with open(profile_path) as f:
        data = json.load(f)

    implementations = []
    for key, p in data.get("profiles", {}).items():
        impl = Implementation(
            provider=p["provider"],
            model=p["model"],
            endpoint=p.get("deployment"),
            profile=ModelProfile(
                avg_latency_ms=p.get("avg_latency_ms", 1000),
                cost_per_call=p.get("cost_per_1k", 0) / 1000,
                accuracy=p.get("accuracy_overall"),
                p95_latency_ms=p.get("p95_latency_ms")
            )
        )
        implementations.append(impl)

    return {
        "classify_difficulty": implementations,
        "simple_solver": implementations,
        "complex_solver": implementations
    }
