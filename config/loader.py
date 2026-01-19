"""YAML configuration loader for implementations."""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any

from selector import Implementation, ModelProfile, SLOConstraints


class ConfigLoader:
    def __init__(self, config_path: Optional[str] = None):
        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = Path(__file__).parent / "implementations.yaml"

        self._config: Dict[str, Any] = {}
        self._profiles: Dict[str, Dict] = {}
        self._implementations: Dict[str, List[Implementation]] = {}

    def load(self) -> "ConfigLoader":
        if not self.config_path.exists():
            print(f"[config] No config file at {self.config_path}, using empty config")
            return self

        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f) or {}

        self._load_profiles()
        self._parse_implementations()

        return self

    def _load_profiles(self) -> None:
        profiles_path_str = self._config.get('profiles')
        if not profiles_path_str:
            return

        profiles_path = Path(__file__).parent.parent.parent / profiles_path_str
        if not profiles_path.exists():
            print(f"[config] Profiles file not found: {profiles_path}")
            return

        with open(profiles_path, 'r') as f:
            data = json.load(f)

        for key, profile in data.get('profiles', {}).items():
            provider = profile.get('provider')
            model = profile.get('model')
            if provider and model:
                model_clean = model.split('@')[0]
                lookup_key = f"{provider}/{model_clean}"
                self._profiles[lookup_key] = profile

        print(f"[config] Loaded {len(self._profiles)} profiles from {profiles_path_str}")

    def _get_profile(self, provider: str, model: str) -> Optional[ModelProfile]:
        lookup_key = f"{provider}/{model}"
        profile_data = self._profiles.get(lookup_key)

        if not profile_data:
            return None

        return ModelProfile(
            avg_latency_ms=profile_data.get('avg_latency_ms', 1000.0),
            cost_per_call=profile_data.get('cost_per_1k', 0) / 1000,
            accuracy=profile_data.get('accuracy_overall'),
            p95_latency_ms=profile_data.get('p95_latency_ms'),
            throughput_qps=profile_data.get('throughput_qps')
        )

    def _parse_implementations(self) -> None:
        tasks = self._config.get('tasks', {})

        for task_id, task_config in tasks.items():
            impls = []
            for impl_config in task_config.get('implementations', []):
                provider = impl_config['provider']
                model = impl_config['model']

                profile = self._get_profile(provider, model)
                if profile:
                    print(f"[config] {task_id}: {provider}/{model} -> profile loaded")
                else:
                    print(f"[config] {task_id}: {provider}/{model} -> no profile found")

                impl = Implementation(
                    provider=provider,
                    model=model,
                    endpoint=impl_config.get('endpoint'),
                    config=impl_config.get('config', {}),
                    profile=profile
                )
                impls.append(impl)

            self._implementations[task_id] = impls

    def get_implementations(self, task_id: str) -> List[Implementation]:
        return self._implementations.get(task_id, [])

    def get_all_implementations(self) -> Dict[str, List[Implementation]]:
        return self._implementations

    def get_slo_constraints(self) -> SLOConstraints:
        return SLOConstraints()


def load_implementations(config_path: Optional[str] = None) -> ConfigLoader:
    return ConfigLoader(config_path).load()
