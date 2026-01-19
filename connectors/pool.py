from typing import Dict, Any, Optional
from threading import Lock
from .base import BackendConnector


class ConnectorPool:

    _instance = None
    _lock = Lock()
    _connectors: Dict[str, BackendConnector]= {}
    _registry: Dict[str, Any] = {} # Dynamic Registry

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register_provider(cls, provider_name: str, connector_class: Any):

        with cls._lock:
            cls._registry[provider_name] = connector_class

    @classmethod
    def get(cls, provider: str, **kwargs) -> BackendConnector:
        key = f"{provider}_{hash(frozenset(kwargs.items()))}"

        if key not in cls._connectors:
            with cls._lock:
                if key not in cls._connectors:
                    cls._connectors[key] = cls._create(provider, **kwargs)

        return cls._connectors[key]

    @classmethod
    def _create(cls, provider: str, **kwargs) -> BackendConnector:

        # Dynamic Registry Implementation (New)
        if provider in cls._registry:
            return cls._registry[provider](**kwargs)

        # Backward Compatibility
        if provider == "openai":
            from .openai import OpenAIConnector
            return OpenAIConnector(**kwargs)
        elif provider == "ollama":
            from .ollama import OllamaConnector
            return OllamaConnector(**kwargs)
        elif provider in ("claude", "anthropic"):
            from .claude import ClaudeConnector
            return ClaudeConnector(**kwargs)
        elif provider == "custom":
            from .custom_classifier import CustomClassifierConnector
            return CustomClassifierConnector(**kwargs)
        elif provider == "triton":
            from .triton import TritonConnector
            return TritonConnector(**kwargs)
        elif provider == "torchserve":
            from .torchserve import TorchServeConnector
            return TorchServeConnector(**kwargs)

        raise ValueError(f"Unknown provider: {provider}")

    @classmethod
    def clear(cls):
        with cls._lock:
            for connector in cls._connectors.values():
                try:
                    connector.close()
                except Exception as e:
                    print(f"Error closing connector: {e}")
            cls._connectors.clear()
