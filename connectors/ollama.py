"""Ollama connector with translation pipeline."""

import os
from typing import Optional, Dict, Any

from .base import LLMConnector, InvocationError

try:
    import requests
except ImportError:
    requests = None


class OllamaConnector(LLMConnector):
    """Ollama API connector with translation pipeline.

    Translation:
    - Input: {"prompt": str, ...} → Ollama JSON payload
    - Output: Ollama response → {"output": str}
    """

    def __init__(self, base_url: Optional[str] = None, timeout: int = 120):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip('/')
        self.timeout = timeout
        self._mock_mode = False
        self.session = None

        if requests is None:
            self._mock_mode = True
        else:
            self.session = requests.Session()
            # Check if Ollama is available
            try:
                self.session.get(f"{self.base_url}/api/tags", timeout=2)
            except Exception:
                self._mock_mode = True

    def _translate_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert standard input to Ollama JSON format.

        Input: {"prompt": str, "temperature": float, "max_tokens": int, ...}
        Output: {"model": str, "prompt": str, "stream": False, "options": {...}}
        """
        return {
            "model": self._current_model,
            "prompt": data.get("prompt", ""),
            "stream": False,
            "options": {
                "temperature": data.get("temperature", 0.7),
                "num_predict": data.get("max_tokens", 1000),
            }
        }

    def _invoke(self, payload: Dict[str, Any]) -> Any:
        """Call Ollama HTTP API.

        Handles mock mode when Ollama is unavailable.
        """
        if self._mock_mode:
            return {"mock": True, "content": f"[Mock {payload.get('model', 'ollama')}] Response for: {payload.get('prompt', '')[:50]}..."}

        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise InvocationError(f"Ollama API call failed: {e}") from e

    def _translate_output(self, response: Any) -> Dict[str, Any]:
        """Convert Ollama response to standard output format.

        Input: Ollama JSON response or mock dict
        Output: {"output": str}
        """
        if isinstance(response, dict) and response.get("mock"):
            return {"output": response["content"]}

        return {"output": response.get("response", "").strip()}

    def close(self):
        """Release resources."""
        if self.session:
            self.session.close()
