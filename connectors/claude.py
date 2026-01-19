"""Claude (Anthropic) connector with translation pipeline."""

import os
from typing import Optional, Dict, Any

from .base import LLMConnector, InvocationError

try:
    import anthropic
except ImportError:
    anthropic = None


class ClaudeConnector(LLMConnector):
    """Anthropic Claude API connector with translation pipeline.

    Translation:
    - Input: {"prompt": str, ...} → Claude messages format
    - Output: Claude response → {"output": str}
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._mock_mode = False
        self.client = None

        if anthropic is None:
            self._mock_mode = True
        elif not self.api_key:
            self._mock_mode = True
        else:
            self.client = anthropic.Anthropic(api_key=self.api_key, base_url=base_url)

    def _translate_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert standard input to Claude messages format.

        Input: {"prompt": str, "temperature": float, "max_tokens": int, ...}
        Output: {"model": str, "messages": [...], "temperature": float, ...}
        """
        return {
            "model": self._current_model,
            "messages": [{"role": "user", "content": data.get("prompt", "")}],
            "temperature": data.get("temperature", 0.7),
            "max_tokens": data.get("max_tokens", 1000),
        }

    def _invoke(self, payload: Dict[str, Any]) -> Any:
        """Call Claude API.

        Handles mock mode for testing without API key.
        """
        if self._mock_mode:
            prompt = payload["messages"][0]["content"] if payload.get("messages") else ""
            return {"mock": True, "content": f"[Mock {payload.get('model', 'claude')}] Response for: {prompt[:50]}..."}

        try:
            return self.client.messages.create(**payload)
        except Exception as e:
            raise InvocationError(f"Claude API call failed: {e}") from e

    def _translate_output(self, response: Any) -> Dict[str, Any]:
        """Convert Claude response to standard output format.

        Input: Claude Message or mock dict
        Output: {"output": str}
        """
        if isinstance(response, dict) and response.get("mock"):
            return {"output": response["content"]}

        return {"output": response.content[0].text.strip()}

    def close(self):
        """Release resources."""
        if self.client and hasattr(self.client, 'close'):
            self.client.close()
