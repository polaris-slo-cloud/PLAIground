"""OpenAI connector with translation pipeline."""

import os
from typing import Optional, Dict, Any

from .base import LLMConnector, InvocationError

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class OpenAIConnector(LLMConnector):
    """OpenAI API connector with translation pipeline.

    Translation:
    - Input: {"prompt": str, ...} → OpenAI messages format
    - Output: OpenAI response → {"output": str}
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._mock_mode = False
        self.client = None

        if not self.api_key:
            self._mock_mode = True
        elif OpenAI is None:
            self._mock_mode = True
        else:
            self.client = OpenAI(api_key=self.api_key, base_url=base_url)

    def _translate_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert standard input to OpenAI messages format.

        Input: {"prompt": str, "temperature": float, "max_tokens": int, ...}
        Output: {"model": str, "messages": [...], "temperature": float, ...}
        """
        prompt = data.get("prompt", "")

        return {
            "model": self._current_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": data.get("temperature", 0.7),
            "max_tokens": data.get("max_tokens", 1000),
        }

    def _invoke(self, payload: Dict[str, Any]) -> Any:
        """Call OpenAI API.

        Handles mock mode for testing without API key.
        """
        if self._mock_mode:
            # Return mock response structure
            prompt = payload["messages"][0]["content"] if payload.get("messages") else ""
            return {"mock": True, "content": f"[Mock {payload.get('model', 'gpt')}] Response for: {prompt[:50]}..."}

        try:
            return self.client.chat.completions.create(**payload)
        except Exception as e:
            raise InvocationError(f"OpenAI API call failed: {e}") from e

    def _translate_output(self, response: Any) -> Dict[str, Any]:
        """Convert OpenAI response to standard output format.

        Input: OpenAI ChatCompletion or mock dict
        Output: {"output": str}
        """
        if isinstance(response, dict) and response.get("mock"):
            return {"output": response["content"]}

        return {"output": response.choices[0].message.content.strip()}

    def close(self):
        """Release resources."""
        if self.client and hasattr(self.client, 'close'):
            self.client.close()
