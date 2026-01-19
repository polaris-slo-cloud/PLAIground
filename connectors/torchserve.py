"""TorchServe Connector with translation pipeline.

Generic infrastructure adapter - NOT task-specific.
Can serve ANY model deployed on TorchServe.
"""

import random
from typing import Dict, Any, Optional, Tuple

from .base import ObjectDetectionConnector, InvocationError

# Graceful import
REQUESTS_AVAILABLE = False
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None


class TorchServeConnector(ObjectDetectionConnector):
    """Adapter to PyTorch TorchServe with translation pipeline.

    This connector knows NOTHING about the task - it just:
    1. Translates input dict → HTTP request format
    2. Sends to TorchServe endpoint
    3. Translates HTTP response → output dict

    Translation:
    - Input: {"image": ..., "data": ...} → HTTP request (binary or JSON)
    - Output: TorchServe response → {"detections": [...], "output": ...}
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:8080",
        model_name: Optional[str] = None,
        timeout: int = 30
    ):
        self.endpoint = endpoint.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self._mock_mode = not REQUESTS_AVAILABLE

        if REQUESTS_AVAILABLE:
            # Check if TorchServe is available
            try:
                management_endpoint = self.endpoint.replace(":8080", ":8081")
                requests.get(f"{management_endpoint}/ping", timeout=2)
            except Exception:
                print(f"[torchserve] Server not available at {self.endpoint} - running in mock mode")
                self._mock_mode = True

    def health_check(self) -> bool:
        """Check if TorchServe is healthy."""
        if self._mock_mode:
            return False
        try:
            management_endpoint = self.endpoint.replace(":8080", ":8081")
            response = requests.get(f"{management_endpoint}/ping", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def _translate_input(self, data: Dict[str, Any]) -> Tuple[Any, Dict[str, str]]:
        """Convert input dict to HTTP request format.

        Input: {"image": path/bytes, "data": ..., ...}
        Output: (body, headers) tuple ready for HTTP request

        Detects content type:
        - Image data → binary with octet-stream header
        - Other → JSON
        """
        if self._mock_mode:
            return (data, {})

        # Check for image data
        if "image" in data or "image_bytes" in data:
            image_data = data.get("image") or data.get("image_bytes")

            # Read file if path provided
            if isinstance(image_data, str):
                try:
                    with open(image_data, "rb") as f:
                        image_data = f.read()
                except FileNotFoundError:
                    # Treat as raw data if file not found
                    image_data = image_data.encode() if isinstance(image_data, str) else image_data

            return (image_data, {"Content-Type": "application/octet-stream"})

        # Default to JSON
        return (data, {"Content-Type": "application/json"})

    def _invoke(self, payload: Tuple[Any, Dict[str, str]]) -> Any:
        """Send HTTP request to TorchServe.

        Handles mock mode when TorchServe unavailable.
        """
        if self._mock_mode:
            return self._mock_response(payload[0])

        body, headers = payload
        target_model = self._current_model or self.model_name

        if not target_model:
            raise ValueError("No model specified")

        url = f"{self.endpoint}/predictions/{target_model}"

        try:
            if headers.get("Content-Type") == "application/json":
                response = requests.post(url, json=body, headers=headers, timeout=self.timeout)
            else:
                response = requests.post(url, data=body, headers=headers, timeout=self.timeout)

            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise InvocationError(f"TorchServe request failed: {e}") from e

    def _translate_output(self, response: Any) -> Dict[str, Any]:
        """Convert TorchServe response to standard output dict.

        Input: JSON response (list or dict) or mock dict
        Output: {"detections": [...], "has_detection": bool, "output": ...}
        """
        if isinstance(response, dict) and response.get("_mock"):
            return response

        # TorchServe can return list (detections) or dict
        if isinstance(response, list):
            return {
                "detections": response,
                "has_detection": len(response) > 0,
                "output": response
            }
        elif isinstance(response, dict):
            detections = response.get("detections", [])
            return {
                "detections": detections,
                "has_detection": len(detections) > 0,
                "output": response.get("output", response),
                **response
            }
        else:
            return {"output": response}

    def _mock_response(self, input_data: Any) -> Dict[str, Any]:
        """Generate mock response when TorchServe unavailable."""
        # Check if image-like input
        is_image = isinstance(input_data, (bytes, bytearray)) or (
            isinstance(input_data, dict) and ("image" in input_data or "image_bytes" in input_data)
        )

        if is_image:
            has_detection = random.random() < 0.6
            detections = []
            if has_detection:
                num_detections = random.randint(1, 3)
                for _ in range(num_detections):
                    detections.append({
                        "bbox": [
                            random.randint(50, 300),
                            random.randint(50, 200),
                            random.randint(100, 200),
                            random.randint(100, 200)
                        ],
                        "label": random.choice(["object_1", "object_2", "object_3"]),
                        "confidence": round(random.uniform(0.6, 0.95), 2)
                    })

            return {
                "detections": detections,
                "has_detection": len(detections) > 0,
                "output": "detected" if detections else "none",
                "_mock": True
            }

        # Generic response
        return {
            "output": f"Mock response from {self._current_model or self.model_name}",
            "_mock": True
        }

    def generate(self, model_id: str, prompt: str, **kwargs) -> str:
        """LLMConnector interface - for text models on TorchServe."""
        self._current_model = model_id
        result = self.execute({"data": prompt, **kwargs})
        return result.get("output", str(result))

    def close(self):
        """Release resources."""
        pass
