"""Triton Inference Server Connector with translation pipeline.

Generic infrastructure adapter - NOT task-specific.
Can serve ANY model deployed on Triton.
"""

from typing import Dict, Any, List, Optional

from .base import ObjectDetectionConnector, InvocationError

# Graceful import - don't fail if tritonclient not installed
TRITON_AVAILABLE = False
try:
    import tritonclient.http as httpclient
    import tritonclient.grpc as grpcclient
    import numpy as np
    TRITON_AVAILABLE = True
except ImportError:
    httpclient = None
    grpcclient = None
    np = None


class TritonConnector(ObjectDetectionConnector):
    """Adapter to NVIDIA Triton Inference Server with translation pipeline.

    This connector knows NOTHING about the task - it just:
    1. Translates input dict → Triton tensor format
    2. Sends to Triton server
    3. Translates Triton response → output dict

    Translation:
    - Input: {"image": ..., "input": ...} → Triton InferInput tensors
    - Output: Triton InferResult → {"detections": [...], "output": ...}
    """

    def __init__(
        self,
        endpoint: str = "localhost:8000",
        protocol: str = "http",
        model_name: Optional[str] = None,
        model_version: str = "1"
    ):
        self.endpoint = endpoint
        self.protocol = protocol
        self.model_name = model_name
        self.model_version = model_version
        self._client = None
        self._metadata = None
        self._mock_mode = not TRITON_AVAILABLE

        if TRITON_AVAILABLE:
            self._init_client()

    def _init_client(self):
        """Initialize Triton client."""
        try:
            if self.protocol == "http":
                self._client = httpclient.InferenceServerClient(url=self.endpoint)
            else:
                self._client = grpcclient.InferenceServerClient(url=self.endpoint)

            # Check if server is available
            if not self._client.is_server_live():
                print(f"[triton] Server not live at {self.endpoint} - running in mock mode")
                self._mock_mode = True
        except Exception as e:
            print(f"[triton] Failed to connect: {e} - running in mock mode")
            self._mock_mode = True

    def health_check(self) -> bool:
        """Check if Triton server is healthy."""
        if self._mock_mode or self._client is None:
            return False
        try:
            return self._client.is_server_live()
        except Exception:
            return False

    def _translate_input(self, data: Dict[str, Any]) -> List:
        """Convert input dict to Triton InferInput tensors.

        Input: {"image": np.array, "input": np.array, ...}
        Output: List[InferInput] ready for Triton
        """
        if self._mock_mode:
            return data  # Pass through for mock

        target_model = self._current_model or self.model_name
        if not target_model:
            raise ValueError("No model specified")

        # Get metadata if not cached
        if self._metadata is None:
            self._metadata = self._client.get_model_metadata(target_model, self.model_version)

        inputs = []
        for inp in self._metadata.inputs:
            inp_name = inp.name
            if inp_name in data:
                tensor_data = data[inp_name]

                # Convert to numpy if needed
                if not isinstance(tensor_data, np.ndarray):
                    tensor_data = np.array(tensor_data)

                if self.protocol == "http":
                    triton_input = httpclient.InferInput(inp_name, tensor_data.shape, inp.datatype)
                    triton_input.set_data_from_numpy(tensor_data)
                else:
                    triton_input = grpcclient.InferInput(inp_name, tensor_data.shape, inp.datatype)
                    triton_input.set_data_from_numpy(tensor_data)

                inputs.append(triton_input)

        return inputs

    def _invoke(self, payload: Any) -> Any:
        """Send tensors to Triton and receive response.

        Handles mock mode when Triton is unavailable.
        """
        if self._mock_mode:
            return self._mock_response(payload)

        target_model = self._current_model or self.model_name

        try:
            response = self._client.infer(
                model_name=target_model,
                model_version=self.model_version,
                inputs=payload
            )
            return response
        except Exception as e:
            raise InvocationError(f"Triton inference failed: {e}") from e

    def _translate_output(self, response: Any) -> Dict[str, Any]:
        """Convert Triton response to standard output dict.

        Input: Triton InferResult or mock dict
        Output: {"detections": [...], "has_detection": bool, "output": str}
        """
        if isinstance(response, dict) and response.get("_mock"):
            return response

        # Parse real Triton response
        outputs = {}
        if self._metadata:
            for out in self._metadata.outputs:
                out_name = out.name
                try:
                    outputs[out_name] = response.as_numpy(out_name)
                except Exception:
                    pass

        # Normalize to standard format
        detections = outputs.get("detections", [])
        if isinstance(detections, np.ndarray):
            detections = detections.tolist()

        return {
            "detections": detections,
            "has_detection": len(detections) > 0,
            "output": "detected" if detections else "none",
            **outputs
        }

    def _mock_response(self, input_data: Any) -> Dict[str, Any]:
        """Generate mock response when Triton unavailable."""
        import random

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

    def generate(self, model_id: str, prompt: str, **kwargs) -> str:
        """LLMConnector interface - for text models on Triton."""
        self._current_model = model_id
        result = self.execute({"prompt": prompt, **kwargs})
        return result.get("output", str(result))

    def close(self):
        """Release resources."""
        self._client = None
        self._metadata = None
