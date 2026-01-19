from abc import ABC, abstractmethod
from typing import Dict, Any, List, TypeVar, Generic

# Type variables for generic connector
InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')


class ConnectorError(Exception):
    """Base exception for connector errors."""
    pass


class InvocationError(ConnectorError):
    """Error during backend communication."""
    pass


class TranslationError(ConnectorError):
    """Error during input/output translation."""
    pass


class BackendConnector(ABC, Generic[InputT, OutputT]):
    """Abstract base class for all backend connectors.

    Implements the translation pipeline pattern:
    1. _translate_input: Convert standard input to backend format
    2. _transport: Send to backend and receive response
    3. _translate_output: Convert backend response to standard output

    The execute() method orchestrates this pipeline.
    """

    _verbose: bool = False  # Set to False to disable pipeline logging

    def execute(self, input_data: InputT) -> OutputT:
        """Execute the full translation pipeline.

        This is the template method that orchestrates:
        translate_input → transport → translate_output
        """
        connector_name = self.__class__.__name__
        try:
            if self._verbose:
                print(f"    [connector:{connector_name}] execute() started")
                print(f"    [connector:{connector_name}] step 1: _translate_input")
            payload = self._translate_input(input_data)

            if self._verbose:
                print(f"    [connector:{connector_name}] step 2: _transport")
            response = self._invoke(payload)

            if self._verbose:
                print(f"    [connector:{connector_name}] step 3: _translate_output")
            output = self._translate_output(response)

            if self._verbose:
                print(f"    [connector:{connector_name}] execute() completed")
            return output
        except ConnectorError:
            raise
        except Exception as e:
            raise ConnectorError(f"Execution failed: {e}") from e

    @abstractmethod
    def _translate_input(self, data: InputT) -> Any:
        """Convert standard input to backend-specific format.

        Args:
            data: Standard input (typically Dict from DataContract)

        Returns:
            Backend-specific payload (messages, tensors, etc.)
        """
        pass

    @abstractmethod
    def _invoke(self, payload: Any) -> Any:
        """Send payload to backend and receive response.

        Handles authentication, retries, and network errors.

        Args:
            payload: Backend-specific payload from _translate_input

        Returns:
            Raw backend response
        """
        pass

    @abstractmethod
    def _translate_output(self, response: Any) -> OutputT:
        """Convert backend response to standard output format.

        Args:
            response: Raw backend response from _transport

        Returns:
            Standard output (typically Dict matching DataContract)
        """
        pass

    @abstractmethod
    def close(self):
        """Release resources."""
        pass


class LLMConnector(BackendConnector[Dict[str, Any], Dict[str, Any]]):
    """Abstract base class for LLM connectors.

    Extends BackendConnector with backward-compatible generate() method.
    """

    _current_model: str = ""

    def generate(self, model_id: str, prompt: str, **kwargs) -> str:
        """Generate text from prompt (backward compatible).

        This method wraps execute() for backward compatibility.
        """
        self._current_model = model_id
        result = self.execute({"prompt": prompt, **kwargs})
        return result.get("output", "")


class ObjectDetectionConnector(BackendConnector[Dict[str, Any], Dict[str, Any]]):
    """Abstract base class for object detection connectors."""

    _current_model: str = ""

    def detect(self, model_id: str, image: str, **kwargs) -> List[Dict[str, Any]]:
        """Detect objects in image (backward compatible).

        Args:
            model_id: Model identifier
            image: Image path or base64 data

        Returns:
            List of detections: [{
                "bbox": [x1, y1, x2, y2],
                "label": str,
                "confidence": float
            }]
        """
        self._current_model = model_id
        result = self.execute({"image": image, **kwargs})
        return result.get("detections", [])
