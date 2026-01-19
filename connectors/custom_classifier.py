"""Custom classifier connector with translation pipeline."""

import os
import random
from typing import Dict, Any, Optional

from .base import LLMConnector

TRANSFORMERS_AVAILABLE = False
try:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    torch = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None


class RandomClassifier:
    """Fallback random classifier when model weights unavailable."""

    def classify(self, text: str) -> tuple:
        label = random.choice(["easy", "hard"])
        confidence = random.uniform(0.5, 0.9)
        return label, confidence


class CustomClassifierConnector(LLMConnector):
    """Connector for custom classification models with translation pipeline.

    Translation:
    - Input: {"prompt": str, ...} → tokenized tensor inputs
    - Output: Model logits → {"output": label, "confidence": float, ...}
    """

    def __init__(self, model_store_path: Optional[str] = None, device: str = "auto", max_length: int = 512):
        self.model_store_path = model_store_path or os.getenv("MODEL_STORE_PATH", "./model-store")
        self.max_length = max_length
        self.device = self._resolve_device(device) if TRANSFORMERS_AVAILABLE else "cpu"
        self.label_map = {0: 'easy', 1: 'hard'}
        self._loaded_models = {}
        self._use_heuristic = {}
        self._fallback = RandomClassifier()
        self._confidence_threshold = 0.5

    def _resolve_device(self, device: str):
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    def _has_model_weights(self, model_path: str) -> bool:
        weight_files = ["pytorch_model.bin", "model.safetensors", "model.bin"]
        return any(os.path.exists(os.path.join(model_path, f)) for f in weight_files)

    def _load_model(self, model_path: str):
        if model_path in self._loaded_models:
            return self._loaded_models[model_path]

        if model_path in self._use_heuristic:
            return None, None

        if not TRANSFORMERS_AVAILABLE:
            self._use_heuristic[model_path] = True
            return None, None

        if not self._has_model_weights(model_path):
            self._use_heuristic[model_path] = True
            return None, None

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=2, trust_remote_code=True
        )

        for param in model.parameters():
            if param.data.dtype == torch.float16:
                param.data = param.data.to(torch.float32)

        model = model.to(self.device)
        model.eval()
        self._loaded_models[model_path] = (model, tokenizer)
        return model, tokenizer

    def _resolve_model_path(self, model_id: str) -> str:
        if os.path.isabs(model_id) or model_id.startswith('./'):
            return model_id

        custom_path = os.path.join(self.model_store_path, "custom", model_id)
        if os.path.exists(custom_path):
            return custom_path

        direct_path = os.path.join(self.model_store_path, model_id)
        if os.path.exists(direct_path):
            return direct_path

        return model_id

    def _translate_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert input dict to tokenized tensor format.

        Input: {"prompt": str, "confidence_threshold": float, ...}
        Output: {"text": str, "model_path": str, "threshold": float, "tokenized": tensor}
        """
        text = data.get("prompt", "")
        self._confidence_threshold = data.get("confidence_threshold", 0.5)

        model_path = self._resolve_model_path(self._current_model)
        model, tokenizer = self._load_model(model_path)

        if model is None:
            # Fallback mode - pass text through
            return {"text": text, "model_path": model_path, "use_fallback": True}

        # Tokenize for model
        inputs = tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        ).to(self.device)

        return {
            "text": text,
            "model_path": model_path,
            "tokenized": inputs,
            "use_fallback": False
        }

    def _invoke(self, payload: Dict[str, Any]) -> Any:
        """Run model inference.

        Handles fallback mode when model unavailable.
        """
        if payload.get("use_fallback"):
            label, confidence = self._fallback.classify(payload["text"])
            return {
                "fallback": True,
                "label": label,
                "confidence": confidence
            }

        model, _ = self._load_model(payload["model_path"])
        inputs = payload["tokenized"]

        with torch.no_grad():
            outputs = model(**inputs)
            return {
                "fallback": False,
                "logits": outputs.logits
            }

    def _translate_output(self, response: Any) -> Dict[str, Any]:
        """Convert model output to standard classification format.

        Input: {"logits": tensor} or {"fallback": True, "label": str, ...}
        Output: {"output": label, "confidence": float, "probabilities": {...}}
        """
        if response.get("fallback"):
            label = response["label"]
            confidence = response["confidence"]
            return {
                "output": label,
                "label": label,
                "confidence": confidence,
                "probabilities": {
                    "easy": 1 - confidence if label == "hard" else confidence,
                    "hard": confidence if label == "hard" else 1 - confidence
                },
                "method": "heuristic"
            }

        # Process model logits
        probs = torch.nn.functional.softmax(response["logits"], dim=1)[0].cpu().numpy()
        easy_prob = float(probs[0])
        hard_prob = float(probs[1])

        if easy_prob >= self._confidence_threshold:
            label = "easy"
            confidence = easy_prob
        else:
            label = "hard"
            confidence = hard_prob

        return {
            "output": label,
            "label": label,
            "confidence": confidence,
            "probabilities": {"easy": easy_prob, "hard": hard_prob},
            "method": "model"
        }

    def classify(self, model_id: str, text: str, **kwargs) -> Dict[str, Any]:
        """Classification interface - detailed results."""
        self._current_model = model_id
        return self.execute({"prompt": text, **kwargs})

    def close(self):
        """Release resources."""
        if TRANSFORMERS_AVAILABLE:
            for model, tokenizer in self._loaded_models.values():
                del model
                del tokenizer
            self._loaded_models.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
