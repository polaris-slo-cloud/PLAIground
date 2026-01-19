from abc import ABC, abstractmethod
from typing import Dict, Any

from contracts.system_contract import SystemContract


class HandlerError(Exception):
    pass


class CapabilityHandler(ABC):

    @abstractmethod
    def execute(self, system_contract: SystemContract, config: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
        pass

class LLMHandler(CapabilityHandler):

    def execute(self, system_contract: SystemContract, config: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:

        prompt_template = config.get("prompt")

        if prompt_template:
            try:
                prompt = prompt_template.format(**input_data)
            except KeyError as e:
                raise ValueError(f"Missing input key for prompt template: {e}")
        else:
            prompt = input_data.get("prompt", str(input_data))

        exec_config = config.copy()
        exec_config.pop("prompt", None)

        # Backward compatibility
        if hasattr(system_contract.connector, 'generate'):
            result = system_contract.connector.generate(system_contract.model_id, prompt, **exec_config)
            return {"output": result}

        return system_contract.connector.execute({"prompt": prompt, **config})

class ObjectDetectionHandler(CapabilityHandler):

    def execute(self, system_contract: SystemContract, config: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:

        # This is specificity to certain connector implementations
        connector = system_contract.connector

        if hasattr(connector, '_current_model'):
            connector._current_model = system_contract.model_id

        # Backward compatibility
        if hasattr(connector, 'detect'):
            image = input_data.get("image", "")
            config = system_contract.get_effective_config({})
            detections = connector.detect(system_contract.model_id, image, **config)
        else:
            result = connector.execute(input_data)
            detections = result.get("detections", [])

        has_detection = len(detections) > 0
        return {
            "detections": detections,
            "has_detection": has_detection,
            "output": "detected" if has_detection else "none"
        }

class CustomClassificationHandler(CapabilityHandler):
    def execute(self, system_contract: SystemContract, config: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:

        connector = system_contract.connector

        prompt_template = config.get("prompt")
        if prompt_template:
            prompt = prompt_template.format(**input_data)
        else:
            prompt = input_data.get("prompt", str(input_data))

        if hasattr(connector, 'classify'):
            result = connector.classify(system_contract.model_id, prompt, **config)
        elif hasattr(connector, 'execute'):
            result = connector.execute({"prompt": prompt, **config})
        else:
            output_text = connector.generate(system_contract.model_id, prompt, **config)
            result = {"output": output_text}

        if "output" not in result:
            result["output"] = result.get("label", str(result))

        return result

class CapabilityRegistry:

    _handlers: Dict[str, CapabilityHandler] = {
        "llm": LLMHandler(),
        "object_detection": ObjectDetectionHandler(),
        "classification": CustomClassificationHandler(),
    }

    @classmethod
    def register(cls, capability:str, handler:CapabilityHandler) -> None:
        cls._handlers[capability] = handler

    @classmethod
    def get_handler(cls, capability:str) -> CapabilityHandler:
        handler = cls._handlers.get(capability)
        if not handler:
            raise HandlerError("Unknown capability {}".format(capability))

        return handler