from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

try:
    from transformers import pipeline
except ImportError:  # pragma: no cover - handled at runtime for environments without deps
    pipeline = None


@dataclass
class StructuredCommand:
    raw_input: str
    intent: str
    device: str | None
    action: str | None
    value: str | int | None
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HuggingFaceCommandParser:
    INTENT_LABELS = [
        "turn_on_device",
        "turn_off_device",
        "set_fan_speed",
        "set_temperature",
        "get_device_status",
        "get_temperature",
    ]

    DEVICE_KEYWORDS = {
        "light": ("light", "lamp"),
        "fan": ("fan",),
        "thermostat": ("thermostat", "temperature", "heater"),
    }

    FAN_SPEEDS = ("low", "medium", "high")
    DEFAULT_MODEL_NAME = "facebook/bart-large-mnli"

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        self._classifier = None

        if pipeline is not None:
            self._classifier = pipeline(
                "zero-shot-classification",
                model=self.model_name,
            )

    @property
    def is_available(self) -> bool:
        return self._classifier is not None

    def parse(self, user_input: str) -> StructuredCommand:
        if not self.is_available:
            raise RuntimeError(
                "Transformers is not installed. Install it with "
                "'pip install transformers torch' to enable AI command parsing."
            )

        classification = self._classifier(
            user_input,
            candidate_labels=self.INTENT_LABELS,
            hypothesis_template="This smart home command is about {}.",
        )

        intent = classification["labels"][0]
        confidence = float(classification["scores"][0])
        device = self._extract_device(user_input, intent)
        action, value = self._extract_action_and_value(user_input, intent)

        return StructuredCommand(
            raw_input=user_input,
            intent=intent,
            device=device,
            action=action,
            value=value,
            confidence=confidence,
        )

    def _extract_device(self, user_input: str, intent: str) -> str | None:
        lowered = user_input.lower()

        for device, keywords in self.DEVICE_KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                return device

        if intent == "get_temperature" or intent == "set_temperature":
            return "thermostat"

        return None

    def _extract_action_and_value(
        self, user_input: str, intent: str
    ) -> tuple[str | None, str | int | None]:
        lowered = user_input.lower()

        if intent == "turn_on_device":
            return "turn_on", None

        if intent == "turn_off_device":
            return "turn_off", None

        if intent == "set_fan_speed":
            for speed in self.FAN_SPEEDS:
                if speed in lowered:
                    return "set_speed", speed
            return "set_speed", None

        if intent == "set_temperature":
            match = re.search(r"(\d{2})", lowered)
            return "set_temperature", int(match.group(1)) if match else None

        if intent in {"get_device_status", "get_temperature"}:
            return "get_status", None

        return None, None
