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
        "increase_temperature",
        "decrease_temperature",
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
    MIN_CONFIDENCE = 0.5

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
        rule_based_command = self._rule_based_parse(user_input)
        if rule_based_command is not None:
            return rule_based_command

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

        if not self._is_confident_command(intent, device, action, confidence):
            return self._build_unknown_command(user_input)

        return StructuredCommand(
            raw_input=user_input,
            intent=intent,
            device=device,
            action=action,
            value=value,
            confidence=confidence,
        )

    def _rule_based_parse(self, user_input: str) -> StructuredCommand | None:
        lowered = user_input.lower().strip()
        speed = self._extract_speed(lowered)
        temperature = self._extract_temperature(lowered)

        if self._matches_any(
            lowered,
            (
                r"\b(lower|decrease|reduce)\s+(the\s+)?temperature\b",
                r"\bmake\s+(it\s+)?colder\b",
            ),
        ):
            return StructuredCommand(
                raw_input=user_input,
                intent="decrease_temperature",
                device="thermostat",
                action="decrease_temperature",
                value=1,
                confidence=1.0,
            )

        if self._matches_any(
            lowered,
            (
                r"\b(increase|raise)\s+(the\s+)?temperature\b",
                r"\bmake\s+(it\s+)?warmer\b",
            ),
        ):
            return StructuredCommand(
                raw_input=user_input,
                intent="increase_temperature",
                device="thermostat",
                action="increase_temperature",
                value=1,
                confidence=1.0,
            )

        if self._matches_any(
            lowered,
            (
                r"\b(turn|switch)\s+on\s+(the\s+)?fan\b",
                r"\b(turn|switch)\s+(the\s+)?fan\s+on\b",
            ),
        ):
            return StructuredCommand(
                raw_input=user_input,
                intent="turn_on_device",
                device="fan",
                action="turn_on",
                value=speed,
                confidence=1.0,
            )

        if self._matches_any(
            lowered,
            (
                r"\b(turn|switch)\s+off\s+(the\s+)?fan\b",
                r"\b(turn|switch)\s+(the\s+)?fan\s+off\b",
            ),
        ):
            return StructuredCommand(
                raw_input=user_input,
                intent="turn_off_device",
                device="fan",
                action="turn_off",
                value=None,
                confidence=1.0,
            )

        if self._matches_any(
            lowered,
            (
                r"\b(turn|switch)\s+on\s+(the\s+)?light\b",
                r"\b(turn|switch)\s+(the\s+)?light\s+on\b",
                r"\b(turn|switch)\s+on\s+(the\s+)?lamp\b",
                r"\b(turn|switch)\s+(the\s+)?lamp\s+on\b",
            ),
        ):
            return StructuredCommand(
                raw_input=user_input,
                intent="turn_on_device",
                device="light",
                action="turn_on",
                value=None,
                confidence=1.0,
            )

        if self._matches_any(
            lowered,
            (
                r"\b(turn|switch)\s+off\s+(the\s+)?light\b",
                r"\b(turn|switch)\s+(the\s+)?light\s+off\b",
                r"\b(turn|switch)\s+off\s+(the\s+)?lamp\b",
                r"\b(turn|switch)\s+(the\s+)?lamp\s+off\b",
            ),
        ):
            return StructuredCommand(
                raw_input=user_input,
                intent="turn_off_device",
                device="light",
                action="turn_off",
                value=None,
                confidence=1.0,
            )

        if self._matches_any(
            lowered,
            (
                r"\bset\s+(the\s+)?fan\s+speed\s+to\s+(low|medium|high)\b",
                r"\bset\s+(the\s+)?speed\s+of\s+(the\s+)?fan\s+to\s+(low|medium|high)\b",
            ),
        ):
            return StructuredCommand(
                raw_input=user_input,
                intent="set_fan_speed",
                device="fan",
                action="set_speed",
                value=speed,
                confidence=1.0,
            )

        if (
            temperature is not None
            and self._matches_any(
                lowered,
                (
                    r"\bset\s+(the\s+)?thermostat\s+to\b",
                    r"\bset\s+(the\s+)?temperature\s+to\b",
                ),
            )
        ):
            return StructuredCommand(
                raw_input=user_input,
                intent="set_temperature",
                device="thermostat",
                action="set_temperature",
                value=temperature,
                confidence=1.0,
            )

        if self._matches_any(
            lowered,
            (
                r"\bwhat\s+is\s+(the\s+)?current\s+temperature\b",
                r"\bwhat\s+is\s+(the\s+)?temperature\b",
                r"\bcurrent\s+temperature\b",
            ),
        ):
            return StructuredCommand(
                raw_input=user_input,
                intent="get_temperature",
                device="thermostat",
                action="get_status",
                value=None,
                confidence=1.0,
            )

        if "status" in lowered:
            device = self._extract_device(lowered, "get_device_status")
            return StructuredCommand(
                raw_input=user_input,
                intent="get_device_status",
                device=device,
                action="get_status",
                value=None,
                confidence=1.0,
            )

        return None

    def _matches_any(self, user_input: str, patterns: tuple[str, ...]) -> bool:
        return any(re.search(pattern, user_input) for pattern in patterns)

    def _extract_speed(self, user_input: str) -> str | None:
        for speed in self.FAN_SPEEDS:
            if re.search(rf"\b{speed}\b", user_input):
                return speed
        return None

    def _extract_temperature(self, user_input: str) -> int | None:
        match = re.search(r"\b(\d{2})\b", user_input)
        return int(match.group(1)) if match else None

    def _is_confident_command(
        self,
        intent: str,
        device: str | None,
        action: str | None,
        confidence: float,
    ) -> bool:
        if confidence < self.MIN_CONFIDENCE:
            return False

        if action is None:
            return False

        if intent != "get_device_status" and device is None:
            return False

        return True

    def _build_unknown_command(self, user_input: str) -> StructuredCommand:
        return StructuredCommand(
            raw_input=user_input,
            intent="unknown",
            device=None,
            action=None,
            value=None,
            confidence=0.0,
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

        if intent == "increase_temperature":
            return "increase_temperature", 1

        if intent == "decrease_temperature":
            return "decrease_temperature", 1

        if intent in {"get_device_status", "get_temperature"}:
            return "get_status", None

        return None, None
