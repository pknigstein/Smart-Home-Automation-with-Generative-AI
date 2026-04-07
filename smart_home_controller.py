from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from command_parser import HuggingFaceCommandParser, StructuredCommand
from smart_devices import Fan, Light, Thermostat


@dataclass
class CommandResult:
    parsed_command: StructuredCommand
    feedback: str
    device_status: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "parsed_command": self.parsed_command.to_dict(),
            "feedback": self.feedback,
            "device_status": self.device_status,
        }


class SmartHomeController:
    def __init__(
        self,
        light: Light,
        fan: Fan,
        thermostat: Thermostat,
        parser: HuggingFaceCommandParser,
    ) -> None:
        self.devices = {
            "light": light,
            "fan": fan,
            "thermostat": thermostat,
        }
        self.parser = parser

    def process_command(self, user_input: str) -> CommandResult:
        parsed_command = self.parser.parse(user_input)
        feedback, device_status = self._execute(parsed_command)
        return CommandResult(
            parsed_command=parsed_command,
            feedback=feedback,
            device_status=device_status,
        )

    def _execute(
        self, command: StructuredCommand
    ) -> tuple[str, dict[str, Any] | None]:
        device_name = command.device
        device = self.devices.get(device_name) if device_name else None

        if device is None and command.intent != "get_device_status":
            return "I could not determine which device to control.", None

        try:
            if command.action == "turn_on":
                device.turn_on()
                status = device.get_status()
                return self._build_power_feedback(device_name, True), status

            if command.action == "turn_off":
                device.turn_off()
                status = device.get_status()
                return self._build_power_feedback(device_name, False), status

            if command.action == "set_speed":
                if not isinstance(device, Fan):
                    return "Fan speed can only be set for the fan.", None
                if command.value is None:
                    return "Please specify a fan speed: low, medium, or high.", None
                device.set_speed(str(command.value))
                if not device.is_on:
                    device.turn_on()
                status = device.get_status()
                return self._build_fan_speed_feedback(device.speed), status

            if command.action == "set_temperature":
                if not isinstance(device, Thermostat):
                    return "Temperature can only be set on the thermostat.", None
                if not isinstance(command.value, int):
                    return "Please provide a temperature between 18°C and 30°C.", None
                device.set_temperature(command.value)
                status = device.get_status()
                return self._build_temperature_feedback(device.temperature), status

            if command.action == "get_status":
                if device is not None:
                    status = device.get_status()
                    return self._build_status_feedback(device_name, status), status

                all_statuses = {
                    name: current_device.get_status()
                    for name, current_device in self.devices.items()
                }
                return self._build_overall_status_feedback(all_statuses), all_statuses

            return "I understood the command, but I could not execute it.", None
        except ValueError as error:
            return str(error), device.get_status() if device is not None else None

    def _build_power_feedback(self, device_name: str | None, is_on: bool) -> str:
        label = "device" if device_name is None else f"{device_name}"
        power_state = "ON" if is_on else "OFF"
        return f"The {label} is now {power_state}."

    def _build_fan_speed_feedback(self, speed: str) -> str:
        return f"The fan speed is set to {speed}."

    def _build_temperature_feedback(self, temperature: int) -> str:
        return f"The thermostat is set to {temperature}°C."

    def _build_status_feedback(
        self, device_name: str | None, status: dict[str, Any]
    ) -> str:
        if device_name == "light":
            return f"The light is currently {status['power'].upper()}."

        if device_name == "fan":
            return (
                f"The fan is currently {status['power'].upper()} "
                f"at {status['speed']} speed."
            )

        if device_name == "thermostat":
            return f"The current temperature is {status['temperature']}°C."

        return f"Current status: {status}"

    def _build_overall_status_feedback(self, statuses: dict[str, dict[str, Any]]) -> str:
        light_status = statuses["light"]["power"].upper()
        fan_status = statuses["fan"]["power"].upper()
        fan_speed = statuses["fan"]["speed"]
        temperature = statuses["thermostat"]["temperature"]
        return (
            f"Light: {light_status}. "
            f"Fan: {fan_status} at {fan_speed} speed. "
            f"Thermostat: {temperature}°C."
        )


def build_demo_controller() -> SmartHomeController:
    parser = HuggingFaceCommandParser()
    return SmartHomeController(
        light=Light(name="Living Room"),
        fan=Fan(name="Bedroom"),
        thermostat=Thermostat(name="Hallway"),
        parser=parser,
    )


def demo() -> None:
    controller = build_demo_controller()
    example_commands = [
        "Turn on the light",
        "Set the fan speed to high",
        "What is the current temperature?",
    ]

    for command in example_commands:
        result = controller.process_command(command)
        print(f"User input: {command}")
        print(f"Structured command: {result.parsed_command.to_dict()}")
        print(f"Feedback: {result.feedback}")
        print(f"Status: {result.device_status}")
        print("-" * 60)


if __name__ == "__main__":
    demo()
