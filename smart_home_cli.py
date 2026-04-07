from __future__ import annotations

import argparse
from typing import Any

from smart_home_controller import build_demo_controller


EXAMPLE_COMMANDS = [
    "Turn on the light",
    "Set the fan speed to medium",
    "Set the thermostat to 24 degrees",
    "What is the current temperature?",
    "What is the status of the fan?",
]


def format_status(status: dict[str, Any] | None) -> str:
    if status is None:
        return "No device status available."

    if "type" in status:
        device_type = status["type"]

        if device_type == "light":
            return f"Light status: {status['power'].upper()}"

        if device_type == "fan":
            return (
                f"Fan status: {status['power'].upper()} "
                f"at {status['speed']} speed"
            )

        if device_type == "thermostat":
            return f"Thermostat status: {status['temperature']}°C"

    parts = []
    for device_name, device_status in status.items():
        if device_name == "light":
            parts.append(f"Light {device_status['power'].upper()}")
        elif device_name == "fan":
            parts.append(
                f"Fan {device_status['power'].upper()} at {device_status['speed']}"
            )
        elif device_name == "thermostat":
            parts.append(f"Thermostat {device_status['temperature']}°C")

    return " | ".join(parts) if parts else str(status)


def print_welcome(show_structured: bool) -> None:
    print("Smart Home AI Demo")
    print("=" * 50)
    print("Type a natural language command to control the smart home.")
    print("Commands: 'help' for examples, 'status' for all device states, 'exit' to quit.")
    if show_structured:
        print("Structured command output is ENABLED for this session.")
    print()
    print("Example prompts:")
    for example in EXAMPLE_COMMANDS:
        print(f"- {example}")
    print()
    print("Loading the Hugging Face model may take a moment on the first command.")
    print("=" * 50)


def run_cli(show_structured: bool) -> None:
    controller = build_demo_controller()
    print_welcome(show_structured)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            break

        if not user_input:
            continue

        normalized = user_input.lower()

        if normalized in {"exit", "quit"}:
            print("Session ended.")
            break

        if normalized == "help":
            print("Try one of these example commands:")
            for example in EXAMPLE_COMMANDS:
                print(f"- {example}")
            continue

        if normalized == "status":
            all_statuses = {
                name: device.get_status() for name, device in controller.devices.items()
            }
            print(f"Assistant: {format_status(all_statuses)}")
            continue

        try:
            result = controller.process_command(user_input)
        except RuntimeError as error:
            print(f"Assistant: {error}")
            break

        print(f"Assistant: {result.feedback}")
        print(f"Status: {format_status(result.device_status)}")
        if show_structured:
            print(f"Structured command: {result.parsed_command.to_dict()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive CLI demo for the smart home application."
    )
    parser.add_argument(
        "--show-structured",
        action="store_true",
        help="Display the parsed structured command after each user input.",
    )
    args = parser.parse_args()
    run_cli(show_structured=args.show_structured)


if __name__ == "__main__":
    main()
