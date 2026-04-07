from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Light:
    name: str
    is_on: bool = False

    def turn_on(self) -> str:
        self.is_on = True
        return f"{self.name} light turned on."

    def turn_off(self) -> str:
        self.is_on = False
        return f"{self.name} light turned off."

    def get_status(self) -> dict[str, str]:
        return {
            "device": self.name,
            "type": "light",
            "power": "on" if self.is_on else "off",
        }


@dataclass
class Fan:
    VALID_SPEEDS: tuple[str, ...] = field(
        default=("low", "medium", "high"), init=False, repr=False
    )

    name: str
    is_on: bool = False
    speed: str = "low"

    def turn_on(self, speed: str | None = None) -> str:
        self.is_on = True
        if speed is not None:
            self.set_speed(speed)
        return f"{self.name} fan turned on at {self.speed} speed."

    def turn_off(self) -> str:
        self.is_on = False
        return f"{self.name} fan turned off."

    def set_speed(self, speed: str) -> str:
        normalized_speed = speed.lower()
        if normalized_speed not in self.VALID_SPEEDS:
            valid = ", ".join(self.VALID_SPEEDS)
            raise ValueError(f"Invalid fan speed '{speed}'. Choose from: {valid}.")

        self.speed = normalized_speed
        return f"{self.name} fan speed set to {self.speed}."

    def get_status(self) -> dict[str, str]:
        return {
            "device": self.name,
            "type": "fan",
            "power": "on" if self.is_on else "off",
            "speed": self.speed,
        }


@dataclass
class Thermostat:
    MIN_TEMPERATURE: int = field(default=18, init=False, repr=False)
    MAX_TEMPERATURE: int = field(default=30, init=False, repr=False)

    name: str
    temperature: int = 22

    def set_temperature(self, temperature: int) -> str:
        if not self.MIN_TEMPERATURE <= temperature <= self.MAX_TEMPERATURE:
            raise ValueError(
                f"Temperature must be between {self.MIN_TEMPERATURE}°C "
                f"and {self.MAX_TEMPERATURE}°C."
            )

        self.temperature = temperature
        return f"{self.name} thermostat set to {self.temperature}°C."

    def increase_temperature(self, amount: int = 1) -> str:
        return self.set_temperature(self.temperature + amount)

    def decrease_temperature(self, amount: int = 1) -> str:
        return self.set_temperature(self.temperature - amount)

    def get_status(self) -> dict[str, str | int]:
        return {
            "device": self.name,
            "type": "thermostat",
            "temperature": self.temperature,
            "range": f"{self.MIN_TEMPERATURE}°C-{self.MAX_TEMPERATURE}°C",
        }
