# Smart Home Automation with Generative AI

A simple Python smart home application that simulates household devices and controls them through natural language commands interpreted by a Hugging Face Transformers model.

## Overview

This project demonstrates how Generative AI can be used as a natural language interface for a smart home system. Users can type commands such as:

- `Turn on the light`
- `Set the fan speed to high`
- `Set the thermostat to 24 degrees`
- `What is the current temperature?`

The application parses each request into a structured command and then maps that command to the correct device action.

## Features

- Simulated smart home devices in Python
- Natural language command interpretation using Hugging Face Transformers
- Structured command extraction with intent, device, action, value, and confidence
- Action execution for light, fan, and thermostat devices
- Clear user-facing feedback messages after each command
- Simple demo flow showing end-to-end command handling

## Project Structure

- `smart_devices.py`
  Defines the simulated device classes:
  - `Light`
  - `Fan`
  - `Thermostat`

- `command_parser.py`
  Uses a Hugging Face zero-shot classification pipeline to interpret natural language and convert it into a `StructuredCommand`.

- `smart_home_controller.py`
  Maps parsed commands to device actions and generates user feedback such as:
  - `The light is now ON.`
  - `The fan speed is set to high.`
  - `The thermostat is set to 24°C.`

- `requirements.txt`
  Lists the Python dependencies needed to run the project.

## How It Works

The application follows this flow:

1. The user enters a natural language command.
2. The Hugging Face model classifies the command intent.
3. The parser extracts structured fields such as:
   - `intent`
   - `device`
   - `action`
   - `value`
   - `confidence`
4. The controller maps the parsed command to the correct device method.
5. The system returns a confirmation or status message to the user.

## Supported Devices

### Light

- ON/OFF control
- Queryable status

Example responses:

- `The light is now ON.`
- `The light is currently OFF.`

### Fan

- ON/OFF control
- Speed control: `low`, `medium`, `high`
- Queryable status

Example responses:

- `The fan speed is set to medium.`
- `The fan is currently ON at high speed.`

### Thermostat

- Adjustable temperature range from `18°C` to `30°C`
- Queryable current temperature

Example responses:

- `The thermostat is set to 24°C.`
- `The current temperature is 22°C.`

## AI Command Parsing

The project uses Hugging Face Transformers with the zero-shot classification pipeline and the model:

- `facebook/bart-large-mnli`

This model helps classify user intent into categories such as:

- `turn_on_device`
- `turn_off_device`
- `set_fan_speed`
- `set_temperature`
- `get_device_status`
- `get_temperature`

After classification, the parser uses simple extraction rules to identify:

- Which device is being targeted
- Which action should be executed
- Any value required, such as fan speed or thermostat temperature

## Installation

### 1. Clone or open the project folder

```bash
cd "/Users/philippkonigstein/Documents/Smart Home Automation with Generative AI"
```

### 2. Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

## Running the Demo

Run the demo script:

```bash
python3 smart_home_controller.py
```

This will execute a few example commands and print:

- The original user input
- The structured command produced by the parser
- The feedback returned to the user
- The updated device status

## Example Output

```text
User input: Turn on the light
Structured command: {'raw_input': 'Turn on the light', 'intent': 'turn_on_device', 'device': 'light', 'action': 'turn_on', 'value': None, 'confidence': 0.8254660964012146}
Feedback: The light is now ON.
Status: {'device': 'Living Room', 'type': 'light', 'power': 'on'}
```

```text
User input: Set the fan speed to high
Structured command: {'raw_input': 'Set the fan speed to high', 'intent': 'set_fan_speed', 'device': 'fan', 'action': 'set_speed', 'value': 'high', 'confidence': 0.9139582514762878}
Feedback: The fan speed is set to high.
Status: {'device': 'Bedroom', 'type': 'fan', 'power': 'on', 'speed': 'high'}
```

```text
User input: What is the current temperature?
Structured command: {'raw_input': 'What is the current temperature?', 'intent': 'get_temperature', 'device': 'thermostat', 'action': 'get_status', 'value': None, 'confidence': 0.5420470833778381}
Feedback: The current temperature is 22°C.
Status: {'device': 'Hallway', 'type': 'thermostat', 'temperature': 22, 'range': '18°C-30°C'}
```

```text
User input: Set the thermostat to 24 degrees
Structured command: {'raw_input': 'Set the thermostat to 24 degrees', 'intent': 'set_temperature', 'device': 'thermostat', 'action': 'set_temperature', 'value': 24, 'confidence': 0.8900000000000000}
Feedback: The thermostat is set to 24°C.
Status: {'device': 'Hallway', 'type': 'thermostat', 'temperature': 24, 'range': '18°C-30°C'}
```

```text
User input: Set the fan speed to low
Structured command: {'raw_input': 'Set the fan speed to low', 'intent': 'set_fan_speed', 'device': 'fan', 'action': 'set_speed', 'value': 'low', 'confidence': 0.8800000000000000}
Feedback: The fan must be ON before its speed can be changed.
Status: {'device': 'Bedroom', 'type': 'fan', 'power': 'off', 'speed': None}
```

## Running the Interactive Demo

```
python3 smart_home_cli.py --show-structured
```

## Requirements

- Python 3.10 or later recommended
- Internet access for the first model download from Hugging Face

Dependencies:

- `transformers`
- `torch`

## Notes

- The device simulation is local and in-memory only.
- The parser combines AI-based intent classification with rule-based extraction for device names and values.
- The first run may take longer because the model weights must be downloaded and loaded.

## Possible Next Steps

- Add support for multiple rooms and multiple devices of the same type
- Build a CLI loop for interactive conversations
- Add a web interface or REST API
- Store device states persistently
- Replace or extend rule-based extraction with a more advanced structured generation approach

## License

This project currently does not include a license file.
