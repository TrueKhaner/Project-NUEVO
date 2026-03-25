# Arduino Firmware

This folder contains the Arduino Mega 2560 firmware for Project NUEVO.

The firmware is the low-level controller for:
- DC motors, steppers, and servos
- odometry and encoder feedback
- IMU and voltage monitoring
- TLV communication with the Raspberry Pi
- system state, safety policy, and user I/O

The current release target is firmware `v0.9.7`.

## What You Need

- Arduino IDE 2.x
- Arduino AVR Boards core installed
- Arduino Mega 2560 target selected

## Important Serial Buffer Setting

The stock AVR serial buffers are too small for the current `Serial2` link. Add a
`platform.local.txt` file for the installed Arduino AVR core and put this line in it:

```text
compiler.cpp.extra_flags=-DSERIAL_RX_BUFFER_SIZE=512 -DSERIAL_TX_BUFFER_SIZE=256
```

Typical locations:

Windows:

```text
C:\Users\<your-user>\AppData\Local\Arduino15\packages\arduino\hardware\avr\1.8.7\platform.local.txt
```

macOS:

```text
~/Library/Arduino15/packages/arduino/hardware/avr/1.8.7/platform.local.txt
```

Notes:
- these macros apply to all linked `HardwareSerial` instances in the stock AVR core
- if the Arduino AVR Boards version changes, check that `platform.local.txt` still exists under the new versioned directory

## Build and Upload

Using Arduino IDE:

1. Open [`arduino/arduino.ino`](arduino/arduino.ino) in Arduino IDE.
2. Select board: `Arduino Mega or Mega 2560`.
3. Select the correct port.
4. Click `Verify` to build.
5. Click `Upload`.

## Optional CLI Workflow

```bash
arduino-cli compile --fqbn arduino:avr:mega firmware/arduino
arduino-cli upload --fqbn arduino:avr:mega -p /dev/ttyUSB0 firmware/arduino
```

Replace `/dev/ttyUSB0` with the correct port for your machine.

## Main Configuration File

Most project-level firmware tuning starts in:

- [`arduino/src/config.h`](arduino/src/config.h)

That file contains:
- robot geometry
- telemetry rates
- motor, encoder, and sensor enable flags
- battery configuration
- debug and timing settings

## Documentation

Keep this README short and use the docs below for technical details:

- [`docs/README.md`](docs/README.md): firmware documentation map
- [`docs/architecture.md`](docs/architecture.md): runtime structure and startup order
- [`docs/communication_and_state.md`](docs/communication_and_state.md): TLV flow, state machine, safety, deferred work
- [`docs/motion_control.md`](docs/motion_control.md): DC control, steppers, servos, homing, odometry
- [`docs/sensors_and_i2c.md`](docs/sensors_and_i2c.md): IMU, magnetometer calibration, voltage, shared I2C behavior
- [`docs/technical_notes.md`](docs/technical_notes.md): timing, UART, memory, and low-level constraints

For protocol-level documentation, use:

- [`../docs/COMMUNICATION_PROTOCOL.md`](../docs/COMMUNICATION_PROTOCOL.md)
- [`../tlv_protocol/TLV_Payloads.md`](../tlv_protocol/TLV_Payloads.md)

## Folder Structure

```text
firmware/
├── arduino/                  # Main Arduino sketch and production firmware source
│   ├── arduino.ino           # Firmware entry point: setup(), loop(), ISR vectors
│   └── src/
│       ├── config.h          # Compile-time robot, timing, telemetry, and feature settings
│       ├── pins.h            # Board pin assignments and timer/PWM aliases
│       ├── drivers/          # Per-device drivers: DC motors, IMU, servos, NeoPixel
│       ├── messages/         # TLV type IDs and payload definitions used on the wire
│       ├── modules/          # Higher-level subsystems: MessageCenter, SensorManager, kinematics, etc.
│       └── lib/              # Vendored third-party Arduino libraries
├── docs/                     # Firmware-specific technical documentation
└── tests/                    # Standalone bring-up sketches and hardware checks
```
