# Firmware Documentation Map

This folder contains focused implementation notes for the Arduino firmware.

Use [`../README.md`](../README.md) first for the high-level firmware overview.
Then use the files here for subsystem-level details.

## Current Documents

| Document | Purpose |
|---|---|
| [`architecture.md`](architecture.md) | Top-level runtime structure, startup sequence, execution layers, and module ownership |
| [`communication_and_state.md`](communication_and_state.md) | TLV RX/TX flow, UART task behavior, state machine, safety policy, deferred work |
| [`motion_control.md`](motion_control.md) | DC mixed control, motor feedback, steppers, servos, and odometry |
| [`sensors_and_i2c.md`](sensors_and_i2c.md) | IMU, voltage monitoring, magnetometer calibration, and shared `Wire` bus behavior |
| [`technical_notes.md`](technical_notes.md) | Timer ownership, UART timing, memory limits, and other low-level constraints |
| [`pin_table_rev_A.md`](pin_table_rev_A.md) | Rev. A pin map |
| [`pin_table_rev_B.md`](pin_table_rev_B.md) | Rev. B pin map |

## Document Ownership Rules

- Keep firmware implementation details here, not in the cross-project `docs/`
  folder.
- Keep protocol semantics in [`../../docs/COMMUNICATION_PROTOCOL.md`](../../docs/COMMUNICATION_PROTOCOL.md).
- Keep byte-level payload layouts in [`../../tlv_protocol/TLV_Payloads.md`](../../tlv_protocol/TLV_Payloads.md).
- Keep compile-time truth in code:
  - [`../arduino/src/config.h`](../arduino/src/config.h)
  - [`../arduino/src/pins.h`](../arduino/src/pins.h)
  - [`../arduino/src/messages/TLV_Payloads.h`](../arduino/src/messages/TLV_Payloads.h)

If a firmware behavior changes, prefer updating the focused subsystem document
here instead of growing `firmware/README.md` into a second full specification.
