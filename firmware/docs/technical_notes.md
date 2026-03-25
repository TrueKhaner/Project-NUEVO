# Firmware Technical Notes

This document captures the low-level constraints and design choices behind the firmware.

Use this together with:

- [`architecture.md`](architecture.md) for the top-level runtime map
- [`communication_and_state.md`](communication_and_state.md) for TLV/state behavior
- [`motion_control.md`](motion_control.md) for motor/stepper/servo implementation
- [`sensors_and_i2c.md`](sensors_and_i2c.md) for the shared I2C bus and sensor stack

## MCU Constraints

Target MCU: **ATmega2560** on the Arduino Mega 2560.

Important limits that shape the firmware:

- fixed hardware interrupt priorities
- no DMA / PIO style peripheral for WS2812 signaling
- limited SRAM, so debug/reporting must stay bounded
- UART reliability depends on keeping interrupt-blocking windows short

## Timer Ownership

| Timer | Current role | Notes |
|------|--------------|-------|
| `Timer0` | Arduino core timebase | leave untouched |
| `Timer1` | `800 Hz` DC round-robin overflow ISR | short latch/apply slice only |
| `Timer2` | available / driver-local PWM usage | not central to system timing |
| `Timer3` | `10 kHz` stepper overflow ISR + OC3A hardware PWM use where mapped | also samples UART2 fault edges in the ISR |
| `Timer4` | PWM carrier only | no overflow ISR in the runtime design |
| `Timer5` | discrete LED PWM pins | normal hardware PWM use |

`analogWrite()` is avoided on Timer1/3/4-owned outputs because it can reconfigure timer modes and break the intended timing setup.

## UART Timing Context

The Raspberry Pi link runs at the baud rate configured in [`arduino/src/config.h`](../arduino/src/config.h).

The runtime relies on two ideas:

1. keep ISR bodies short enough that RX overrun is rare
2. keep RX/TX draining in the fast loop path instead of moving UART work into control ISRs

The tuned link profile is:

- `RPI_BAUD_RATE = 200000`
- `taskUART() = 100 Hz`
- telemetry spread across a 10-slot scheduler wheel instead of bunching all due streams into the same UART pass

`LoopMonitor` budgets are set conservatively for ISR slices:

- `PID_ISR_UART_BUDGET_US`
- `STEPPER_ISR_UART_BUDGET_US`
- `PID_ROUND_BUDGET_US`

These are budget checks, not proof of link integrity. Real UART health still comes from:

- `dor`
- `fe`
- `crc`
- `frame`
- `tlv`

reported by `MessageCenter` / `StatusReporter`.

The lightweight USB fault-event logger now reports:

- loop-overrun deltas by slot name
- control handoff deltas (`miss`, `late`, `reuse`, `cross`)
- UART deltas including `oversize` explicitly

## Mixed DC Control Timing

### Slot timing

Timer1 runs at `800 Hz`, so:

- one slot = `1.25 ms`
- four slots = one full DC round = `5 ms`

With four motors, that gives `200 Hz` per motor.

### Pipeline model

The current DC path uses a per-slot mixed round-robin pipeline:

- Timer1 services one motor slot every `1.25 ms`
- the loop computes the matching motor for that slot
- the next time that same motor returns `5 ms` later, its staged output is published

This keeps each motor at `200 Hz` while avoiding a large four-motor compute burst in the loop.

### What the timing fields mean

In the human-readable status output:

- `pid`: one Timer1 ISR slice
- `pidr`: full Timer1 four-slot round span
- `motor`: loop-side DC compute round
- `uart`: soft UART task wall-clock duration
- `debug`: status formatting/chunking time
- `flush`: USB debug drain time

`ControlWin` counters are the best way to judge control handoff quality:

- `missed`
- `late`
- `reused`
- `cross`

Low single-digit counts are acceptable; sustained growth indicates handoff pressure.

## NeoPixel / WS2812 Policy

The current firmware supports a **single WS2812 state-indicator pixel**.

Important constraint on AVR:

- WS2812 signaling needs a tightly timed uninterrupted burst
- on AVR, that typically means masking interrupts during `show()`
- the transmit time scales with pixel count

Rule of thumb:

- about `30 us` per RGB pixel
- plus the frame latch/reset time

That makes multi-pixel WS2812 animation a bad fit for this firmware. The supported WS2812 runtime is:

- one pixel
- infrequent updates
- state-change indication only

If richer LED animation is needed later, prefer clocked LEDs such as APA102/SK9822 or move LED driving off the Mega.

## Discrete LED Policy

Discrete LEDs are reserved for user/TLV control.

The firmware no longer uses the discrete red LED as an automatic low-battery or fault indicator. Automatic state indication belongs to the NeoPixel only.

## Input Sampling Policy

Shared button / limit-switch GPIOs are sampled in `taskSensors()` at `100 Hz` through `UserIO::sampleInputs()`.

Why this is intentional:

- input sampling is state acquisition, not safety policy
- stepper homing already reads its relevant limit inputs directly in the stepper path
- moving all buttons/limits to interrupts would increase complexity and ISR load with little value

## State Machine Policy Notes

`SystemManager` owns all transition policy.

Important current battery behavior:

- `RUNNING` can start without a battery
- enable commands remain ineffective while battery is absent
- once battery power has been seen during that `RUNNING` session, losing it trips `ERROR`
- `RESET` can still clear `ERROR` back to `IDLE`; actuators remain disabled until policy allows them again

## Debug/Status Strategy

The human-readable status reporter is kept because it is useful for field stability checks, but it is intentionally decoupled from system behavior:

- `StatusReporter` only reports
- `SystemManager` decides transitions
- `SafetyManager` only detects faults
- `MessageCenter` owns communication
- `DebugLog::flush()` is intentionally time-sliced so USB debug output cannot
  monopolize the loop during active control

Status reporting can be disabled from `config.h`:

- `STATUS_REPORTER_ENABLED`
- `STATUS_REPORT_HZ`

Independent of the full reporter, the firmware can also keep a lightweight
fault-event logger enabled:

- `FAULT_EVENT_LOG_ENABLED`
- `FAULT_EVENT_MIN_INTERVAL_MS`

That path only emits short one-line USB debug messages when new loop, control,
or UART faults appear.

## Memory Headroom

The firmware still runs with relatively tight SRAM headroom on the Mega. The exact numbers vary by build flags, but debug/status features are the first thing to disable if you need more margin while preserving behavior.

Recommended approach:

1. keep the reporter on during bring-up and soak testing
2. reduce report frequency or disable it for production-like runs if needed
