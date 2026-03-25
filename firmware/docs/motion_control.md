# Motion Control Implementation

This document describes how DC motors, steppers, encoders, odometry, and servos are implemented.

## 1. DC Motor Control: Mixed ISR + Loop Round-Robin

The DC motor path is intentionally split between Timer1 ISR work and loop-side compute.

### Timer1 rate and slot model

Timer1 runs at `800 Hz`.

- one Timer1 overflow = one slot = `1.25 ms`
- four slots = one full DC round = `5 ms`
- each motor gets one slice per round = `200 Hz` effective apply cadence per motor

### What happens in `ISR(TIMER1_OVF_vect)`

The ISR in `arduino.ino` does three things:

1. reads Timer1 counter ticks for cheap timing measurement
2. calls `MotorControlCoordinator::servicePidIsrSlice(...)`
3. records the ISR slice time in `LoopMonitor`

`servicePidIsrSlice(...)` owns the real round-robin logic:

- slot `0` starts a new round
- the current motor's previously prepared output is published if ready
- the current motor's already-published output is applied
- the current motor's encoder feedback is latched
- that same motor is queued for one loop-side compute pass
- the slot index is advanced for the next Timer1 tick
- when the last slot closes a running round, the full round span is returned for `pidr`

### What happens in loop context

The fast lane calls `fastMotorCompute()`, which calls `taskMotors()` whenever
`MotorControlCoordinator` reports a pending motor slot.

`taskMotors()` now computes exactly one motor per invocation and stages the next
output for that same motor only. If a small backlog exists, it may compute a
few pending motor slots in one pass, but only within a bounded wall-clock
budget.

So the pipeline is:

- slot `N` applies motor `i` using the output previously published for motor `i`
- slot `N` latches fresh feedback for motor `i`
- loop computes motor `i`
- the next time motor `i` returns `5 ms` later, its staged output is published

This keeps each motor at `200 Hz` while avoiding large bursty loop-side compute.

## 2. `MotorControlCoordinator`

`MotorControlCoordinator` exists to keep the mixed control handoff understandable and centralized.

It owns:

- current ISR slot
- round count
- requested/computed/applied slot-request counters
- compute/apply sequence counters
- per-slot pending request state
- per-window handoff counters used by `StatusReporter`

The status line:

`req/cmp/app=.../.../... | prep/app=.../... | pipe=...`

comes directly from this module.

The `ControlWin` counters mean:

- `missed`: a motor slot came around again before its previous request had even started or finished compute
- `late`: compute started after that motor's `5 ms` deadline window had already been crossed
- `reused`: a motor slot had to reuse its previous published output because the new one was not ready
- `cross`: compute finished after a newer request for the same motor had already arrived

## 3. `DCMotor`

Each `DCMotor` object owns one physical motor channel.

### Major responsibilities

- keep track of its pins
- read encoder position via an `IEncoderCounter`
- maintain a simple fixed-rate encoder velocity estimate
- run cascade position/velocity control
- stage and publish H-bridge outputs
- detect basic encoder-stuck / feedback failure conditions

### Control modes

Supported modes:

- `DC_MODE_DISABLED`
- `DC_MODE_POSITION`
- `DC_MODE_VELOCITY`
- `DC_MODE_PWM`
- `DC_MODE_HOMING`

Additional DC actuator actions:

- `resetPosition()`
  - resets the encoder baseline to a specific count (the UI currently uses zero)
  - preserves the active mode unless a later command changes it
- `home(direction, homeVelocity)`
  - drives the motor toward its configured home limit
  - when the limit triggers, the encoder count is reset to zero and the motor is disabled

Optional DC home limits are configured in `config.h` with:

- `PIN_M1_LIMIT`
- `PIN_M2_LIMIT`
- `PIN_M3_LIMIT`
- `PIN_M4_LIMIT`

They use the same `LIMIT_ACTIVE_LOW` policy as stepper home switches.

### Feedback and control update model

`DCMotor` is split across three loop/ISR paths:

- `refreshFeedback()`
  - loop-side shared feedback refresh at `200 Hz`
  - updates cached position / velocity for every motor
- `service()`
  - loop-side control compute
  - reads cached encoder feedback
  - runs fixed-point PID math
  - prepares `stagedDuty_` / `stagedDrive_`
- `update()`
  - ISR-side apply
  - writes already-prepared drive state to the H-bridge output pins / OCR register

This split is the main reason the current DC path fits within the UART-safe ISR budget while still keeping disabled-motor telemetry and odometry live.

### Current-sense note

The driver still contains optional current-sense support, but
`DC_CURRENT_SENSE_ENABLED = 0`, so the loop-side
motor service path does not spend ADC time sampling CT pins.

### Fixed-point control implementation

The current control path uses a Q16 fixed-point implementation for the hot control logic.

That choice reduces floating-point cost in the control loop and keeps `service()` predictable on AVR. The main active path uses the fixed-point fields and helper routines.

## 4. Encoders and Velocity Feedback

### Encoder counters

The firmware supports:

- external interrupt channels for M1 and M2
- PCINT-backed channels for M3 and M4

That mapping is configured in `ISRScheduler`.

The encoder back-end is intentionally minimal:

- count edges
- provide atomic count/timestamp snapshots
- allow explicit encoder re-zeroing

Velocity is now computed by the DC motor path itself at a fixed `200 Hz` refresh rate using the latest encoder count plus real encoder-edge timing, followed by a small low-pass filter. The same cached velocity is then consumed by:

- the PID control path
- `DC_STATE_ALL` telemetry
- differential-drive odometry

Zero detection uses the real timestamp of the last encoder edge, not scheduler
tick timing, so low-speed motion remains observable while still allowing the
estimate to settle cleanly to zero after motion stops.

## 5. Stepper Control

Stepper control is entirely separate from the DC path.

### `StepperManager`

`StepperManager` owns:

- creation of the four `StepperMotor` instances
- Timer3 configuration
- `timerISR()` dispatch across all stepper channels
- group operations such as `disableAll()` and `emergencyStopAll()`

### Timer3

Timer3 runs at `10 kHz`, so each overflow is `100 us`.

The ISR in `arduino.ino`:

1. samples Timer3 ticks for timing measurement
2. samples UART2 fault edges
3. calls `StepperManager::timerISR()`
4. records the stepper ISR time in `LoopMonitor`

### `StepperMotor`

Each stepper channel owns:

- step / dir / enable pins
- target motion state
- pulse generation state
- homing / limit logic

The Timer3 hot path is trimmed to keep the stepper ISR small:

- `timerCallback()` no longer recomputes `currentVelocity_` with an integer divide on every emitted step
- the current stepper speed is now derived lazily from `stepInterval_` when status is queried
- the stop-interval constant is precomputed instead of recalculated in the deceleration ISR path

Stepper limit checks for homing are read directly in the stepper path, not through the slower `UserIO` cache. That is why the general input sampling cadence does not limit homing responsiveness.

## 6. Servos

Servo control is separate from the Timer1/Timer3 motor paths.

### Hardware model

Servos are driven through a PCA9685 connected over I2C.

`ServoController` owns:

- PCA9685 initialization
- output-enable control via OE pin
- pulse-width to PWM conversion
- single-channel and bulk-channel writes
- I2C health tracking

### Why servo writes are deferred

Servo commands come in through `MessageCenter`, but the actual PCA9685 writes are deferred into `processDeferredServo()`.

That means:

- decode handlers only update desired servo state
- the actual I2C transaction happens later from soft-task context

This avoids putting I2C latency into packet decode or ISR timing-critical code.

### What "enabled" means for servos

The current servo output state is controlled by two layers:

- firmware policy in `SystemManager`
- actual PCA9685 output enable in `ServoController`

If all servo channels are disabled, `processDeferredServo()` disables PCA9685 outputs. If any servo channel is enabled, it ensures the PCA9685 outputs are enabled before applying pulses.

## 7. Robot Kinematics / Odometry

`RobotKinematics` is a singleton-style static module.

It assumes a differential-drive model and uses the configured left/right DC motor channels as odometry sources.

The module maintains:

- `x`, `y`, `theta`
- instantaneous body-frame `vx`, `vy`, `vTheta`

It updates from:

- left/right encoder tick counts
- left/right motor velocity estimates

Current odometry behavior:

- heading resets to `INITIAL_THETA` from `RobotKinematics.h`
- `ODOM_LEFT_MOTOR_DIR_INVERTED` and `ODOM_RIGHT_MOTOR_DIR_INVERTED` apply to
  both encoder deltas and wheel velocities
- `RobotKinematics::reseed()` preserves pose while updating the encoder
  baseline after an odometry-wheel encoder is explicitly re-zeroed
- kinematics telemetry remains available in `IDLE`, so odometry reset does not
  depend on motor enable state

This module is used for telemetry and UI state, not for the low-level control loops.

## 8. Motion-Related Timing Signals

The most important motion timing fields in the human-readable report are:

- `pid`: one Timer1 ISR slice
- `pidr`: one full 4-slot Timer1 round
- `motor`: one loop-side DC compute round
- `step`: one Timer3 ISR service slice

For control correctness, the most useful status counters are usually:

- `req/cmp/app`
- `prep/app`
- `pipe`
- `ControlWin`

Those show whether the pipeline is coherent, not just whether a budget was exceeded.
