# Sensors and Shared I2C Bus

This document describes the sensor stack that is actually supported in the
current Arduino firmware.

## Supported on Arduino

- ICM-20948 IMU
- battery / 5 V / servo-rail voltage monitoring
- PCA9685 servo controller on the same `Wire` bus

## Not supported on Arduino

- Garmin / SparkFun LIDAR-Lite v4
- SparkFun Qwiic ultrasonic sensor(s)

The LIDAR-Lite v4 and the Qwiic ultrasonic are now treated as Raspberry Pi-side
sensors only. The Arduino firmware no longer contains an ultrasonic polling or
telemetry path. If the robot uses lidar or ultrasonic sensing, connect those
sensors to the Pi-side bus and use the Pi-side tooling under
[`ros2_ws/tests/`](../../ros2_ws/tests/).

## 1. Shared `Wire` Bus Ownership

The Arduino Mega uses one global `Wire` bus for all I2C peripherals that remain
on the MCU side:

- IMU
- PCA9685 servo controller

The firmware does not implement a separate software I2C arbiter object. Bus
access is serialized by execution context instead:

- only one soft task runs at a time
- no `Wire` calls are allowed from Timer1 / Timer3 / encoder ISR paths
- deferred servo work runs from the UART soft task, not from decode context

So there is no true concurrent I2C access, but a slow transaction can still
lengthen a soft task and add loop jitter.

## 2. Why I2C Stays Out of ISRs

`Wire` is too slow and too opaque for the hard real-time budgets in this
firmware.

The current rules are:

- no I2C in `ISR(TIMER1_OVF_vect)`
- no I2C in `ISR(TIMER3_OVF_vect)`
- no I2C in encoder interrupts
- no I2C in TLV decode handlers

This is why:

- sensor polling lives in `SensorManager::tick()`
- servo writes are deferred and applied later from `MessageCenter::processDeferred()`
- magnetometer calibration save/apply/clear side effects are deferred as well

## 3. `SensorManager`

`SensorManager` is the central owner of Arduino-side sensing.

It runs from `taskSensors()` at `100 Hz` and dispatches internal work at three
rates:

- `update100Hz()` -> alternating IMU read and Fusion phases
- `update50Hz()` -> reserved medium-rate lane
- `update10Hz()` -> voltage rails

`taskSensors()` also calls `UserIO::sampleInputs()` so button and limit-switch
sampling stays with state acquisition, not safety policy.

### IMU path

Files:

- `drivers/IMUDriver.*`
- `modules/SensorManager.*`
- `lib/Fusion/*`

Implementation:

- `IMUDriver` wraps the SparkFun ICM-20948 library
- the firmware does not use the DMP path
- `SensorManager::update100Hz()` alternates between:
  - a raw ICM-20948 read phase
  - a Fusion update phase using the most recent sample
- if magnetometer calibration is active, the fusion path is 9-DoF
- otherwise the firmware uses 6-DoF fusion without magnetometer correction

Current timing model:

- blocking I2C read at an effective `25 Hz`
- Fusion update at an effective `25 Hz` on the alternating sensor tick
- `FusionWrapper` is initialized with `IMU_UPDATE_FREQ_HZ = 25`
- timing reported separately in the status output as `imu`

### Voltage monitoring

Voltage rails are not I2C devices. They are sampled through the ADC in
`SensorManager::updateVoltages()`.

Measured rails:

- battery input
- 5 V rail
- servo rail

These values feed:

- `SystemManager` battery-enable and battery-fault policy
- status reporting
- TLV telemetry

## 4. Magnetometer Calibration

Magnetometer calibration is now split between the Arduino and the bridge.

Arduino responsibilities:

- enter/leave magnetometer sampling mode
- stream `SENSOR_MAG_CAL_STATUS`
- keep sampling state out of ISR paths
- apply and persist the final calibration in EEPROM
- apply the saved correction before feeding mag data into Fusion

Bridge responsibilities:

- collect raw magnetometer samples during sampling mode
- evaluate sample coverage / quality
- fit the final hard-iron offset plus soft-iron `3×3` matrix
- send the final result back in one `MAG_CAL_APPLY` command

State machine:

- `IDLE`
- `SAMPLING`
- `COMPLETE`
- `SAVED`
- `ERROR`

Important implementation details:

- sampling is updated from the IMU update path
- `PersistentStorage::init()` must run before `SensorManager::init()` so saved
  calibration can be loaded on boot
- `startMagCalibration()` temporarily disables any previously active mag
  calibration so the bridge sees raw magnetometer data
- `cancelMagCalibration()` restores the previous active calibration, if one
  existed
- persistence goes through `PersistentStorage`
- start/stop/save/apply/clear side effects are deferred out of message decode
  context

The firmware still tracks min/max and midpoint offsets while sampling because
they are useful for progress reporting and for a hard-iron fallback if the
bridge cannot produce a good soft-iron fit. The normal path is:

1. UI enters calibration mode and ensures the firmware is in `IDLE`
2. bridge sends `MAG_CAL_START`
3. user moves the robot in a figure-8 while also rotating and tilting it
   through many orientations
4. bridge auto-sends `MAG_CAL_APPLY` once sample coverage is good enough
5. firmware saves the offset + matrix and returns to 9-DoF fusion

So the calibration flow touches IMU I2C, EEPROM, and Fusion input correction,
while still staying out of ISR paths.

## 5. PCA9685 Servo Controller on the Same Bus

This is the most important I2C interaction on the Arduino side.

Files:

- `drivers/ServoController.*`
- `modules/MessageCenter.*`

Implementation:

- `ServoController` owns PCA9685 initialization, OE control, pulse-width
  conversion, and grouped channel writes
- the PCA9685 shares the same `Wire` bus as the IMU
- servo I2C writes are never done directly from packet decode
- decode handlers only mark pending servo work
- `MessageCenter::processDeferred()` later applies the pending writes in soft
  task context

This deferred path exists specifically to keep the decode-side UART path short
and to avoid I2C work in timing-sensitive contexts.

## 6. Timing and Debug Output

There are two different sensor-related timing views in the status report.

### Whole soft sensor task

Reported as:

- `sensor a/b/c (...) us`

This is the timing of `taskSensors()` as a whole, including:

- `SensorManager::tick()`
- `UserIO::sampleInputs()`

## 7. Bus Settings and Stability Notes

Current bus settings are:

- `I2C_BUS_CLOCK_HZ = 100000`
- `SERVO_I2C_CLOCK_HZ = 100000`
- `I2C_WIRE_TIMEOUT_US = 5000`

The PCA9685 servo driver explicitly forces the shared bus back to
`SERVO_I2C_CLOCK_HZ` before runtime servo transactions. That avoids the case
where other shared-bus users have already touched `Wire` and later servo writes
inherit the wrong bus speed on marginal hardware.

The servo controller and IMU are expected to coexist on that bus. If a new I2C
device is added on the Arduino side, it should follow
the same rules:

- no ISR use
- no decode-context use
- explicit timeout behavior if the library can block

## 8. Pi-Side Range Sensors

Range sensors are Pi-side only in the released system design.

- Arduino firmware: no lidar or ultrasonic support
- Raspberry Pi side: lidar and ultrasonic support live there instead

Keep that separation unless the hardware topology changes enough to justify a
new Arduino-side design.
