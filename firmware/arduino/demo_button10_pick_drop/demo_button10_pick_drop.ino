/*
  Button-10 Pick-And-Drop Demo Firmware
  =====================================
  Upload this sketch for the initial camera-free physical demo.

  Protocol from Raspberry Pi, newline terminated:
    PING
    STATUS?
    SET_SPEEDS <z> <x> <r>
    HOME_ZX
    HOME_R
    ZERO_ALL
    GOTO_Z <steps> [speed]
    GOTO_X <steps> [speed]
    GOTO_R <steps> [speed]
    OPEN_GRIP
    CLOSE_GRIP

  Telemetry to Raspberry Pi:
    BUTTON PRESSED / BUTTON RELEASED
    POS <z> <x> <r>
    HOME <z_homed> <x_homed> <r_homed>
    SERVO 13 <feedback_us> <OPEN|CLOSED|MOVING>
    DONE <token>
*/

#include <Arduino.h>
#include <Wire.h>

// =============================================================================
// HARDWARE PIN MAP - STRICT DEMO ASSIGNMENTS
// =============================================================================
const uint8_t PIN_BUTTON_10 = 10;  // Physical trigger button, active low.

const uint8_t PIN_LIM1_Z_BOTTOM = 40;
const uint8_t PIN_LIM2_X_HOME = 41;
const uint8_t PIN_LIM3_R_HOME = 48;

const uint8_t PIN_ST2_STEP = 15;
const uint8_t PIN_ST2_DIR = 23;
const uint8_t PIN_ST2_EN = 27;

const uint8_t PIN_ST3_STEP = 32;
const uint8_t PIN_ST3_DIR = 24;
const uint8_t PIN_ST3_EN = 28;

const uint8_t PIN_ST4_STEP = 33;
const uint8_t PIN_ST4_DIR = 25;
const uint8_t PIN_ST4_EN = 29;

const uint8_t GRIPPER_SERVO_CHANNEL = 13;
const uint8_t PCA9685_ADDR = 0x40;

// =============================================================================
// TUNABLE PARAMETERS
// =============================================================================
uint16_t STEPPER_2_SPEED = 450;
uint16_t STEPPER_3_SPEED = 500;
uint16_t STEPPER_4_SPEED = 350;

const int8_t Z_HOME_DIR = -1;
const int8_t X_HOME_DIR = -1;
const int8_t R_HOME_DIR = -1;

const uint16_t HOMING_SPEED_Z = 250;
const uint16_t HOMING_SPEED_X = 250;
const uint16_t HOMING_SPEED_R = 200;

const uint16_t SERVO_OPEN_US = 1000;
const uint16_t SERVO_CLOSED_US = 2000;
const uint16_t SERVO_TOLERANCE_US = 40;
const uint16_t SERVO_SETTLE_MS = 700;

const uint32_t STATUS_PERIOD_MS = 200;
const uint16_t STEP_PULSE_HIGH_US = 4;

// =============================================================================
// STEPPER MODEL
// =============================================================================
struct Axis {
  const char *name;
  uint8_t stepPin;
  uint8_t dirPin;
  uint8_t enPin;
  uint8_t limitPin;
  int8_t homeDir;
  int32_t position;
  int32_t target;
  uint16_t speed;
  uint32_t intervalUs;
  uint32_t lastStepUs;
  bool enabled;
  bool moving;
  bool homing;
  bool homed;
};

Axis zAxis = {"Z", PIN_ST2_STEP, PIN_ST2_DIR, PIN_ST2_EN, PIN_LIM1_Z_BOTTOM, Z_HOME_DIR, 0, 0, STEPPER_2_SPEED, 0, 0, false, false, false, false};
Axis xAxis = {"X", PIN_ST3_STEP, PIN_ST3_DIR, PIN_ST3_EN, PIN_LIM2_X_HOME, X_HOME_DIR, 0, 0, STEPPER_3_SPEED, 0, 0, false, false, false, false};
Axis rAxis = {"R", PIN_ST4_STEP, PIN_ST4_DIR, PIN_ST4_EN, PIN_LIM3_R_HOME, R_HOME_DIR, 0, 0, STEPPER_4_SPEED, 0, 0, false, false, false, false};

char commandBuffer[96];
uint8_t commandIndex = 0;
uint32_t lastStatusMs = 0;
bool lastButtonDown = false;
uint16_t lastServoCommandUs = SERVO_OPEN_US;

// =============================================================================
// PCA9685 SERVO HELPERS
// =============================================================================
void pcaWrite(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(PCA9685_ADDR);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

uint8_t pcaRead(uint8_t reg) {
  Wire.beginTransmission(PCA9685_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom(PCA9685_ADDR, (uint8_t)1);
  if (Wire.available()) {
    return Wire.read();
  }
  return 0;
}

void pcaInitServo50Hz() {
  Wire.begin();
  pcaWrite(0x00, 0x10);       // MODE1 sleep.
  pcaWrite(0xFE, 121);        // 50 Hz prescale for 25 MHz clock.
  pcaWrite(0x01, 0x04);       // MODE2 totem pole.
  pcaWrite(0x00, 0x20);       // MODE1 auto-increment, awake.
  delay(5);
}

uint16_t servoUsToTicks(uint16_t pulseUs) {
  return (uint32_t)pulseUs * 4096UL / 20000UL;
}

uint16_t servoTicksToUs(uint16_t ticks) {
  return (uint32_t)ticks * 20000UL / 4096UL;
}

void setServoUs(uint8_t channel, uint16_t pulseUs) {
  uint16_t off = servoUsToTicks(pulseUs);
  uint8_t reg = 0x06 + 4 * channel;
  pcaWrite(reg + 0, 0);
  pcaWrite(reg + 1, 0);
  pcaWrite(reg + 2, off & 0xFF);
  pcaWrite(reg + 3, off >> 8);
  lastServoCommandUs = pulseUs;
}

uint16_t readServoFeedbackUs(uint8_t channel) {
  uint8_t reg = 0x06 + 4 * channel;
  uint16_t off = pcaRead(reg + 2) | ((uint16_t)pcaRead(reg + 3) << 8);
  off &= 0x0FFF;
  return servoTicksToUs(off);
}

const char *servoStateName() {
  uint16_t feedback = readServoFeedbackUs(GRIPPER_SERVO_CHANNEL);
  if (abs((int)feedback - (int)SERVO_OPEN_US) <= SERVO_TOLERANCE_US) {
    return "OPEN";
  }
  if (abs((int)feedback - (int)SERVO_CLOSED_US) <= SERVO_TOLERANCE_US) {
    return "CLOSED";
  }
  return "MOVING";
}

void emitServoTelemetry() {
  Serial.print(F("SERVO "));
  Serial.print(GRIPPER_SERVO_CHANNEL);
  Serial.print(' ');
  Serial.print(readServoFeedbackUs(GRIPPER_SERVO_CHANNEL));
  Serial.print(' ');
  Serial.println(servoStateName());
}

// =============================================================================
// STEPPER HELPERS
// =============================================================================
void initAxis(Axis &axis) {
  pinMode(axis.stepPin, OUTPUT);
  pinMode(axis.dirPin, OUTPUT);
  pinMode(axis.enPin, OUTPUT);
  pinMode(axis.limitPin, INPUT_PULLUP);
  digitalWrite(axis.stepPin, LOW);
  digitalWrite(axis.enPin, HIGH);  // A4988/DRV8825 enable is active low.
}

void setAxisEnabled(Axis &axis, bool enabled) {
  axis.enabled = enabled;
  digitalWrite(axis.enPin, enabled ? LOW : HIGH);
}

bool limitTriggered(const Axis &axis) {
  return digitalRead(axis.limitPin) == LOW;
}

void setAxisSpeed(Axis &axis, uint16_t speed) {
  axis.speed = max((uint16_t)1, speed);
  axis.intervalUs = 1000000UL / axis.speed;
}

void startMove(Axis &axis, int32_t target, uint16_t speed) {
  setAxisSpeed(axis, speed);
  setAxisEnabled(axis, true);
  axis.target = target;
  axis.homing = false;
  axis.moving = axis.position != axis.target;
}

void startHome(Axis &axis, uint16_t speed) {
  setAxisSpeed(axis, speed);
  setAxisEnabled(axis, true);
  axis.homing = true;
  axis.moving = true;
  axis.homed = false;
}

void updateAxis(Axis &axis) {
  if (!axis.moving) {
    return;
  }

  if (axis.homing && limitTriggered(axis)) {
    axis.position = 0;
    axis.target = 0;
    axis.moving = false;
    axis.homing = false;
    axis.homed = true;
    return;
  }

  int8_t dir = axis.homing ? axis.homeDir : (axis.target > axis.position ? 1 : -1);
  if (!axis.homing && axis.position == axis.target) {
    axis.moving = false;
    return;
  }

  uint32_t now = micros();
  if (now - axis.lastStepUs < axis.intervalUs) {
    return;
  }
  axis.lastStepUs = now;

  digitalWrite(axis.dirPin, dir > 0 ? HIGH : LOW);
  digitalWrite(axis.stepPin, HIGH);
  delayMicroseconds(STEP_PULSE_HIGH_US);
  digitalWrite(axis.stepPin, LOW);
  axis.position += dir;
}

void updateAllAxes() {
  updateAxis(zAxis);
  updateAxis(xAxis);
  updateAxis(rAxis);
}

void waitUntil(bool (*donePredicate)()) {
  while (!donePredicate()) {
    processSerial();
    updateAllAxes();
    emitPeriodicStatus();
  }
}

bool zxHomeDone() {
  return !zAxis.moving && !xAxis.moving && zAxis.homed && xAxis.homed;
}

bool rHomeDone() {
  return !rAxis.moving && rAxis.homed;
}

bool zDone() { return !zAxis.moving; }
bool xDone() { return !xAxis.moving; }
bool rDone() { return !rAxis.moving; }

// =============================================================================
// TELEMETRY AND COMMANDS
// =============================================================================
void emitStatus() {
  Serial.print(F("STATUS button="));
  Serial.print(digitalRead(PIN_BUTTON_10) == LOW ? 1 : 0);
  Serial.print(F(" z="));
  Serial.print(zAxis.position);
  Serial.print(F(" x="));
  Serial.print(xAxis.position);
  Serial.print(F(" r="));
  Serial.print(rAxis.position);
  Serial.print(F(" servo_us="));
  Serial.print(readServoFeedbackUs(GRIPPER_SERVO_CHANNEL));
  Serial.print(F(" servo="));
  Serial.println(servoStateName());

  Serial.print(F("HOME "));
  Serial.print(zAxis.homed ? 1 : 0);
  Serial.print(' ');
  Serial.print(xAxis.homed ? 1 : 0);
  Serial.print(' ');
  Serial.println(rAxis.homed ? 1 : 0);

  Serial.print(F("POS "));
  Serial.print(zAxis.position);
  Serial.print(' ');
  Serial.print(xAxis.position);
  Serial.print(' ');
  Serial.println(rAxis.position);
  emitServoTelemetry();
}

void emitPeriodicStatus() {
  if (millis() - lastStatusMs >= STATUS_PERIOD_MS) {
    lastStatusMs = millis();
    emitStatus();
  }
}

void pollButton10() {
  bool down = digitalRead(PIN_BUTTON_10) == LOW;
  if (down != lastButtonDown) {
    lastButtonDown = down;
    Serial.println(down ? F("BUTTON PRESSED") : F("BUTTON RELEASED"));
  }
}

void done(const __FlashStringHelper *token) {
  Serial.print(F("DONE "));
  Serial.println(token);
}

void executeCommand(char *cmd) {
  char *verb = strtok(cmd, " ");
  if (!verb) return;

  if (strcmp(verb, "PING") == 0) {
    Serial.println(F("DONE PONG"));
    return;
  }
  if (strcmp(verb, "STATUS?") == 0) {
    emitStatus();
    return;
  }
  if (strcmp(verb, "SET_SPEEDS") == 0) {
    STEPPER_2_SPEED = atoi(strtok(NULL, " "));
    STEPPER_3_SPEED = atoi(strtok(NULL, " "));
    STEPPER_4_SPEED = atoi(strtok(NULL, " "));
    setAxisSpeed(zAxis, STEPPER_2_SPEED);
    setAxisSpeed(xAxis, STEPPER_3_SPEED);
    setAxisSpeed(rAxis, STEPPER_4_SPEED);
    done(F("SPEEDS"));
    return;
  }
  if (strcmp(verb, "HOME_ZX") == 0) {
    startHome(zAxis, HOMING_SPEED_Z);
    startHome(xAxis, HOMING_SPEED_X);
    waitUntil(zxHomeDone);
    done(F("HOME_ZX"));
    return;
  }
  if (strcmp(verb, "HOME_R") == 0) {
    startHome(rAxis, HOMING_SPEED_R);
    waitUntil(rHomeDone);
    done(F("HOME_R"));
    return;
  }
  if (strcmp(verb, "ZERO_ALL") == 0) {
    zAxis.position = xAxis.position = rAxis.position = 0;
    zAxis.target = xAxis.target = rAxis.target = 0;
    zAxis.homed = xAxis.homed = rAxis.homed = true;
    done(F("ZERO_ALL"));
    return;
  }
  if (strcmp(verb, "GOTO_Z") == 0) {
    int32_t target = atol(strtok(NULL, " "));
    char *speedArg = strtok(NULL, " ");
    startMove(zAxis, target, speedArg ? atoi(speedArg) : STEPPER_2_SPEED);
    waitUntil(zDone);
    done(F("Z"));
    return;
  }
  if (strcmp(verb, "GOTO_X") == 0) {
    int32_t target = atol(strtok(NULL, " "));
    char *speedArg = strtok(NULL, " ");
    startMove(xAxis, target, speedArg ? atoi(speedArg) : STEPPER_3_SPEED);
    waitUntil(xDone);
    done(F("X"));
    return;
  }
  if (strcmp(verb, "GOTO_R") == 0) {
    int32_t target = atol(strtok(NULL, " "));
    char *speedArg = strtok(NULL, " ");
    startMove(rAxis, target, speedArg ? atoi(speedArg) : STEPPER_4_SPEED);
    waitUntil(rDone);
    done(F("R"));
    return;
  }
  if (strcmp(verb, "OPEN_GRIP") == 0) {
    setServoUs(GRIPPER_SERVO_CHANNEL, SERVO_OPEN_US);
    delay(SERVO_SETTLE_MS);
    emitServoTelemetry();
    Serial.println(strcmp(servoStateName(), "OPEN") == 0 ? F("DONE GRIP OPEN") : F("ERROR GRIP NOT_OPEN"));
    return;
  }
  if (strcmp(verb, "CLOSE_GRIP") == 0) {
    setServoUs(GRIPPER_SERVO_CHANNEL, SERVO_CLOSED_US);
    delay(SERVO_SETTLE_MS);
    emitServoTelemetry();
    Serial.println(strcmp(servoStateName(), "CLOSED") == 0 ? F("DONE GRIP CLOSED") : F("ERROR GRIP NOT_CLOSED"));
    return;
  }

  Serial.print(F("ERROR UNKNOWN_CMD "));
  Serial.println(verb);
}

void processSerial() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (commandIndex > 0) {
        commandBuffer[commandIndex] = '\0';
        executeCommand(commandBuffer);
        commandIndex = 0;
      }
    } else if (commandIndex < sizeof(commandBuffer) - 1) {
      commandBuffer[commandIndex++] = c;
    } else {
      commandIndex = 0;
      Serial.println(F("ERROR COMMAND_TOO_LONG"));
    }
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(PIN_BUTTON_10, INPUT_PULLUP);
  initAxis(zAxis);
  initAxis(xAxis);
  initAxis(rAxis);
  setAxisSpeed(zAxis, STEPPER_2_SPEED);
  setAxisSpeed(xAxis, STEPPER_3_SPEED);
  setAxisSpeed(rAxis, STEPPER_4_SPEED);
  pcaInitServo50Hz();
  setServoUs(GRIPPER_SERVO_CHANNEL, SERVO_OPEN_US);
  delay(500);
  lastButtonDown = digitalRead(PIN_BUTTON_10) == LOW;
  Serial.println(F("READY BUTTON10_DEMO"));
  emitStatus();
}

void loop() {
  pollButton10();
  processSerial();
  updateAllAxes();
  emitPeriodicStatus();
}
