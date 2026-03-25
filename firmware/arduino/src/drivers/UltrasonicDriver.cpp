/**
 * @file UltrasonicDriver.cpp
 * @brief Implementation of Qwiic Ultrasonic driver wrapper
 */

#include "UltrasonicDriver.h"
#include "../modules/DebugLog.h"

UltrasonicDriver::UltrasonicDriver()
    : connected_(false)
    , i2cAddr_(0x2F)
    , consecutiveFailures_(0)
    , nextAttemptMs_(0)
    , lastLogMs_(0)
    , zeroDistanceCount_(0)
    , lastDistanceMm_(0)
{
}

bool UltrasonicDriver::init(uint8_t i2cAddr) {
#if ULTRASONIC_COUNT > 0
    i2cAddr_ = i2cAddr;
    Wire.setClock(ULTRASONIC_I2C_CLOCK_HZ);

    if (!ultrasonic_.begin(i2cAddr_)) {
        Wire.setClock(I2C_BUS_CLOCK_HZ);
        DebugLog::printf_P(PSTR("[ULTRA] addr=0x%02X init fail\n"), i2cAddr_);
        connected_ = false;
        consecutiveFailures_ = kFailureThreshold;
        nextAttemptMs_ = millis() + kReconnectBackoffMs;
        return false;
    }

    bool wasDisconnected = !connected_;
    connected_ = true;
    consecutiveFailures_ = 0;
    nextAttemptMs_ = 0;
    zeroDistanceCount_ = 0;
    Wire.setClock(I2C_BUS_CLOCK_HZ);

    if (wasDisconnected) {
        DebugLog::printf_P(PSTR("[ULTRA] addr=0x%02X connected\n"), i2cAddr_);
    }

    return true;
#else
    (void)i2cAddr;
    return false;
#endif
}

uint16_t UltrasonicDriver::getDistanceMm() {
#if ULTRASONIC_COUNT > 0
    uint32_t now = millis();

    if (now < nextAttemptMs_) {
        return ULTRASONIC_ERROR_DISTANCE;
    }

    if (!connected_) {
        if (!init(i2cAddr_)) {
            nextAttemptMs_ = now + kReconnectBackoffMs;
            return ULTRASONIC_ERROR_DISTANCE;
        }
    }

    Wire.setClock(ULTRASONIC_I2C_CLOCK_HZ);

    uint16_t distance = 0;

    // triggerAndRead() returns sfTkError_t; 0 (ksfTkErrOk) means success
    uint8_t err = (uint8_t)ultrasonic_.triggerAndRead(distance);
    if (err != 0U) {
        Wire.setClock(I2C_BUS_CLOCK_HZ);
        if (consecutiveFailures_ < 0xFFU) {
            consecutiveFailures_++;
        }
        if ((uint32_t)(now - lastLogMs_) >= 500U) {
            DebugLog::printf_P(PSTR("[ULTRA] addr=0x%02X read err=%u dist=%u fail=%u\n"),
                               i2cAddr_,
                               (unsigned)err,
                               (unsigned)distance,
                               (unsigned)consecutiveFailures_);
            lastLogMs_ = now;
        }
        if (consecutiveFailures_ >= kFailureThreshold) {
            connected_ = false;
            nextAttemptMs_ = now + kReconnectBackoffMs;
            DebugLog::printf_P(PSTR("[ULTRA] addr=0x%02X offline\n"), i2cAddr_);
        } else {
            nextAttemptMs_ = now + kRetryBackoffMs;
        }
        return ULTRASONIC_ERROR_DISTANCE;
    }

    Wire.setClock(I2C_BUS_CLOCK_HZ);

    if (distance == 0U) {
        zeroDistanceCount_++;
        if ((uint32_t)(now - lastLogMs_) >= 500U) {
            DebugLog::printf_P(PSTR("[ULTRA] addr=0x%02X zero dist=%u zero=%u last=%u\n"),
                               i2cAddr_,
                               (unsigned)distance,
                               (unsigned)zeroDistanceCount_,
                               (unsigned)lastDistanceMm_);
            lastLogMs_ = now;
        }
        consecutiveFailures_ = 0;
        nextAttemptMs_ = 0;
        connected_ = true;
        return lastDistanceMm_;
    }

    connected_ = true;
    consecutiveFailures_ = 0;
    nextAttemptMs_ = 0;
    zeroDistanceCount_ = 0;
    lastDistanceMm_ = distance;

    // The library already returns mm; 0 indicates a sensor error
    return distance;
#else
    return ULTRASONIC_ERROR_DISTANCE;
#endif
}
