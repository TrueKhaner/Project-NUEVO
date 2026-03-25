#ifndef DC_MOTOR_BRINGUP_H
#define DC_MOTOR_BRINGUP_H

#include <Arduino.h>

#include "../config.h"
#include "../drivers/DCMotor.h"
#include "EncoderCounter.h"

class DCMotorBringup {
public:
    static uint16_t countsPerRev();

    static void initAll(DCMotor *motors,
                        IEncoderCounter &encoder1,
                        IEncoderCounter &encoder2,
                        IEncoderCounter &encoder3,
                        IEncoderCounter &encoder4);

private:
    static void initOne(DCMotor &motor,
                        uint8_t motorId,
                        IEncoderCounter &encoder,
                        uint8_t encA,
                        uint8_t encB,
                        bool encoderInvert,
                        bool motorInvert,
                        uint8_t pinEN,
                        uint8_t pinIN1,
                        uint8_t pinIN2);
};

#endif // DC_MOTOR_BRINGUP_H
