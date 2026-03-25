/**
 * @file RobotKinematics.cpp
 * @brief Differential-drive odometry implementation
 *
 * To adapt for a different drive model (e.g., Ackermann steering):
 *   1. Update the constants in RobotKinematics.h to match your geometry.
 *   2. Replace the update() body below with your kinematics equations.
 *      - reset() and the static members do not need to change.
 *      - The interface (getX, getY, getTheta, getVx, getVy, getVTheta) stays the same.
 */

#include "RobotKinematics.h"
#include "../config.h"
#include <math.h>

namespace {
constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 2.0f * kPi;
constexpr float kInitialThetaRad = INITIAL_THETA * kPi / 180.0f;
}

// ============================================================================
// STATIC MEMBER INITIALIZATION
// ============================================================================

float   RobotKinematics::x_             = 0.0f;
float   RobotKinematics::y_             = 0.0f;
float   RobotKinematics::theta_         = kInitialThetaRad;
float   RobotKinematics::vx_            = 0.0f;
float   RobotKinematics::vy_            = 0.0f;
float   RobotKinematics::vTheta_        = 0.0f;
int32_t RobotKinematics::prevLeftTicks_ = 0;
int32_t RobotKinematics::prevRightTicks_= 0;

// ============================================================================
// COMPILE-TIME VALIDATION
// ============================================================================

static_assert(ODOM_LEFT_MOTOR  < NUM_DC_MOTORS, "ODOM_LEFT_MOTOR must be less than NUM_DC_MOTORS");
static_assert(ODOM_RIGHT_MOTOR < NUM_DC_MOTORS, "ODOM_RIGHT_MOTOR must be less than NUM_DC_MOTORS");
static_assert(ODOM_LEFT_MOTOR  != ODOM_RIGHT_MOTOR, "ODOM_LEFT_MOTOR and ODOM_RIGHT_MOTOR must be different");

// ============================================================================
// mm-per-tick conversion factors (computed once at compile time)
// ============================================================================

constexpr float kMmPerTick[4] = {
    (kPi * WHEEL_DIAMETER_MM) / (float)(ENCODER_PPR * ENCODER_1_MODE),
    (kPi * WHEEL_DIAMETER_MM) / (float)(ENCODER_PPR * ENCODER_2_MODE),
    (kPi * WHEEL_DIAMETER_MM) / (float)(ENCODER_PPR * ENCODER_3_MODE),
    (kPi * WHEEL_DIAMETER_MM) / (float)(ENCODER_PPR * ENCODER_4_MODE)
};

// ============================================================================
// PUBLIC API
// ============================================================================

void RobotKinematics::reset(int32_t leftTicks, int32_t rightTicks) {
    x_              = 0.0f;
    y_              = 0.0f;
    theta_          = kInitialThetaRad;
    vx_             = 0.0f;
    vy_             = 0.0f;
    vTheta_         = 0.0f;
    prevLeftTicks_  = leftTicks;
    prevRightTicks_ = rightTicks;
}

void RobotKinematics::reseed(int32_t leftTicks, int32_t rightTicks) {
    vx_ = 0.0f;
    vy_ = 0.0f;
    vTheta_ = 0.0f;
    prevLeftTicks_ = leftTicks;
    prevRightTicks_ = rightTicks;
}

void RobotKinematics::update(int32_t leftTicks, int32_t rightTicks,
                              float leftVelTps, float rightVelTps)
{
    // ---- Odometry integration ----
    int32_t dL = leftTicks  - prevLeftTicks_;
    if (ODOM_LEFT_MOTOR_DIR_INVERTED){
        dL = -dL;
    }

    int32_t dR = rightTicks - prevRightTicks_;
    if (ODOM_RIGHT_MOTOR_DIR_INVERTED){
        dR = -dR;
    }

    prevLeftTicks_  = leftTicks;
    prevRightTicks_ = rightTicks;

    if (dL != 0 || dR != 0) {
        float dLeft   = (float)dL * kMmPerTick[ODOM_LEFT_MOTOR];
        float dRight  = (float)dR * kMmPerTick[ODOM_RIGHT_MOTOR];
        float dCenter = (dLeft + dRight) * 0.5f;
        float dTheta  = (dRight - dLeft) / WHEEL_BASE_MM;

        // Midpoint heading integration (reduces discretization error)
        float headingMid = theta_ + dTheta * 0.5f;
        x_     += dCenter * cosf(headingMid);
        y_     += dCenter * sinf(headingMid);
        theta_ = theta_ + dTheta; // We intentionally NOT mod theta to 2 PI here.
    }

    // ---- Instantaneous body-frame velocities ----
    if (ODOM_LEFT_MOTOR_DIR_INVERTED) {
        leftVelTps = -leftVelTps;
    }
    if (ODOM_RIGHT_MOTOR_DIR_INVERTED) {
        rightVelTps = -rightVelTps;
    }

    float vLeft  = leftVelTps  * kMmPerTick[ODOM_LEFT_MOTOR];
    float vRight = rightVelTps * kMmPerTick[ODOM_RIGHT_MOTOR];

    vx_     = (vLeft + vRight) * 0.5f;
    vy_     = 0.0f;  // Always 0 for differential drive
    vTheta_ = (vRight - vLeft) / WHEEL_BASE_MM;
}
