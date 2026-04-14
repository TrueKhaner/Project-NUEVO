"""
obstacle_avoidance.py — DWA-based obstacle avoidance example
=============================================================
"""

from __future__ import annotations
import time

from robot.robot import FirmwareState, Robot, Unit
from robot.hardware_map import Button, DEFAULT_FSM_HZ, LED, Motor
from robot.util import densify_polyline
from robot.path_planner import PurePursuitPlanner
import math
import numpy as np


# ---------------------------------------------------------------------------
# Robot build configuration
# ---------------------------------------------------------------------------

POSITION_UNIT = Unit.MM
WHEEL_DIAMETER = 74.0
WHEEL_BASE = 333.0
INITIAL_THETA_DEG = 90.0

LEFT_WHEEL_MOTOR = Motor.DC_M1
LEFT_WHEEL_DIR_INVERTED = False
RIGHT_WHEEL_MOTOR = Motor.DC_M2
RIGHT_WHEEL_DIR_INVERTED = True


def configure_robot(robot: Robot) -> None:
    robot.set_unit(POSITION_UNIT)
    robot.set_odometry_parameters(
        wheel_diameter=WHEEL_DIAMETER,
        wheel_base=WHEEL_BASE,
        initial_theta_deg=INITIAL_THETA_DEG,
        left_motor_id=LEFT_WHEEL_MOTOR,
        left_motor_dir_inverted=LEFT_WHEEL_DIR_INVERTED,
        right_motor_id=RIGHT_WHEEL_MOTOR,
        right_motor_dir_inverted=RIGHT_WHEEL_DIR_INVERTED,
    )


def show_idle_leds(robot: Robot) -> None:
    robot.set_led(LED.GREEN, 0)
    robot.set_led(LED.ORANGE, 255)


def show_moving_leds(robot: Robot) -> None:
    robot.set_led(LED.ORANGE, 0)
    robot.set_led(LED.GREEN, 255)


def start_robot(robot: Robot) -> None:
    robot.set_state(FirmwareState.RUNNING)
    robot.reset_odometry()
    robot.wait_for_pose_update(timeout=0.2)


def run(robot: Robot) -> None:
    configure_robot(robot)

    state = "INIT"
    drive_handle = None
    period = 1.0 / float(DEFAULT_FSM_HZ)
    next_tick = time.monotonic()

    while True:
        if state == "INIT":
            start_robot(robot)
            print("[FSM] INIT (odometry reset)")
            path_control_points = [
                (0.0,   0.0),
                (0.0, 2000.0),
                (2000.0, 2000.0),
            ]
            path = np.float64(densify_polyline(path_control_points, spacing=400.0))
            # ----------------------------------------------------------------
            # DWA parameter guide
            #
            # sample resolution vs. dynamic window width
            #   DW width = 2 * max_acc * period = 2 * 400 * 0.02 = 16 mm/s
            #   Using a step > DW width produces only 1 sample → no optimisation.
            #   Rule: step << DW_width.  Here 3 mm/s gives ~5 linear samples.
            #
            # gain_obs_base tuning
            #   obs_cost = 1 / dist_m.  For a 400mm obstacle: obs_cost ≈ 2.5.
            #   goal_cost for a 1m path detour ≈ 1.0.
            #   Break-even: gain_obs * 2.5  ==  gain_goal * 1.0
            #                      gain_obs  ==  2.0 / 2.5 ≈ 0.8  (minimum)
            #   Use 4.0 so obstacle avoidance actively steers early, not just at
            #   the collision boundary.
            #
            # obstacles_range_mm
            #   Must exceed v_max * predict_time + lidar_offset so trajectories
            #   are checked against all reachable obstacles.
            #   min = 300 * 2.0 + 100 = 700 mm → use 800 mm.
            #
            # gain_speed
            #   Penalises stopping (cost = -v / 1000).  A high value (1.0)
            #   overrides obstacle avoidance and keeps the robot at full speed
            #   into obstacles.  Lower to 0.1.
            # ----------------------------------------------------------------
            robot._nav_follow_dwa_path(
                max_vel_mm=300.0,
                max_acc_mm=400.0,
                max_angular_rad=0.8,
                max_angular_acc_rad=1.5,
                lookahead_mm=200.0,
                advance_radius_mm=150.0,
                tolerance_mm=100.0,
                gains_of_costs=[2.0, 0.05, 4.0, 0.1, 0.2], # [gain_goal, gain_heading, gain_obs_base, gain_speed, gain_path]
                period=period,
                predict_time=2.0,
                predict_velocity_samples_resolution=[3.0, 0.01],
                obstacles_range_mm=800.0,
                ttc_weight=0.1,
            )
            print("Path is ready, Entering IDLE state.")
            state = "IDLE"

        elif state == "IDLE":
            show_idle_leds(robot)
            robot._draw_lidar_obstacles()
            print("[FSM] IDLE - Press BTN_1 to enter MOVING state.")
            if robot.get_button(Button.BTN_1):
                print("Start Moving!")
                print("[FSM] MOVING")
                state = "MOVING"

        elif state == "MOVING":
            show_moving_leds(robot)
            # robot._draw_lidar_obstacles()
            state = robot._nav_follow_path_loop(path, period)

        # FSM refresh rate control
        next_tick += period
        sleep_s = next_tick - time.monotonic()
        if sleep_s > 0.0:
            time.sleep(sleep_s)
        else:
            next_tick = time.monotonic()
