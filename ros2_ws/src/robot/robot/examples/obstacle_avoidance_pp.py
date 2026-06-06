from __future__ import annotations

import math
import time
import numpy as np
from robot.hardware_map import (
    Button,
    DEFAULT_FSM_HZ,
    LED,
    INITIAL_THETA_DEG,
    LEFT_WHEEL_DIR_INVERTED,
    LEFT_WHEEL_MOTOR,
    POSITION_UNIT,
    RIGHT_WHEEL_DIR_INVERTED,
    RIGHT_WHEEL_MOTOR,
    WHEEL_BASE,
    WHEEL_DIAMETER,
)
from robot.robot import FirmwareState, Robot
from robot.util import densify_polyline

TAG_ID = 11

# Basic tuning parameters
LINEAR_SPEED = 120.0
MAX_ANGULAR = 1.5  # radians/sec
SAFE_DIST = 250.0  # how close to walls before correction
LOOKAHEAD = 100.0  # pure pursuit
WAYPOINT_TOL = 20.0
DEBUG_PRINT_POINTS = 10  # number of points to show

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
    robot.enable_lidar()
    robot.enable_gps()
    robot.set_tracked_tag_id(TAG_ID)

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

def get_lidar_points(robot: Robot):
    # SDK handles points internally
    if hasattr(robot, "_lidar_points"):
        return np.asarray(robot._lidar_points, dtype=float)
    return np.zeros((0,2))

def compute_wall_repulsion(lidar_pts):
    """Simple repulsion logic to maintain roughly straight path"""
    if len(lidar_pts) == 0:
        return 0.0  # no walls detected
    # distances to left and right
    angles = np.arctan2(lidar_pts[:,1], lidar_pts[:,0])
    dists = np.linalg.norm(lidar_pts, axis=1)
    left_mask = (angles > math.pi/6) & (angles < math.pi/2)
    right_mask = (angles < -math.pi/6) & (angles > -math.pi/2)
    left_dist = np.min(dists[left_mask]) if np.any(left_mask) else None
    right_dist = np.min(dists[right_mask]) if np.any(right_mask) else None
    # compute simple angular correction
    w_correction = 0.0
    Kp = 0.002  # small gain
    if left_dist is not None and left_dist < SAFE_DIST:
        w_correction -= Kp*(SAFE_DIST - left_dist)
    if right_dist is not None and right_dist < SAFE_DIST:
        w_correction += Kp*(SAFE_DIST - right_dist)
    return w_correction

def run(robot: Robot) -> None:
    configure_robot(robot)

    state = "INIT"
    period = 1.0 / float(DEFAULT_FSM_HZ)
    next_tick = time.monotonic()

    # define a simple straight path
    path_control_points = [(0.0,0.0),(0.0,2500.0)]
    path = densify_polyline(path_control_points, spacing=400.0)
    robot._set_obstacle_avoidance_path(path)

    while True:
        if state == "INIT":
            start_robot(robot)
            print("[FSM] INIT - Odometry reset")
            state = "IDLE"

        elif state == "IDLE":
            show_idle_leds(robot)
            robot._draw_lidar_obstacles()

            # debug lidar
            lidar_pts = get_lidar_points(robot)
            if len(lidar_pts) > 0:
                print(f"[DEBUG LiDAR] {len(lidar_pts)} points, first {DEBUG_PRINT_POINTS}:\n{lidar_pts[:DEBUG_PRINT_POINTS]}")
            else:
                print("[DEBUG LiDAR] No points received")

            print("[FSM] IDLE - Press BTN_1 to start moving")
            if robot.get_button(Button.BTN_1):
                print("Starting MOVING")
                state = "MOVING"
            elif robot.get_button(Button.BTN_2):
                print("BTN_2 pressed, shutting down")
                robot.shutdown()

        elif state == "MOVING":
            show_moving_leds(robot)

            # SDK pure pursuit path following
            w_lidar = compute_wall_repulsion(get_lidar_points(robot))
            # get velocity from SDK planner
            state_sdk = robot._nav_follow_pp_path_loop()
            # adjust angular speed for basic wall repulsion
            v, w = robot.get_velocity()
            robot.set_velocity(v, w + w_lidar)

            # debug print
            lidar_pts = get_lidar_points(robot)
            if len(lidar_pts) > 0:
                print(f"[MOVING DEBUG] LiDAR points {len(lidar_pts)}, first {DEBUG_PRINT_POINTS}:\n{lidar_pts[:DEBUG_PRINT_POINTS]}")
                print(f"[MOVING DEBUG] Angular correction from walls: {w_lidar:.4f}")
            else:
                print("[MOVING DEBUG] No LiDAR points detected")

        next_tick += period
        sleep_s = next_tick - time.monotonic()
        if sleep_s > 0.0:
            time.sleep(sleep_s)
        else:
            next_tick = time.monotonic()