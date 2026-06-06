
from __future__ import annotations

import math
import numpy as np


# =============================================================================
# Base class
# =============================================================================

class PathPlanner:
    """
    Abstract base for path planning algorithms.

    Subclasses implement compute_velocity() and optionally get_obstacles().
    """

    def compute_velocity(
        self,
        pose: tuple[float, float, float],
        waypoints: list[tuple[float, float]],
        max_linear: float,
    ) -> tuple[float, float]:
        """
        Return (linear, angular) velocity command.

          pose       — (x, y, theta_rad) in any consistent unit
          waypoints  — remaining waypoints in the same unit, nearest first
          max_linear — maximum forward speed in that unit/s

        Returns (linear, angular_rad_s).
        """
        raise NotImplementedError

    def get_obstacles(self) -> list:
        """
        Return a list of obstacle positions in the robot's frame.
        Override when a 2D lidar topic is available.
        """
        return []


# =============================================================================
# Pure Pursuit
# =============================================================================

class PurePursuitPlanner(PathPlanner):
    """
    Pure-pursuit path follower for differential drive.

    Steers toward a lookahead point on the path. Works well for smooth
    curves. The lookahead_dist controls the trade-off between responsiveness
    and smoothness.
    """

    def __init__(
        self,
        lookahead_dist: float = 150,
        max_angular: float = 2.0,
        goal_tolerance: float = 20.0,
    ) -> None:
        self._lookahead = lookahead_dist
        self._max_angular = max_angular
        self.goal_tolerance = goal_tolerance

    def compute_velocity(
        self,
        pose: tuple[float, float, float],
        waypoints: list[tuple[float, float]],
        max_linear: float,
    ) -> tuple[float, float]:
        x, y, theta = pose
        tx, ty = self._lookahead_point(x, y, waypoints)
        return self.compute_velocity_to_point(pose, (tx, ty), max_linear)

    def compute_velocity_to_point(
        self,
        pose: tuple[float, float, float],
        target: tuple[float, float],
        max_linear: float,
    ) -> tuple[float, float]:
        x, y, theta = pose
        tx, ty = target

        dx = tx - x
        dy = ty - y

        # Transform target into robot frame.
        x_r = math.cos(theta) * dx + math.sin(theta) * dy
        y_r = -math.sin(theta) * dx + math.cos(theta) * dy
        dist = math.hypot(x_r, y_r)

        if dist < 1e-6:
            return 0.0, 0.0

        curvature = 2.0 * y_r / (dist * dist)

        forward_scale = max(0.0, x_r / dist)
        curvature_scale = 1.0 + abs(curvature) * self._lookahead
        linear = max_linear * forward_scale / curvature_scale

        if linear <= 1e-6:
            angular = self._max_angular * math.tanh(y_r / max(self._lookahead, 1e-6))
            return 0.0, angular

        angular = curvature * linear

        if abs(angular) > self._max_angular:
            angular = math.copysign(self._max_angular, angular)
            linear = min(linear, abs(angular / curvature)) if abs(curvature) > 1e-6 else linear

        return linear, angular

    def _lookahead_point(
        self,
        x: float,
        y: float,
        waypoints: list[tuple[float, float]],
    ) -> tuple[float, float]:
        """
        Return the first ordered waypoint beyond the lookahead distance.
        """
        for wx, wy in waypoints:
            if math.hypot(wx - x, wy - y) >= self._lookahead:
                return wx, wy

        return waypoints[-1]

    def CurrentTargetReached(self, target_x, target_y, x, y):
        dist_to_target = np.hypot(target_x - x, target_y - y)
        return dist_to_target < self.goal_tolerance


# =============================================================================
# APF
# =============================================================================

class APFPlanner:
    """
    Artificial Potential Fields planner — single-goal variant.

    Combines attractive force toward one goal point with repulsive forces from
    a world-frame obstacle cloud supplied by the caller.
    """

    def __init__(
        self,
        max_linear: float = 200.0,
        max_angular: float = 2.0,
        repulsion_gain: float = 500.0,
        repulsion_range: float = 300.0,
        goal_tolerance: float = 20.0,
        attraction_gain: float = 1.0,
        heading_gain: float = 2.0,
        force_ema_alpha: float = 0.35,
        robot_front_mm: float = 400.0,
        robot_rear_mm: float = 100.0,
        robot_half_width_mm: float = 200.0,
    ) -> None:
        self._max_linear = max_linear
        self._max_angular = max_angular
        self._rep_gain = repulsion_gain
        self._rep_range = repulsion_range
        self.goal_tolerance = goal_tolerance
        self._attr_gain = attraction_gain
        self._heading_gain = heading_gain
        self._force_alpha = float(force_ema_alpha)
        self._front_mm = float(robot_front_mm)
        self._rear_mm = float(robot_rear_mm)
        self._half_width = float(robot_half_width_mm)

        self._desired_heading: float | None = None
        self._committed_left: bool | None = None

    def navigate_to_goal(
        self,
        pose: tuple[float, float, float],
        goal: tuple[float, float],
        obstacles: np.ndarray,
    ) -> tuple[float, float]:
        """
        Return (linear_mm_s, angular_rad_s) toward goal, avoiding obstacles.
        """
        px, py, theta = pose
        gx, gy = goal

        dx = gx - px
        dy = gy - py
        dist_to_goal = math.hypot(dx, dy)

        if dist_to_goal <= self.goal_tolerance:
            return 0.0, 0.0

        attr_scale = self._attr_gain * min(dist_to_goal, self._rep_range)
        attr_x = attr_scale * dx / dist_to_goal
        attr_y = attr_scale * dy / dist_to_goal

        rep_x = 0.0
        rep_y = 0.0
        tan_x = 0.0
        tan_y = 0.0
        obstacle_scale = 1.0

        obs = np.asarray(obstacles)

        if obs.ndim != 2 or obs.shape[0] == 0:
            self._committed_left = None

        if obs.ndim == 2 and obs.shape[0] > 0:
            centers = obs[:, :2]
            radii = obs[:, 2] if obs.shape[1] >= 3 else np.zeros(obs.shape[0], dtype=float)

            cos_th = math.cos(theta)
            sin_th = math.sin(theta)

            to_obs_x = centers[:, 0] - px
            to_obs_y = centers[:, 1] - py

            local_fwd = to_obs_x * cos_th + to_obs_y * sin_th
            local_right = to_obs_x * sin_th - to_obs_y * cos_th

            clamped_fwd = np.clip(local_fwd, -self._rear_mm, self._front_mm)
            clamped_right = np.clip(local_right, -self._half_width, self._half_width)

            nearest_x = px + clamped_right * sin_th + clamped_fwd * cos_th
            nearest_y = py - clamped_right * cos_th + clamped_fwd * sin_th

            fx_raw = nearest_x - centers[:, 0]
            fy_raw = nearest_y - centers[:, 1]

            rect_dists = np.maximum(np.sqrt(fx_raw * fx_raw + fy_raw * fy_raw), 1e-6)
            boundary_dists = np.maximum(rect_dists - radii, 0.0)

            in_range = boundary_dists < self._rep_range

            if np.any(in_range):
                d = np.maximum(boundary_dists[in_range], 1e-6)
                proximity = 1.0 - (d / self._rep_range)
                mag = self._rep_gain * proximity * proximity

                ux = fx_raw[in_range] / rect_dists[in_range]
                uy = fy_raw[in_range] / rect_dists[in_range]

                rep_x = float(np.sum(mag * ux))
                rep_y = float(np.sum(mag * uy))

                left_tx = -uy
                left_ty = ux
                right_tx = uy
                right_ty = -ux

                if self._committed_left is None:
                    goal_ux = dx / dist_to_goal
                    goal_uy = dy / dist_to_goal

                    left_score = float(np.sum(left_tx * goal_ux + left_ty * goal_uy))
                    right_score = float(np.sum(right_tx * goal_ux + right_ty * goal_uy))

                    self._committed_left = left_score >= right_score

                tangent_mag = 0.25 * self._rep_gain * proximity * proximity

                if self._committed_left:
                    tan_x = float(np.sum(tangent_mag * left_tx))
                    tan_y = float(np.sum(tangent_mag * left_ty))
                else:
                    tan_x = float(np.sum(tangent_mag * right_tx))
                    tan_y = float(np.sum(tangent_mag * right_ty))

                fwd_mask = local_fwd[in_range] > 0

                if np.any(fwd_mask):
                    nearest_fwd_boundary = float(np.min(boundary_dists[in_range][fwd_mask]))
                    obstacle_scale = max(0.15, min(1.0, nearest_fwd_boundary / self._rep_range))
            else:
                self._committed_left = None

        force_x = attr_x + tan_x
        force_y = attr_y + tan_y

        if math.hypot(force_x, force_y) < 1e-6:
            return 0.0, 0.0

        desired_raw = math.atan2(force_y, force_x)

        if self._desired_heading is None:
            self._desired_heading = desired_raw
        else:
            delta = (desired_raw - self._desired_heading + math.pi) % (2.0 * math.pi) - math.pi
            self._desired_heading = (
                self._desired_heading + self._force_alpha * delta + math.pi
            ) % (2.0 * math.pi) - math.pi

        desired = self._desired_heading
        ang_err = (desired - theta + math.pi) % (2.0 * math.pi) - math.pi

        forward_scale = max(0.0, math.cos(ang_err))
        goal_scale = min(1.0, dist_to_goal / max(2.0 * self.goal_tolerance, 1.0))

        linear = self._max_linear * forward_scale * goal_scale * obstacle_scale

        angular = self._heading_gain * ang_err
        angular = max(-self._max_angular, min(self._max_angular, angular))

        return linear, angular


# =============================================================================
# Leashed APF
# =============================================================================

class LeashedAPFPlanner:
    """
    Local goal-seeking planner using a leashed virtual target.
    """

    def __init__(
        self,
        *,
        max_linear: float = 200.0,
        max_angular: float = 1.0,
        target_speed: float = 200.0,
        repulsion_gain: float = 800.0,
        repulsion_range: float = 700.0,
        goal_tolerance: float = 20.0,
        attraction_gain: float = 1.0,
        force_ema_alpha: float = 0.35,
        leash_length_mm: float = 400.0,
        leash_half_angle_deg: float = 60.0,
        inflation_margin_mm: float = 200.0,
        tracker_lookahead_mm: float = 150.0,
    ) -> None:
        self._max_linear = float(max_linear)
        self._max_angular = float(max_angular)
        self._target_speed = float(target_speed)
        self._rep_gain = float(repulsion_gain)
        self._rep_range = float(repulsion_range)
        self.goal_tolerance = float(goal_tolerance)
        self._attr_gain = float(attraction_gain)
        self._force_alpha = float(force_ema_alpha)
        self._leash_length = float(leash_length_mm)
        self._leash_half_angle = math.radians(float(leash_half_angle_deg))
        self._inflation_margin = float(inflation_margin_mm)

        self._tracker = PurePursuitPlanner(
            lookahead_dist=float(tracker_lookahead_mm),
            max_angular=float(max_angular),
            goal_tolerance=float(goal_tolerance),
        )

        self._virtual_target: tuple[float, float] | None = None
        self._force_ema: np.ndarray | None = None

    def reset(self) -> None:
        self._virtual_target = None
        self._force_ema = None

    def get_virtual_target(self) -> tuple[float, float] | None:
        return self._virtual_target

    def navigate_to_goal(
        self,
        pose: tuple[float, float, float],
        goal: tuple[float, float],
        obstacles: np.ndarray,
        dt: float,
    ) -> tuple[float, float]:
        px, py, _theta = pose
        gx, gy = goal

        if math.hypot(gx - px, gy - py) <= self.goal_tolerance:
            self._virtual_target = goal
            return 0.0, 0.0

        target = self.update_virtual_target(pose, goal, obstacles, dt)

        return self._tracker.compute_velocity_to_point(pose, target, self._max_linear)

    def update_virtual_target(
        self,
        pose: tuple[float, float, float],
        goal: tuple[float, float],
        obstacles: np.ndarray,
        dt: float,
    ) -> tuple[float, float]:
        px, py, theta = pose
        gx, gy = goal

        dt = max(1e-3, float(dt))

        if self._virtual_target is None:
            self._virtual_target = (float(px), float(py))

        vx, vy = self._virtual_target
        raw_force = self._compute_force((vx, vy), goal, obstacles)

        if self._force_ema is None:
            filtered_force = raw_force
        else:
            filtered_force = (
                (1.0 - self._force_alpha) * self._force_ema
                + self._force_alpha * raw_force
            )

        self._force_ema = filtered_force

        force_norm = float(np.linalg.norm(filtered_force))
        goal_dist = math.hypot(gx - vx, gy - vy)

        if goal_dist <= self.goal_tolerance:
            candidate = (gx, gy)
        elif force_norm <= 1e-6:
            candidate = (vx, vy)
        else:
            direction = filtered_force / force_norm
            step_speed = min(self._target_speed, goal_dist / dt)

            candidate = (
                vx + float(direction[0]) * step_speed * dt,
                vy + float(direction[1]) * step_speed * dt,
            )

        constrained = self._apply_leash(candidate, pose)
        self._virtual_target = constrained

        return constrained

    def _compute_force(
        self,
        point: tuple[float, float],
        goal: tuple[float, float],
        obstacles: np.ndarray,
    ) -> np.ndarray:
        vx, vy = point
        gx, gy = goal

        goal_vec = np.array([gx - vx, gy - vy], dtype=float)
        goal_dist = float(np.linalg.norm(goal_vec))

        if goal_dist > 1e-6:
            attr = self._attr_gain * (goal_vec / goal_dist)
        else:
            attr = np.zeros(2, dtype=float)

        rep = np.zeros(2, dtype=float)
        nearest_obs: tuple[float, float, float] | None = None
        nearest_boundary = float("inf")

        obs = np.asarray(obstacles, dtype=float)

        if obs.ndim == 2 and obs.shape[0] > 0:
            for row in obs:
                ox = float(row[0])
                oy = float(row[1])
                radius = float(row[2]) if row.shape[0] >= 3 else 0.0
                eff_radius = radius + self._inflation_margin

                away = np.array([vx - ox, vy - oy], dtype=float)
                center_dist = float(np.linalg.norm(away))

                if center_dist <= 1e-6:
                    away = np.array([0.0, 1.0], dtype=float)
                    center_dist = 1e-6

                boundary = center_dist - eff_radius

                if boundary >= self._rep_range:
                    continue

                if boundary < nearest_boundary:
                    nearest_boundary = boundary
                    nearest_obs = (ox, oy, eff_radius)

                direction = away / center_dist
                clearance = max(boundary, 1.0)

                rep_mag = self._rep_gain * max(
                    0.0,
                    (1.0 / clearance) - (1.0 / self._rep_range),
                )

                rep += direction * rep_mag

        total = attr + rep

        if nearest_obs is not None and np.linalg.norm(total) < 0.25:
            ox, oy, _eff_radius = nearest_obs
            away = np.array([vx - ox, vy - oy], dtype=float)
            norm = float(np.linalg.norm(away))

            if norm > 1e-6:
                away /= norm
                tangent = np.array([-away[1], away[0]], dtype=float)
                total += 0.15 * self._attr_gain * tangent

        return total

    def _apply_leash(
        self,
        target: tuple[float, float],
        pose: tuple[float, float, float],
    ) -> tuple[float, float]:
        px, py, theta = pose
        tx, ty = target

        dx = tx - px
        dy = ty - py

        local_x = math.cos(theta) * dx + math.sin(theta) * dy
        local_y = -math.sin(theta) * dx + math.cos(theta) * dy

        distance = math.hypot(local_x, local_y)
        angle = math.atan2(local_y, local_x)

        clipped_distance = min(distance, self._leash_length)
        clipped_angle = max(
            -self._leash_half_angle,
            min(self._leash_half_angle, angle),
        )

        local_x = clipped_distance * math.cos(clipped_angle)
        local_y = clipped_distance * math.sin(clipped_angle)

        world_x = px + math.cos(theta) * local_x - math.sin(theta) * local_y
        world_y = py + math.sin(theta) * local_x + math.cos(theta) * local_y

        return float(world_x), float(world_y)


# =============================================================================
# Pure Pursuit With Lane-Switch Obstacle Avoidance
# =============================================================================

class PurePursuitPlannerWithAvoidance(PathPlanner):
    def __init__(
        self,
        lookahead_distance: float = 100.0,
        max_linear_speed: float = 130.0,
        max_angular_speed: float = 1.0,
        goal_tolerance: float = 20.0,
        obstacles_range: float = 400.0,
        view_angle: float = np.pi / 2,
        safe_dist: float = 150.0,
        avoidance_delay: int = 200,
        offset: float = 120.0,
        x_L: float = 0.0,
        lane_width: float = 500.0,
        alpha_Ld: float = 0.8,
        obstacle_avoidance: bool = True,
    ):
        self.Ld = lookahead_distance
        self.raw_LD = lookahead_distance

        self.v_max = max_linear_speed
        self.w_max = max_angular_speed

        self.goal_tolerance = goal_tolerance
        self.obstacles_range = obstacles_range
        self.view_angle = view_angle
        self.safe_dist = safe_dist
        self.alpha_Ld = alpha_Ld
        self.obstacle_avoidance = obstacle_avoidance
        self.x_L = x_L
        self.lane_width = lane_width
        self.offset = offset

        self.avoidance_active = False
        self.avoidance_counter = 0
        self.avoidance_delay = avoidance_delay

        # Starting lane.
        self.current_lane = "Left"

        self.raw_path: list[tuple[float, float]] = []
        self.remaining_path: list[tuple[float, float]] = []

    def set_path(self, path: list[tuple[float, float]]):
        self.raw_path = path.copy()

        if self.current_lane == "Center":
            self.remaining_path = path.copy()

        elif self.current_lane == "Left":
            self.remaining_path = []
            for x_, y_ in self.raw_path:
                self.remaining_path.append((x_ - self.offset, y_))

        elif self.current_lane == "Right":
            self.remaining_path = []
            for x_, y_ in self.raw_path:
                self.remaining_path.append((x_ + self.offset, y_))

    def _advance_remaining_path(
        self,
        x: float,
        y: float,
    ) -> list[tuple[float, float]]:
        while len(self.remaining_path) > 1:
            next_x_mm, next_y_mm = self.remaining_path[0]

            if np.hypot(x - next_x_mm, y - next_y_mm) > self.Ld:
                break

            self.remaining_path.pop(0)
            self.raw_path.pop(0)

            if self.avoidance_active:
                self.avoidance_active = False
                self.Ld = self.raw_LD
                self.avoidance_counter = 0

        return self.remaining_path

    def _lookahead_point(self, path, x, y):
        path = np.array(path)
        position = np.array([x, y])

        for i in range(len(path)):
            dist = np.linalg.norm(path[i, :] - position)

            if dist >= self.Ld:
                return path[i]

        return path[-1]

    def TargetReached(self, path, x, y):
        if self.avoidance_active:
            return False

        goal_x, goal_y = path[0]
        dist_to_goal = np.hypot(goal_x - x, goal_y - y)

        return dist_to_goal < self.goal_tolerance

    def gen_obstacle_waypoint(self, pose, obstacles_r):
        x, y, theta = pose

        if len(obstacles_r) > 0:
            # Lidar orientation is 180 degrees rotated from robot forward.
            obstacles_r = (
                np.array(
                    [
                        [np.cos(np.pi), -np.sin(np.pi)],
                        [np.sin(np.pi), np.cos(np.pi)],
                    ]
                )
                @ obstacles_r.T
            ).T

            # Only consider obstacles inside the front view angle.
            angles = np.arctan2(obstacles_r[:, 1], obstacles_r[:, 0])
            obstacles_r = obstacles_r[np.abs(angles) <= self.view_angle, :]

            # Filter obstacles outside detecting range.
            dists = np.linalg.norm(obstacles_r, axis=1)
            obstacles_r = obstacles_r[dists < self.obstacles_range]

            if len(obstacles_r) == 0:
                return

            # Transform obstacles from robot frame to world frame.
            obstacles = (
                np.array(
                    [
                        [np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)],
                    ]
                )
                @ obstacles_r.T
            ).T + np.array([[x, y]])

            # Filter by lane width.
            lane_mask = np.abs(obstacles[:, 0] - self.x_L) < self.lane_width
            obstacles_r = obstacles_r[lane_mask, :]
            obstacles = obstacles[lane_mask, :]

            if len(obstacles_r) == 0:
                return

            # Modify waypoint if current waypoint is too close to obstacles.
            if len(self.remaining_path) > 1:
                waypoint = np.float64([self.remaining_path[0]])
                waypoint_dists = np.sqrt(np.sum((waypoint - obstacles) ** 2, axis=1))

                if np.any(waypoint_dists < self.safe_dist):
                    new_wp = (
                        (self.remaining_path[0][0] + self.remaining_path[1][0]) / 2,
                        (self.remaining_path[0][1] + self.remaining_path[1][1]) / 2,
                    )

                    self.remaining_path[0] = new_wp
                    self.raw_path[0] = new_wp

            if len(obstacles_r) > 0 and self.avoidance_counter <= 0:
                dists = np.linalg.norm(obstacles_r, axis=1)
                arg_dist = np.argmin(dists)

                closest_pt = obstacles[arg_dist, :]

                change_lane = False

                if (
                    closest_pt[0] < self.x_L
                    and self.current_lane != "Right"
                ) or (
                    closest_pt[0] > self.x_L
                    and self.current_lane != "Left"
                ):
                    change_lane = True

                    # Reduce lookahead to track added waypoints more precisely.
                    self.Ld = self.raw_LD * self.alpha_Ld

                    self.avoidance_counter = self.avoidance_delay
                    self.avoidance_active = True

                if change_lane:
                    self.remaining_path = []

                    for x_, y_ in self.raw_path:
                        if closest_pt[0] < self.x_L:
                            self.remaining_path.append((x_ + self.offset, y_))
                            self.current_lane = "Right"
                        else:
                            self.remaining_path.append((x_ - self.offset, y_))
                            self.current_lane = "Left"

                    print("Change Lane!!! SEES OBJECT Current lane is:", self.current_lane)

                    if np.hypot(x - closest_pt[0], y - closest_pt[1]) < (
                        self.safe_dist + self.obstacles_range
                    ) / 2:
                        print("Too Close!!!")

                        if self.current_lane == "Right":
                            self.remaining_path.insert(
                                0,
                                (x + self.offset, y + self.offset / 2),
                            )
                            self.raw_path.insert(
                                0,
                                (x + self.offset, y + self.offset),
                            )

                        elif self.current_lane == "Left":
                            self.remaining_path.insert(
                                0,
                                (x - self.offset, y + self.offset / 2),
                            )
                            self.raw_path.insert(
                                0,
                                (x - self.offset, y + self.offset),
                            )

        if self.avoidance_counter > 0:
            self.avoidance_counter -= 1

    def compute_velocity(self, pose, obstacles_r: np.ndarray):
        """
        Compute velocity command.

        This version adds pivot-turn behavior:

        If the lookahead target is far left/right, the robot sets linear speed
        to zero and turns in place. That should cause one wheel to go forward
        and the other wheel to go backward.

        Once the target is mostly ahead again, it returns to normal pure pursuit.
        """

        x, y, theta = pose

        self._advance_remaining_path(x, y)

        if len(self.remaining_path) == 0:
            return 0.0, 0.0

        if self.TargetReached(self.remaining_path, x, y):
            return 0.0, 0.0

        if self.obstacle_avoidance:
            self.gen_obstacle_waypoint(pose, obstacles_r)

        if len(self.remaining_path) == 0:
            return 0.0, 0.0

        target = self._lookahead_point(self.remaining_path, x, y)
        tx, ty = target

        dx = tx - x
        dy = ty - y

        # Transform target point into robot frame.
        x_r = np.cos(theta) * dx + np.sin(theta) * dy
        y_r = -np.sin(theta) * dx + np.cos(theta) * dy

        dist = math.hypot(x_r, y_r)

        if dist < 1e-6:
            return 0.0, 0.0

        # ---------------------------------------------------------------------
        # PIVOT-TURN LOGIC
        # ---------------------------------------------------------------------
        # heading_error:
        #   positive = target is to robot's left
        #   negative = target is to robot's right
        #
        # If the heading error is large, we force linear = 0.
        # That makes the robot rotate in place instead of driving in an arc.
        # ---------------------------------------------------------------------

        heading_error = math.atan2(y_r, x_r)

        PIVOT_TURN_ANGLE = math.radians(25.0)

        if abs(heading_error) > PIVOT_TURN_ANGLE:
            angular = self.w_max * math.tanh(heading_error)

            # This is the key line:
            # linear = 0 means one wheel should spin forward and the other backward.
            return 0.0, angular

        # ---------------------------------------------------------------------
        # NORMAL PURE PURSUIT
        # ---------------------------------------------------------------------

        curvature = 2.0 * y_r / (dist * dist)

        forward_scale = max(0.0, x_r / dist)
        curvature_scale = 1.0 + abs(curvature) * self.Ld

        linear = self.v_max * forward_scale / curvature_scale

        if linear <= 1e-6:
            angular = self.w_max * math.tanh(y_r / max(self.Ld, 1e-6))
            return 0.0, angular

        angular = curvature * linear

        if abs(angular) > self.w_max:
            angular = math.copysign(self.w_max, angular)
            linear = min(linear, abs(angular / curvature)) if abs(curvature) > 1e-6 else linear

        return linear, angular

    def motion(self, pose, v, w, dt):
        x, y, theta = pose

        pose[0] += v * np.cos(theta) * dt
        pose[1] += v * np.sin(theta) * dt
        pose[2] += w * dt

        return pose


# =============================================================================
# Helper
# =============================================================================

def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi 
