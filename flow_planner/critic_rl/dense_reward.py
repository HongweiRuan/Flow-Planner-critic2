"""Incremental nuPlan reward components computed directly at every step."""

import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import numpy as np

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.metrics.evaluation_metrics.common.ego_progress_along_expert_route import (
    PerFrameProgressAlongRouteComputer,
)
from nuplan.planning.metrics.evaluation_metrics.common.no_ego_at_fault_collisions import (
    find_new_collisions,
)
from nuplan.planning.metrics.evaluation_metrics.common.speed_limit_compliance import (
    SpeedLimitViolationExtractor,
)
from nuplan.planning.metrics.evaluation_metrics.common.time_to_collision_within_bound import (
    _compute_time_to_collision_at_timestamp,
)
from nuplan.planning.metrics.utils.collision_utils import CollisionType, VRU_types, object_types
from nuplan.planning.metrics.utils.route_extractor import (
    extract_corners_route,
    get_common_or_connected_route_objs_of_corners,
    get_current_route_objects,
    get_distance_of_closest_baseline_point_to_its_start,
    get_route,
    get_route_baseline_roadblock_linkedlist,
    get_route_simplified,
)
from nuplan.planning.metrics.utils.state_extractors import extract_ego_center
from nuplan.planning.simulation.observation.idm.utils import is_agent_ahead, is_agent_behind

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DenseRewardConfig:
    """Weights follow the official closed-loop weighted-average score."""

    progress_weight: float = 5.0
    ttc_weight: float = 5.0
    speed_weight: float = 4.0
    comfort_weight: float = 2.0
    weighted_normalizer: float = 16.0
    collision_scale: float = 1.0
    drivable_scale: float = 1.0
    direction_scale: float = 1.0
    making_progress_scale: float = 1.0


class NuPlanStepRewardComputer:
    """Stateful O(1)-history reward features using official nuPlan helpers."""

    def __init__(
        self,
        metrics_engine: Any,
        normalization_steps: Optional[int] = None,
        config: Optional[DenseRewardConfig] = None,
    ) -> None:
        self.metrics = {metric.name: metric for metric in metrics_engine.metrics}
        self.normalization_steps = normalization_steps
        self.config = config or DenseRewardConfig()
        self.reset()

    def reset(self) -> None:
        self._initialized = False
        self._expected_steps = 1
        self._previous_state: Optional[Any] = None
        self._progress_computer: Optional[PerFrameProgressAlongRouteComputer] = None
        self._expert_progress = 2.0
        self._collided_track_ids: Set[str] = set()
        self._collision_counts: Dict[str, int] = defaultdict(int)
        self._collision_score = 1.0
        self._far_from_drivable_area = False
        self._drivable_score = 1.0
        self._direction_route_id: Optional[str] = None
        self._direction_distance: Optional[float] = None
        self._direction_window: Deque[Tuple[int, float]] = deque()
        self._direction_min_progress = 0.0
        self._direction_score = 1.0
        self._total_progress = 0.0
        self._making_progress_score = 1.0

    @staticmethod
    def _point(ego_state: Any) -> Point2D:
        return Point2D(float(ego_state.center.x), float(ego_state.center.y))

    def _initialize(self, history: Any, scenario: Any) -> None:
        data = history.data
        current_state = data[-1].ego_state
        self._previous_state = data[-2].ego_state if len(data) >= 2 else current_state
        scenario_steps = max(1, int(scenario.get_number_of_iterations()) - 1)
        if self.normalization_steps is not None:
            scenario_steps = min(scenario_steps, int(self.normalization_steps))
        self._expected_steps = max(1, scenario_steps)

        try:
            expert_states = list(scenario.get_expert_ego_trajectory())
            expert_poses = extract_ego_center(expert_states)
            expert_route = get_route(history.map_api, expert_poses)
            simplified = get_route_simplified(expert_route)
            if simplified:
                linked_route = get_route_baseline_roadblock_linkedlist(history.map_api, simplified)
                expert_computer = PerFrameProgressAlongRouteComputer(linked_route)
                self._expert_progress = max(2.0, abs(float(np.sum(expert_computer(expert_poses)))))
                self._progress_computer = PerFrameProgressAlongRouteComputer(linked_route)
                self._progress_computer([self._point(self._previous_state)])
        except Exception as error:
            logger.warning("Falling back to heading-projected progress: %s", error)
            self._progress_computer = None

        initial_route = get_current_route_objects(history.map_api, self._point(self._previous_state))
        if initial_route:
            self._direction_route_id = initial_route[0].id
            self._direction_distance = get_distance_of_closest_baseline_point_to_its_start(
                initial_route[0].baseline_path, self._point(self._previous_state)
            )
        self._initialized = True

    def _progress_delta(self, current_state: Any) -> float:
        assert self._previous_state is not None
        current_pose = self._point(current_state)
        if self._progress_computer is not None:
            computer = self._progress_computer
            try:
                if computer.curr_roadblock_pair.road_block.contains_point(current_pose):
                    distance = get_distance_of_closest_baseline_point_to_its_start(
                        computer.curr_roadblock_pair.base_line, current_pose
                    )
                    progress = float(distance - computer.prev_distance_to_start)
                    computer.prev_distance_to_start = distance
                    return progress
                return float(computer.get_multi_block_progress(current_pose))
            except Exception as error:
                logger.debug("Route progress fallback at one step: %s", error)

        previous = self._previous_state.center
        dx = float(current_state.center.x - previous.x)
        dy = float(current_state.center.y - previous.y)
        return float(dx * np.cos(previous.heading) + dy * np.sin(previous.heading))

    @staticmethod
    def _route_context(history: Any, ego_state: Any) -> Tuple[List[Any], Any, Any]:
        center_route = get_current_route_objects(history.map_api, Point2D(float(ego_state.center.x), float(ego_state.center.y)))
        corners_route_list = extract_corners_route(history.map_api, [ego_state.car_footprint])
        corners_route = corners_route_list[0] if corners_route_list else None
        common_route = (
            get_common_or_connected_route_objs_of_corners([corners_route])[0]
            if corners_route is not None
            else None
        )
        return center_route, corners_route, common_route

    def _collision_delta(
        self, ego_state: Any, observation: Any, common_route: Any
    ) -> Tuple[float, bool]:
        if not hasattr(observation, "tracked_objects"):
            return 0.0, False
        self._collided_track_ids, collisions = find_new_collisions(
            ego_state, observation, self._collided_track_ids
        )
        at_fault = False
        for collision in collisions.values():
            collision_is_fault = collision.collision_type in (
                CollisionType.ACTIVE_FRONT_COLLISION,
                CollisionType.STOPPED_TRACK_COLLISION,
            ) or (
                collision.collision_type == CollisionType.ACTIVE_LATERAL_COLLISION
                and not common_route
            )
            if not collision_is_fault:
                continue
            at_fault = True
            track_type = collision.tracked_object_type
            if track_type in VRU_types:
                self._collision_counts["vru"] += 1
            elif track_type == TrackedObjectType.VEHICLE:
                self._collision_counts["vehicle"] += 1
            elif track_type in object_types:
                self._collision_counts["object"] += 1

        metric = self.metrics.get("no_ego_at_fault_collisions")
        thresholds = {
            "vru": int(getattr(metric, "_max_violation_threshold_vru", 0)),
            "vehicle": int(getattr(metric, "_max_violation_threshold_vehicle", 0)),
            "object": int(getattr(metric, "_max_violation_threshold_object", 1)),
        }
        score = 1.0
        for category, threshold in thresholds.items():
            score *= max(0.0, 1.0 - self._collision_counts[category] / (threshold + 1))
        delta = score - self._collision_score
        self._collision_score = score
        return float(delta), at_fault

    def _drivable_delta(
        self,
        history: Any,
        ego_state: Any,
        center_route: List[Any],
        corners_route: Any,
    ) -> float:
        previous = self._drivable_score
        metric = self.metrics.get("drivable_area_compliance")
        if metric is not None and corners_route is not None:
            _, self._far_from_drivable_area = metric.compute_violation_for_iteration(
                history.map_api,
                list(ego_state.car_footprint.all_corners()),
                corners_route,
                center_route,
                self._far_from_drivable_area,
            )
        else:
            outside = any(
                not history.map_api.is_in_layer(corner, SemanticMapLayer.DRIVABLE_AREA)
                for corner in ego_state.car_footprint.all_corners()
            )
            self._far_from_drivable_area = self._far_from_drivable_area or outside
        self._drivable_score = float(not self._far_from_drivable_area)
        return self._drivable_score - previous

    def _direction_delta(
        self, ego_state: Any, center_route: List[Any]
    ) -> float:
        timestamp = int(ego_state.time_point.time_us)
        progress = 0.0
        if center_route:
            route = center_route[0]
            distance = get_distance_of_closest_baseline_point_to_its_start(
                route.baseline_path, self._point(ego_state)
            )
            if route.id == self._direction_route_id and self._direction_distance is not None:
                progress = float(distance - self._direction_distance)
            self._direction_route_id = route.id
            self._direction_distance = distance
        else:
            self._direction_route_id = None
            self._direction_distance = None

        metric = self.metrics.get("driving_direction_compliance")
        horizon_us = int(float(getattr(metric, "_time_horizon", 1.0)) * 1e6)
        self._direction_window.append((timestamp, progress))
        while self._direction_window and timestamp - self._direction_window[0][0] > horizon_us:
            self._direction_window.popleft()
        window_progress = float(sum(value for _, value in self._direction_window))
        self._direction_min_progress = min(self._direction_min_progress, window_progress)
        negative_progress = abs(min(0.0, self._direction_min_progress))
        compliant = float(getattr(metric, "_driving_direction_compliance_threshold", 2.0))
        violation = float(getattr(metric, "_driving_direction_violation_threshold", 6.0))
        score = 1.0 if negative_progress < compliant else 0.5 if negative_progress < violation else 0.0
        delta = score - self._direction_score
        self._direction_score = score
        return float(delta)

    def _making_progress_delta(self) -> float:
        """Telescoping delta for the ego_is_making_progress multiplier.

        Official metric passes when accumulated ego progress divided by expert
        progress reaches a minimum ratio; below that the whole episode score is
        multiplied by zero. Mirrors the other multipliers: the score starts at 1
        and only drops if the ratio never clears the threshold, so the per-step
        deltas telescope to (final_pass - 1).
        """
        metric = self.metrics.get("ego_is_making_progress")
        threshold = float(getattr(metric, "_min_progress_threshold", 0.2))
        ratio = self._total_progress / max(self._expert_progress, 1e-6)
        score = 1.0 if ratio >= threshold else 0.0
        delta = score - self._making_progress_score
        self._making_progress_score = score
        return float(delta)

    def _speed_quality(self, ego_state: Any, center_route: List[Any]) -> float:
        if not center_route:
            return 1.0
        violation = SpeedLimitViolationExtractor._get_speed_limit_violation(
            ego_state, int(ego_state.time_point.time_us), center_route
        )
        if violation is None:
            return 1.0
        metric = self.metrics.get("speed_limit_compliance")
        threshold = max(float(getattr(metric, "_max_overspeed_value_threshold", 2.23)), 1e-3)
        overspeed = float(violation.violation_depths[0])
        return float(np.clip(1.0 - overspeed / threshold, 0.0, 1.0))

    def _ttc_quality(
        self,
        history: Any,
        ego_state: Any,
        observation: Any,
        common_route: Any,
        at_fault_collision: bool,
    ) -> float:
        if not hasattr(observation, "tracked_objects"):
            return 1.0
        allow_lateral = not common_route or history.map_api.is_in_layer(
            ego_state.rear_axle, SemanticMapLayer.INTERSECTION
        )
        tracks = [
            track
            for track in observation.tracked_objects
            if track.track_token not in self._collided_track_ids
            and (
                is_agent_ahead(ego_state.rear_axle, track.center)
                or (allow_lateral and not is_agent_behind(ego_state.rear_axle, track.center))
            )
        ]
        poses = np.asarray(
            [
                [float(track.center.x), float(track.center.y), float(track.center.heading)]
                for track in tracks
            ],
            dtype=np.float64,
        ).reshape((-1, 3))
        speeds = np.asarray(
            [track.velocity.magnitude() if isinstance(track, Agent) else 0.0 for track in tracks],
            dtype=np.float64,
        )
        boxes = np.asarray([track.box for track in tracks], dtype=object)
        metric = self.metrics.get("time_to_collision_within_bound")
        time_step = float(getattr(metric, "_time_step_size", 0.1))
        time_horizon = float(getattr(metric, "_time_horizon", 3.0))
        least_ttc = max(float(getattr(metric, "_least_min_ttc", 0.95)), 1e-3)
        timestamp = int(ego_state.time_point.time_us)
        ttc = _compute_time_to_collision_at_timestamp(
            timestamp=timestamp,
            ego_state=ego_state,
            ego_speed=np.asarray(ego_state.dynamic_car_state.speed),
            tracks_poses=poses,
            tracks_speed=speeds,
            tracks_boxes=boxes,
            timestamps_at_fault_collisions=[timestamp] if at_fault_collision else [],
            time_step_size=time_step,
            time_horizon=time_horizon,
            stopped_speed_threshold=5e-3,
        )
        return 1.0 if ttc is None else float(np.clip(ttc / least_ttc, 0.0, 1.0))

    @staticmethod
    def _bound_quality(value: float, lower: float, upper: float) -> float:
        if lower <= value <= upper:
            return 1.0
        scale = max(abs(lower), abs(upper), 1e-3)
        excess = lower - value if value < lower else value - upper
        return float(np.clip(1.0 - excess / scale, 0.0, 1.0))

    def _comfort_quality(self, ego_state: Any) -> float:
        assert self._previous_state is not None
        current_dynamic = ego_state.dynamic_car_state
        previous_dynamic = self._previous_state.dynamic_car_state
        dt = max(
            (ego_state.time_point.time_us - self._previous_state.time_point.time_us) * 1e-6,
            1e-3,
        )
        acceleration = current_dynamic.center_acceleration_2d
        previous_acceleration = previous_dynamic.center_acceleration_2d
        jerk_x = float((acceleration.x - previous_acceleration.x) / dt)
        jerk_y = float((acceleration.y - previous_acceleration.y) / dt)
        values = {
            "ego_jerk": float(np.hypot(jerk_x, jerk_y)),
            "ego_lat_acceleration": float(acceleration.y),
            "ego_lon_acceleration": float(acceleration.x),
            "ego_lon_jerk": jerk_x,
            "ego_yaw_acceleration": float(current_dynamic.angular_acceleration),
            "ego_yaw_rate": float(current_dynamic.angular_velocity),
        }
        qualities: List[float] = []
        for name, value in values.items():
            metric = self.metrics.get(name)
            if name == "ego_lon_acceleration":
                qualities.append(
                    self._bound_quality(
                        value,
                        float(getattr(metric, "_min_lon_accel", -4.05)),
                        float(getattr(metric, "_max_lon_accel", 2.40)),
                    )
                )
                continue
            threshold_name = {
                "ego_jerk": "_max_abs_mag_jerk",
                "ego_lat_acceleration": "_max_abs_lat_accel",
                "ego_lon_jerk": "_max_abs_lon_jerk",
                "ego_yaw_acceleration": "_max_abs_yaw_accel",
                "ego_yaw_rate": "_max_abs_yaw_rate",
            }[name]
            fallback = {
                "ego_jerk": 8.37,
                "ego_lat_acceleration": 4.89,
                "ego_lon_jerk": 4.13,
                "ego_yaw_acceleration": 1.93,
                "ego_yaw_rate": 0.95,
            }[name]
            threshold = max(float(getattr(metric, threshold_name, fallback)), 1e-3)
            qualities.append(self._bound_quality(value, -threshold, threshold))
        return float(min(qualities)) if qualities else 1.0

    def compute(self, history: Any, scenario: Any) -> Dict[str, float]:
        """Return an explicit reward contribution for every official component."""
        if not self._initialized:
            self._initialize(history, scenario)
        sample = history.data[-1]
        ego_state = sample.ego_state
        center_route, corners_route, common_route = self._route_context(history, ego_state)
        collision_delta, at_fault = self._collision_delta(
            ego_state, sample.observation, common_route
        )
        progress_delta = self._progress_delta(ego_state)
        self._total_progress += progress_delta
        making_progress_delta = self._making_progress_delta()
        ttc_quality = self._ttc_quality(
            history, ego_state, sample.observation, common_route, at_fault
        )
        speed_quality = self._speed_quality(ego_state, center_route)
        comfort_quality = self._comfort_quality(ego_state)
        drivable_delta = self._drivable_delta(
            history, ego_state, center_route, corners_route
        )
        direction_delta = self._direction_delta(ego_state, center_route)

        cfg = self.config
        normalizer = max(cfg.weighted_normalizer, 1e-6)
        components = {
            "progress": cfg.progress_weight
            / normalizer
            * float(np.clip(progress_delta / max(self._expert_progress, 1e-6), -1.0, 1.0)),
            "time_to_collision": cfg.ttc_weight
            / normalizer
            * float(ttc_quality / self._expected_steps),
            "speed_limit": cfg.speed_weight
            / normalizer
            * float(speed_quality / self._expected_steps),
            "comfort": cfg.comfort_weight
            / normalizer
            * float(comfort_quality / self._expected_steps),
            "making_progress": cfg.making_progress_scale * float(making_progress_delta),
            "collision": cfg.collision_scale * float(collision_delta),
            "drivable_area": cfg.drivable_scale * float(drivable_delta),
            "driving_direction": cfg.direction_scale * float(direction_delta),
        }
        self._previous_state = ego_state
        return {name: float(value) for name, value in components.items()}
