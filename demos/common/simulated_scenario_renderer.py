#!/usr/bin/env python3
"""Utilities to render a simulated nuPlan rollout to MP4.

This module renders the simulation history saved by the devkit's
SimulationLogCallback. It is reusable by any script that produces a
SimulationLog file.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection

from nuplan.planning.simulation.history.simulation_history import SimulationHistorySample
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.simulation_log import SimulationLog


AGENT_CLASS_COLORS = {
    "vehicle": "#2ca02c",
    "pedestrian": "#ff7f0e",
    "bicycle": "#9467bd",
    "bus": "#1f77b4",
    "generic_object": "#8c564b",
    "ego": "#d62728",
    "unknown": "#17becf",
}


def normalize_category_name(name: str | None) -> str:
    if not name:
        return "unknown"
    normalized = str(name).strip().lower()
    if "pedestrian" in normalized:
        return "pedestrian"
    if "bicycle" in normalized or "bike" in normalized:
        return "bicycle"
    if "bus" in normalized:
        return "bus"
    if "vehicle" in normalized:
        return "vehicle"
    if "generic" in normalized:
        return "generic_object"
    if "ego" in normalized:
        return "ego"
    return "unknown"


def oriented_box_polygon(x: float, y: float, length: float, width: float, yaw: float) -> list[tuple[float, float]]:
    half_l = max(float(length), 0.2) / 2.0
    half_w = max(float(width), 0.2) / 2.0
    cos_yaw = math.cos(float(yaw))
    sin_yaw = math.sin(float(yaw))
    corners = [(half_l, half_w), (half_l, -half_w), (-half_l, -half_w), (-half_l, half_w)]
    polygon = []
    for dx, dy in corners:
        rx = float(x) + dx * cos_yaw - dy * sin_yaw
        ry = float(y) + dx * sin_yaw + dy * cos_yaw
        polygon.append((rx, ry))
    return polygon


def _ego_polygon(ego_state) -> list[tuple[float, float]]:
    box = ego_state.car_footprint.oriented_box
    return oriented_box_polygon(box.center.x, box.center.y, box.length, box.width, box.center.heading)


def _sample_agent_polygons(sample: SimulationHistorySample) -> tuple[list[list[tuple[float, float]]], list[str], list[list[tuple[float, float]]]]:
    agent_polygons: list[list[tuple[float, float]]] = []
    agent_colors: list[str] = []
    velocity_segments: list[list[tuple[float, float]]] = []

    observation = sample.observation
    if not isinstance(observation, DetectionsTracks):
        return agent_polygons, agent_colors, velocity_segments

    for agent in observation.tracked_objects:
        if not hasattr(agent, "box"):
            continue
        box = agent.box
        polygon = oriented_box_polygon(box.center.x, box.center.y, box.length, box.width, box.center.heading)
        agent_polygons.append(polygon)
        category = normalize_category_name(getattr(getattr(agent, "tracked_object_type", None), "name", None))
        agent_colors.append(AGENT_CLASS_COLORS.get(category, AGENT_CLASS_COLORS["unknown"]))
        velocity = getattr(agent, "velocity", None)
        if velocity is not None:
            velocity_segments.append(
                [
                    (float(box.center.x), float(box.center.y)),
                    (float(box.center.x) + float(velocity.x), float(box.center.y) + float(velocity.y)),
                ]
            )

    return agent_polygons, agent_colors, velocity_segments


def render_simulation_log_mp4(
    log_path: Path,
    output_path: Path,
    *,
    fps: int = 10,
    record_seconds: float = 30.0,
    margin_m: float = 40.0,
) -> Path:
    """Render a saved SimulationLog to MP4.

    If record_seconds is <= 0, all available samples are rendered.
    """
    simulation_log = SimulationLog.load_data(log_path)
    scenario = simulation_log.scenario
    history = simulation_log.simulation_history
    samples = list(history.data)
    if not samples:
        raise RuntimeError(f"Simulation log has no samples: {log_path}")

    fps = max(int(fps), 1)
    if record_seconds and record_seconds > 0:
        max_frames = max(1, int(math.ceil(record_seconds * fps)))
        samples = samples[:max_frames]

    mission_goal = scenario.get_mission_goal()
    scenario_name = getattr(scenario, "scenario_name", "unknown")
    scenario_type = getattr(scenario, "scenario_type", "unknown")
    log_name = getattr(scenario, "log_name", "unknown")
    planner_name = getattr(simulation_log.planner, "name", lambda: "unknown")()

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.set_facecolor("#f7f4ee")
    ax.grid(True, color="#ded8cd", linewidth=0.5, alpha=0.5)

    if mission_goal is not None:
        ax.scatter(
            [mission_goal.x],
            [mission_goal.y],
            s=180,
            c="#ffd700",
            marker="*",
            edgecolors="#c9a301",
            linewidths=1.5,
            zorder=7,
        )

    ego_path, = ax.plot([], [], color="#1f77b4", linewidth=2.5, zorder=8)
    ego_box = PolyCollection([], facecolors="#d62728", edgecolors="#7f1d1d", linewidths=1.1, alpha=0.95, zorder=10)
    ax.add_collection(ego_box)
    ego_heading, = ax.plot([], [], color="#9e1f1f", linewidth=2.2, zorder=11)
    agent_boxes = PolyCollection([], facecolors=[], edgecolors="#1f1f1f", linewidths=0.6, alpha=0.7, zorder=8)
    ax.add_collection(agent_boxes)
    agent_velocity = LineCollection([], linewidths=1.0, alpha=0.5, zorder=8)
    ax.add_collection(agent_velocity)
    planned_trajectory, = ax.plot([], [], color="#2f4f4f", linewidth=1.8, linestyle="--", zorder=9)
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=9)

    def update(frame_idx: int):
        sample = samples[frame_idx]
        ego_state = sample.ego_state
        current_x = float(ego_state.center.x)
        current_y = float(ego_state.center.y)

        tail_samples = samples[: frame_idx + 1][-30:]
        ego_path.set_data(
            [float(item.ego_state.center.x) for item in tail_samples],
            [float(item.ego_state.center.y) for item in tail_samples],
        )
        ego_box.set_verts([_ego_polygon(ego_state)])
        ego_heading.set_data(
            [current_x, current_x + math.cos(float(ego_state.center.heading)) * 6.0],
            [current_y, current_y + math.sin(float(ego_state.center.heading)) * 6.0],
        )

        agent_polygons, agent_colors, velocity_segments = _sample_agent_polygons(sample)
        agent_boxes.set_verts(agent_polygons)
        agent_boxes.set_facecolors(agent_colors)
        agent_velocity.set_segments(velocity_segments)

        trajectory = sample.trajectory
        sampled_trajectory = trajectory.get_sampled_trajectory() if trajectory is not None else []
        if sampled_trajectory:
            traj_x = []
            traj_y = []
            for state in sampled_trajectory:
                center = getattr(state, "center", None)
                if center is not None:
                    traj_x.append(float(center.x))
                    traj_y.append(float(center.y))
            planned_trajectory.set_data(traj_x, traj_y)
        else:
            planned_trajectory.set_data([], [])

        ax.set_xlim(current_x - margin_m, current_x + margin_m)
        ax.set_ylim(current_y - margin_m, current_y + margin_m)

        time_text.set_text(
            f"scenario={scenario_name} ({scenario_type})\n"
            f"log={log_name}\n"
            f"planner={planner_name}\n"
            f"frame={frame_idx + 1:04d}/{len(samples):04d}  t={float(sample.iteration.time_s):.2f}s\n"
            f"ego=({current_x:.1f}, {current_y:.1f})"
        )
        return ego_path, ego_box, ego_heading, agent_boxes, agent_velocity, planned_trajectory, time_text

    ani = animation.FuncAnimation(fig, update, frames=len(samples), interval=1000 / fps, blit=False)
    writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(str(output_path), writer=writer, dpi=150)
    plt.close(fig)
    return output_path
