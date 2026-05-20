#!/usr/bin/env python3
"""Shared utilities to render recorded nuPlan scenarios to MP4.

This module contains reusable rendering/data-loading logic so multiple demos
or scripts can produce consistent recorded-scenario videos.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import wkb
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon
from shapely.ops import transform as shapely_transform

from common.nuplan_demo_utils import (
    add_common_args,
    choose_col,
    connect,
    ensure_output_dir,
    find_db_files,
    first_db_with_tables,
    list_tables,
    print_db_selection,
    table_columns,
)

MAP_UTM_EPSG = {
    "sg-one-north": 32648,
    "us-ma-boston": 32619,
    "us-nv-las-vegas-strip": 32611,
    "us-pa-pittsburgh-hazelwood": 32617,
}

STATUS_COLORS = {
    "green": "#2ca02c",
    "yellow": "#f2b701",
    "red": "#d62728",
    "unknown": "#6c757d",
}

AGENT_CLASS_COLORS = {
    "vehicle": "#2ca02c",
    "pedestrian": "#ff7f0e",
    "bicycle": "#9467bd",
    "bus": "#1f77b4",
    "generic_object": "#8c564b",
    "unknown": "#17becf",
}

MAP_LAYER_SPECS = [
    {"table": "drivable_area", "kind": "polygon", "facecolor": "#f0eee8", "edgecolor": "none", "alpha": 0.35, "zorder": -1},
    {"table": "road_segments", "kind": "polygon", "facecolor": "#ebe7df", "edgecolor": "none", "alpha": 0.65, "zorder": 0},
    {"table": "lanes_polygons", "kind": "polygon", "facecolor": "#e7e4dc", "edgecolor": "none", "alpha": 0.45, "zorder": 1},
    {"table": "lane_groups_polygons", "kind": "polygon", "facecolor": "#d8d4cb", "edgecolor": "none", "alpha": 0.2, "zorder": 1},
    {"table": "intersections", "kind": "polygon", "facecolor": "#ddd8cf", "edgecolor": "none", "alpha": 0.55, "zorder": 1},
    {"table": "crosswalks", "kind": "polygon", "facecolor": "#f6e6a9", "edgecolor": "#d9c36a", "alpha": 0.75, "zorder": 2},
    {"table": "walkways", "kind": "polygon", "facecolor": "#dcefd8", "edgecolor": "none", "alpha": 0.4, "zorder": 2},
    {"table": "carpark_areas", "kind": "polygon", "facecolor": "#dae6f5", "edgecolor": "none", "alpha": 0.35, "zorder": 2},
    {"table": "stop_polygons", "kind": "polygon", "facecolor": "#f3d6d6", "edgecolor": "#c77d7d", "alpha": 0.65, "zorder": 3},
    {"table": "boundaries", "kind": "line", "color": "#7d7a73", "linewidth": 0.7, "alpha": 0.6, "zorder": 4},
    {"table": "lane_group_connectors", "kind": "line", "color": "#6d685d", "linewidth": 0.6, "alpha": 0.45, "zorder": 4},
    {"table": "lane_connectors", "kind": "line", "color": "#9b968c", "linewidth": 1.0, "alpha": 0.7, "zorder": 5},
    {"table": "baseline_paths", "kind": "line", "color": "#4472c4", "linewidth": 0.9, "alpha": 0.5, "zorder": 5},
    {"table": "traffic_lights", "kind": "point", "color": "#222222", "size": 18, "alpha": 0.85, "zorder": 6},
]


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
    return "unknown"


def yaw_from_quaternion(qw: float, qx: float, qy: float, qz: float) -> float:
    # Standard Z-axis yaw extraction from quaternion.
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(math.atan2(siny_cosp, cosy_cosp))


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


def find_map_root(data_root: Path) -> Path:
    candidates = [
        Path(os.environ.get("NUPLAN_MAPS_ROOT", "")),
        data_root / "maps",
        data_root / "nuplan-v1.1" / "maps",
    ]
    for candidate in candidates:
        if str(candidate) and candidate.exists():
            return candidate
    return data_root / "maps"


def load_log_map_info(db_path: Path) -> tuple[str, str]:
    with connect(db_path) as con:
        row = con.execute("SELECT location, map_version FROM log LIMIT 1").fetchone()
    if row is None:
        raise RuntimeError("Could not read log metadata from the selected DB.")
    return str(row[0]), str(row[1])


def load_route_goal(db_path: Path) -> tuple[float, float] | None:
    with connect(db_path) as con:
        tables = list_tables(con)
        if "scene" not in tables or "ego_pose" not in tables:
            return None
        row = con.execute(
            """
            SELECT ep.x, ep.y
            FROM scene AS sc
            JOIN ego_pose AS ep ON ep.token = sc.goal_ego_pose_token
            LIMIT 1
            """
        ).fetchone()
    if row is None:
        return None
    return float(row[0]), float(row[1])


def load_scene_metadata(db_path: Path) -> dict[str, str]:
    with connect(db_path) as con:
        tables = list_tables(con)
        if "scene" not in tables:
            return {}
        row = con.execute("SELECT name, roadblock_ids FROM scene LIMIT 1").fetchone()
    if row is None:
        return {}
    return {
        "scene_name": str(row[0] or ""),
        "roadblock_ids": str(row[1] or ""),
    }


def resolve_map_path(data_root: Path, map_version: str, location: str) -> Path:
    map_root = find_map_root(data_root)
    candidates = [map_root / map_version, map_root / location]
    for base in candidates:
        if base.exists():
            matches = sorted(base.glob("*/map.gpkg"))
            if matches:
                return matches[0]
    matches = sorted(map_root.glob(f"{map_version}/*/map.gpkg"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not locate map.gpkg for map_version={map_version} under {map_root}")


def resolve_local_crs(map_version: str) -> str:
    epsg = MAP_UTM_EPSG.get(map_version)
    if epsg is None:
        raise RuntimeError(
            f"Unsupported map version for local CRS conversion: {map_version}. "
            f"Known map versions: {', '.join(sorted(MAP_UTM_EPSG))}"
        )
    return f"EPSG:{epsg}"


def geopackage_geom_to_wkb(blob: bytes) -> bytes:
    if len(blob) < 8 or blob[:2] != b"GP":
        raise ValueError("Geometry blob is not a valid GeoPackage geometry")
    flags = blob[3]
    envelope_indicator = (flags >> 1) & 0b111
    envelope_sizes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}
    if envelope_indicator not in envelope_sizes:
        raise ValueError(f"Unsupported GeoPackage envelope indicator: {envelope_indicator}")
    return blob[8 + envelope_sizes[envelope_indicator] :]


def geometry_to_arrays(geometry: object) -> list[list[tuple[float, float]]]:
    if geometry is None or getattr(geometry, "is_empty", True):
        return []
    if isinstance(geometry, Point):
        return [[(float(geometry.x), float(geometry.y))]]
    if isinstance(geometry, LineString):
        return [[(float(x), float(y)) for x, y in geometry.coords]]
    if isinstance(geometry, Polygon):
        return [[(float(x), float(y)) for x, y in geometry.exterior.coords]]
    if isinstance(geometry, (MultiLineString, MultiPolygon)):
        arrays: list[list[tuple[float, float]]] = []
        for part in geometry.geoms:
            arrays.extend(geometry_to_arrays(part))
        return arrays
    if hasattr(geometry, "geoms"):
        arrays = []
        for part in geometry.geoms:
            arrays.extend(geometry_to_arrays(part))
        return arrays
    return []


def local_bbox_to_lonlat_bounds(
    ego_df: pd.DataFrame,
    margin_m: float,
    transformer_local_to_geo: Transformer,
) -> tuple[float, float, float, float]:
    min_x = float(ego_df["x"].min()) - margin_m
    max_x = float(ego_df["x"].max()) + margin_m
    min_y = float(ego_df["y"].min()) - margin_m
    max_y = float(ego_df["y"].max()) + margin_m
    corners = [(min_x, min_y), (min_x, max_y), (max_x, min_y), (max_x, max_y)]
    lonlat = [transformer_local_to_geo.transform(x, y) for x, y in corners]
    lons = [pair[0] for pair in lonlat]
    lats = [pair[1] for pair in lonlat]
    return min(lons), min(lats), max(lons), max(lats)


def load_map_layers(
    map_path: Path,
    bbox_lonlat: tuple[float, float, float, float],
    transformer_geo_to_local: Transformer,
) -> tuple[dict[str, list[list[tuple[float, float]]]], dict[int, list[list[tuple[float, float]]]]]:
    min_lon, min_lat, max_lon, max_lat = bbox_lonlat
    static_layers: dict[str, list[list[tuple[float, float]]]] = {}
    lane_connector_segments: dict[int, list[list[tuple[float, float]]]] = {}

    with connect(map_path) as con:
        tables = list_tables(con)
        for spec in MAP_LAYER_SPECS:
            table = spec["table"]
            rtree_table = f"rtree_{table}_geom"
            if table not in tables or rtree_table not in tables:
                continue

            rows = con.execute(
                f"""
                SELECT src.fid, src.geom
                FROM {table} AS src
                JOIN {rtree_table} AS idx ON idx.id = src.fid
                WHERE idx.minx <= ? AND idx.maxx >= ?
                  AND idx.miny <= ? AND idx.maxy >= ?
                LIMIT 4000
                """,
                (max_lon, min_lon, max_lat, min_lat),
            ).fetchall()

            arrays: list[list[tuple[float, float]]] = []
            for fid, geom_blob in rows:
                geometry = wkb.loads(geopackage_geom_to_wkb(geom_blob))
                local_geometry = shapely_transform(transformer_geo_to_local.transform, geometry)
                geometry_arrays = geometry_to_arrays(local_geometry)
                if not geometry_arrays:
                    continue
                arrays.extend(geometry_arrays)
                if table == "lane_connectors":
                    lane_connector_segments[int(fid)] = geometry_arrays

            static_layers[table] = arrays

    return static_layers, lane_connector_segments


def load_ego_frames(db_path: Path, frame_limit: int | None, show_schema: bool) -> pd.DataFrame:
    with connect(db_path) as con:
        ego_cols = table_columns(con, "ego_pose")
        lidar_cols = table_columns(con, "lidar_pc")
        if show_schema:
            print("ego_pose columns:", ego_cols)
            print("lidar_pc columns:", lidar_cols)

        x_col = choose_col(ego_cols, ["x", "translation_x", "ego_pose_x"])
        y_col = choose_col(ego_cols, ["y", "translation_y", "ego_pose_y"])
        vx_col = choose_col(ego_cols, ["vx", "velocity_x"])
        vy_col = choose_col(ego_cols, ["vy", "velocity_y"])
        ax_col = choose_col(ego_cols, ["acceleration_x", "ax"])
        ay_col = choose_col(ego_cols, ["acceleration_y", "ay"])
        qw_col = choose_col(ego_cols, ["qw"])
        qx_col = choose_col(ego_cols, ["qx"])
        qy_col = choose_col(ego_cols, ["qy"])
        qz_col = choose_col(ego_cols, ["qz"])
        timestamp_col = choose_col(lidar_cols, ["timestamp"])

        if not x_col or not y_col or not timestamp_col:
            raise RuntimeError(
                f"Could not identify ego frame columns. ego_pose={ego_cols}, lidar_pc={lidar_cols}"
            )

        select_parts = [
            "selected_frames.timestamp_us AS timestamp_us",
            f"ego_pose.{x_col} AS x",
            f"ego_pose.{y_col} AS y",
        ]
        if vx_col:
            select_parts.append(f"ego_pose.{vx_col} AS vx")
        if vy_col:
            select_parts.append(f"ego_pose.{vy_col} AS vy")
        if ax_col:
            select_parts.append(f"ego_pose.{ax_col} AS acceleration_x")
        if ay_col:
            select_parts.append(f"ego_pose.{ay_col} AS acceleration_y")
        if qw_col and qx_col and qy_col and qz_col:
            select_parts.extend(
                [
                    f"ego_pose.{qw_col} AS qw",
                    f"ego_pose.{qx_col} AS qx",
                    f"ego_pose.{qy_col} AS qy",
                    f"ego_pose.{qz_col} AS qz",
                ]
            )

        limit_clause = "" if frame_limit is None else "LIMIT ?"
        query = f"""
            WITH selected_frames AS (
                SELECT token, ego_pose_token, {timestamp_col} AS timestamp_us
                FROM lidar_pc
                ORDER BY {timestamp_col} ASC
                {limit_clause}
            )
            SELECT {', '.join(select_parts)}
            FROM selected_frames
            JOIN ego_pose ON selected_frames.ego_pose_token = ego_pose.token
            ORDER BY selected_frames.timestamp_us ASC
        """
        params: tuple[object, ...] = () if frame_limit is None else (frame_limit,)
        rows = con.execute(query, params).fetchall()

    if not rows:
        raise RuntimeError("No ego frames found in selected DB.")

    df = pd.DataFrame([dict(row) for row in rows])
    df["t_s"] = (df["timestamp_us"] - df["timestamp_us"].iloc[0]) / 1e6
    if "vx" in df.columns and "vy" in df.columns:
        df["speed_mps"] = np.hypot(df["vx"].fillna(0.0), df["vy"].fillna(0.0))
    else:
        df["speed_mps"] = 0.0

    if "acceleration_x" in df.columns and "acceleration_y" in df.columns:
        df["accel_mps2"] = np.hypot(df["acceleration_x"].fillna(0.0), df["acceleration_y"].fillna(0.0))
    else:
        df["accel_mps2"] = 0.0

    if {"qw", "qx", "qy", "qz"}.issubset(df.columns):
        df["yaw"] = df.apply(
            lambda row: yaw_from_quaternion(float(row["qw"]), float(row["qx"]), float(row["qy"]), float(row["qz"])),
            axis=1,
        )
    else:
        df["yaw"] = 0.0

    df["ego_length"] = 4.8
    df["ego_width"] = 2.1
    return df


def load_agent_frames(db_path: Path, frame_limit: int | None, show_schema: bool) -> pd.DataFrame:
    with connect(db_path) as con:
        tables = list_tables(con)
        lb_cols = table_columns(con, "lidar_box")
        lidar_cols = table_columns(con, "lidar_pc")
        track_cols = table_columns(con, "track") if "track" in tables else []
        category_cols = table_columns(con, "category") if "category" in tables else []
        if show_schema:
            print("lidar_box columns:", lb_cols)
            print("lidar_pc columns:", lidar_cols)
            if track_cols:
                print("track columns:", track_cols)
            if category_cols:
                print("category columns:", category_cols)

        x_col = choose_col(lb_cols, ["x", "translation_x", "center_x"])
        y_col = choose_col(lb_cols, ["y", "translation_y", "center_y"])
        length_col = choose_col(lb_cols, ["length", "size_x"])
        width_col = choose_col(lb_cols, ["width", "size_y"])
        track_col = choose_col(lb_cols, ["track_token"])
        yaw_col = choose_col(lb_cols, ["yaw", "heading"])
        vx_col = choose_col(lb_cols, ["vx", "velocity_x"])
        vy_col = choose_col(lb_cols, ["vy", "velocity_y"])
        confidence_col = choose_col(lb_cols, ["confidence"])
        timestamp_col = choose_col(lidar_cols, ["timestamp"])

        track_length_col = choose_col(track_cols, ["length", "size_x"])
        track_width_col = choose_col(track_cols, ["width", "size_y"])
        track_category_col = choose_col(track_cols, ["category_token"])
        category_name_col = choose_col(category_cols, ["name"]) if category_cols else None

        if not x_col or not y_col or not timestamp_col:
            raise RuntimeError(
                f"Could not identify agent frame columns. lidar_box={lb_cols}, lidar_pc={lidar_cols}"
            )

        select_parts = [
            "selected_frames.timestamp_us AS timestamp_us",
            f"lidar_box.{x_col} AS x",
            f"lidar_box.{y_col} AS y",
        ]
        if length_col:
            select_parts.append(f"lidar_box.{length_col} AS length")
        elif track_length_col:
            select_parts.append(f"track.{track_length_col} AS length")
        if width_col:
            select_parts.append(f"lidar_box.{width_col} AS width")
        elif track_width_col:
            select_parts.append(f"track.{track_width_col} AS width")
        if track_col:
            select_parts.append(f"lidar_box.{track_col} AS track_token")
        if yaw_col:
            select_parts.append(f"lidar_box.{yaw_col} AS yaw")
        if vx_col:
            select_parts.append(f"lidar_box.{vx_col} AS vx")
        if vy_col:
            select_parts.append(f"lidar_box.{vy_col} AS vy")
        if confidence_col:
            select_parts.append(f"lidar_box.{confidence_col} AS confidence")
        if category_name_col and track_category_col and track_col:
            select_parts.append(f"category.{category_name_col} AS category_name")

        left_join_track = ""
        left_join_category = ""
        if track_col and "track" in tables:
            left_join_track = f"LEFT JOIN track ON track.token = lidar_box.{track_col}"
            if category_name_col and track_category_col and "category" in tables:
                left_join_category = f"LEFT JOIN category ON category.token = track.{track_category_col}"

        limit_clause = "" if frame_limit is None else "LIMIT ?"
        query = f"""
            WITH selected_frames AS (
                SELECT token, {timestamp_col} AS timestamp_us
                FROM lidar_pc
                ORDER BY {timestamp_col} ASC
                {limit_clause}
            )
            SELECT {', '.join(select_parts)}
            FROM selected_frames
            JOIN lidar_box ON lidar_box.lidar_pc_token = selected_frames.token
            {left_join_track}
            {left_join_category}
            ORDER BY selected_frames.timestamp_us ASC
        """
        params: tuple[object, ...] = () if frame_limit is None else (frame_limit,)
        rows = con.execute(query, params).fetchall()

    if not rows:
        raise RuntimeError("No tracked agent boxes found in selected DB.")

    df = pd.DataFrame([dict(row) for row in rows])
    if "yaw" not in df.columns:
        df["yaw"] = 0.0
    if "length" not in df.columns:
        df["length"] = 4.0
    if "width" not in df.columns:
        df["width"] = 1.8
    if "vx" in df.columns and "vy" in df.columns:
        df["speed_mps"] = np.hypot(df["vx"].fillna(0.0), df["vy"].fillna(0.0))
    else:
        df["speed_mps"] = 0.0
    if "confidence" not in df.columns:
        df["confidence"] = 1.0
    df["category"] = df.get("category_name", "unknown").map(normalize_category_name)
    return df


def load_traffic_light_frames(db_path: Path, frame_limit: int | None, show_schema: bool) -> pd.DataFrame:
    with connect(db_path) as con:
        tables = list_tables(con)
        candidates = [t for t in ["traffic_light_status", "traffic_light_statuses"] if t in tables]
        if not candidates:
            return pd.DataFrame(columns=["timestamp_us", "lane_connector_id", "status"])
        table = candidates[0]
        tl_cols = table_columns(con, table)
        lp_cols = table_columns(con, "lidar_pc")
        if show_schema:
            print(f"{table} columns:", tl_cols)
            print("lidar_pc columns:", lp_cols)

        lane_col = choose_col(tl_cols, ["lane_connector_id", "lane_id", "traffic_light_lane_connector_id"])
        status_col = choose_col(tl_cols, ["status", "traffic_light_status_type", "type"])
        lp_token_col = choose_col(tl_cols, ["lidar_pc_token"])
        timestamp_col = choose_col(lp_cols, ["timestamp"])
        if not lane_col or not status_col or not lp_token_col or not timestamp_col:
            return pd.DataFrame(columns=["timestamp_us", "lane_connector_id", "status"])

        limit_clause = "" if frame_limit is None else "LIMIT ?"
        query = f"""
            WITH selected_frames AS (
                SELECT token, {timestamp_col} AS timestamp_us
                FROM lidar_pc
                ORDER BY {timestamp_col} ASC
                {limit_clause}
            )
            SELECT
                selected_frames.timestamp_us AS timestamp_us,
                tl.{lane_col} AS lane_connector_id,
                tl.{status_col} AS status
            FROM selected_frames
            JOIN {table} AS tl ON tl.{lp_token_col} = selected_frames.token
            ORDER BY selected_frames.timestamp_us ASC
            """
        params: tuple[object, ...] = () if frame_limit is None else (frame_limit,)
        rows = con.execute(query, params).fetchall()

    if not rows:
        return pd.DataFrame(columns=["timestamp_us", "lane_connector_id", "status"])
    return pd.DataFrame([dict(row) for row in rows])


def add_static_map_artists(ax: plt.Axes, static_layers: dict[str, list[list[tuple[float, float]]]]) -> None:
    for spec in MAP_LAYER_SPECS:
        table = spec["table"]
        arrays = static_layers.get(table, [])
        if not arrays:
            continue
        if spec["kind"] == "polygon":
            artist = PolyCollection(
                arrays,
                facecolors=spec["facecolor"],
                edgecolors=spec["edgecolor"],
                linewidths=0.4,
                alpha=spec["alpha"],
                zorder=spec["zorder"],
            )
            ax.add_collection(artist)
        elif spec["kind"] == "line":
            artist = LineCollection(
                arrays,
                colors=spec["color"],
                linewidths=spec["linewidth"],
                alpha=spec["alpha"],
                zorder=spec["zorder"],
            )
            ax.add_collection(artist)
        elif spec["kind"] == "point":
            xs = [coords[0][0] for coords in arrays if coords]
            ys = [coords[0][1] for coords in arrays if coords]
            ax.scatter(xs, ys, s=spec["size"], c=spec["color"], alpha=spec["alpha"], zorder=spec["zorder"])


def traffic_light_segments_for_timestamp(
    frame_tl: pd.DataFrame,
    lane_connector_segments: dict[int, list[list[tuple[float, float]]]],
) -> tuple[list[list[tuple[float, float]]], list[str]]:
    segments: list[list[tuple[float, float]]] = []
    colors: list[str] = []
    if frame_tl.empty:
        return segments, colors

    for row in frame_tl.itertuples(index=False):
        lane_id = int(getattr(row, "lane_connector_id"))
        status = str(getattr(row, "status", "unknown")).lower()
        for segment in lane_connector_segments.get(lane_id, []):
            segments.append(segment)
            colors.append(STATUS_COLORS.get(status, STATUS_COLORS["unknown"]))
    return segments, colors


def render_video(
    ego_df: pd.DataFrame,
    agent_df: pd.DataFrame,
    traffic_light_df: pd.DataFrame,
    static_layers: dict[str, list[list[tuple[float, float]]]],
    lane_connector_segments: dict[int, list[list[tuple[float, float]]]],
    goal_xy: tuple[float, float] | None,
    scene_metadata: dict[str, str],
    output_path: Path,
    fps: int,
    tail_length: int,
    track_tail_length: int,
    margin_m: float,
) -> Path:
    agent_groups = {timestamp: group for timestamp, group in agent_df.groupby("timestamp_us")}
    tl_groups = {timestamp: group for timestamp, group in traffic_light_df.groupby("timestamp_us")}

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.set_facecolor("#f7f4ee")
    add_static_map_artists(ax, static_layers)

    if goal_xy:
        ax.scatter([goal_xy[0]], [goal_xy[1]], s=180, c="#ffd700", marker="*", label="goal", zorder=7, edgecolors="#c9a301", linewidths=1.5)

    highlighted_tl = LineCollection([], linewidths=2.8, alpha=0.95, zorder=7)
    ax.add_collection(highlighted_tl)
    agent_boxes = PolyCollection([], facecolors=[], edgecolors="#1f1f1f", linewidths=0.6, alpha=0.7, zorder=8)
    ax.add_collection(agent_boxes)
    agent_velocity = LineCollection([], linewidths=1.0, alpha=0.5, zorder=8)
    ax.add_collection(agent_velocity)
    track_trails = LineCollection([], colors="#2f4f4f", linewidths=1.0, alpha=0.35, zorder=7)
    ax.add_collection(track_trails)
    tail_line, = ax.plot([], [], color="#1f77b4", linewidth=2.5, label="ego path", zorder=8)
    ego_box = PolyCollection([], facecolors="#d62728", edgecolors="#7f1d1d", linewidths=1.1, alpha=0.95, zorder=10)
    ax.add_collection(ego_box)
    ego_heading, = ax.plot([], [], color="#9e1f1f", linewidth=2.2, zorder=11)
    time_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        family="monospace",
        bbox={"facecolor": "#ffffff", "edgecolor": "#c6c2b7", "alpha": 0.88, "boxstyle": "round,pad=0.35"},
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Recorded nuPlan scenario playback with infrastructure")
    ax.legend(loc="lower right")

    def update(frame_idx: int):
        ego_now = ego_df.iloc[frame_idx]
        start_idx = max(0, frame_idx - tail_length + 1)
        ego_tail = ego_df.iloc[start_idx : frame_idx + 1]
        tail_line.set_data(ego_tail["x"].to_numpy(), ego_tail["y"].to_numpy())

        ego_poly = oriented_box_polygon(
            float(ego_now["x"]),
            float(ego_now["y"]),
            float(ego_now.get("ego_length", 4.8)),
            float(ego_now.get("ego_width", 2.1)),
            float(ego_now.get("yaw", 0.0)),
        )
        ego_box.set_verts([ego_poly])
        heading_len = 3.0
        hx = float(ego_now["x"]) + heading_len * math.cos(float(ego_now.get("yaw", 0.0)))
        hy = float(ego_now["y"]) + heading_len * math.sin(float(ego_now.get("yaw", 0.0)))
        ego_heading.set_data([float(ego_now["x"]), hx], [float(ego_now["y"]), hy])

        frame_agents = agent_groups.get(ego_now["timestamp_us"])
        category_counts: dict[str, int] = {}
        if frame_agents is not None and not frame_agents.empty:
            agent_polys: list[list[tuple[float, float]]] = []
            facecolors: list[str] = []
            velocity_segments: list[list[tuple[float, float]]] = []
            velocity_colors: list[str] = []
            for row in frame_agents.itertuples(index=False):
                category = str(getattr(row, "category", "unknown"))
                base_color = AGENT_CLASS_COLORS.get(category, AGENT_CLASS_COLORS["unknown"])
                category_counts[category] = category_counts.get(category, 0) + 1
                agent_polys.append(
                    oriented_box_polygon(
                        float(getattr(row, "x")),
                        float(getattr(row, "y")),
                        float(getattr(row, "length", 4.0)),
                        float(getattr(row, "width", 1.8)),
                        float(getattr(row, "yaw", 0.0)),
                    )
                )
                confidence = float(getattr(row, "confidence", 1.0))
                confidence = min(max(confidence, 0.1), 1.0)
                facecolors.append(base_color)
                if hasattr(row, "vx") and hasattr(row, "vy"):
                    speed = float(getattr(row, "speed_mps", 0.0))
                    if speed > 0.2:
                        sx = float(getattr(row, "x"))
                        sy = float(getattr(row, "y"))
                        scale = min(3.5, 0.25 * speed)
                        ex = sx + scale * float(getattr(row, "vx", 0.0))
                        ey = sy + scale * float(getattr(row, "vy", 0.0))
                        velocity_segments.append([(sx, sy), (ex, ey)])
                        velocity_colors.append(base_color)
            agent_boxes.set_verts(agent_polys)
            agent_boxes.set_facecolors(facecolors)
            agent_velocity.set_segments(velocity_segments)
            agent_velocity.set_color(velocity_colors)

            if "track_token" in frame_agents.columns:
                trails: list[list[tuple[float, float]]] = []
                current_tracks = set(frame_agents["track_token"].dropna().tolist())
                if current_tracks:
                    history = agent_df[
                        (agent_df["timestamp_us"] <= ego_now["timestamp_us"])
                        & (agent_df["track_token"].isin(current_tracks))
                    ]
                    for _, hist in history.groupby("track_token"):
                        hist_tail = hist.tail(max(track_tail_length, 2))
                        if len(hist_tail) >= 2:
                            trails.append([(float(x), float(y)) for x, y in hist_tail[["x", "y"]].to_numpy()])
                track_trails.set_segments(trails)
            else:
                track_trails.set_segments([])
        else:
            agent_boxes.set_verts([])
            agent_velocity.set_segments([])
            track_trails.set_segments([])

        frame_tl = tl_groups.get(ego_now["timestamp_us"], pd.DataFrame())
        tl_segments, tl_colors = traffic_light_segments_for_timestamp(frame_tl, lane_connector_segments)
        highlighted_tl.set_segments(tl_segments)
        highlighted_tl.set_color(tl_colors)

        x_center = float(ego_now["x"])
        y_center = float(ego_now["y"])
        ax.set_xlim(x_center - margin_m, x_center + margin_m)
        ax.set_ylim(y_center - margin_m, y_center + margin_m)

        tl_counts = {"green": 0, "yellow": 0, "red": 0, "other": 0}
        if not frame_tl.empty and "status" in frame_tl.columns:
            for status in frame_tl["status"].astype(str).str.lower().tolist():
                if status in tl_counts:
                    tl_counts[status] += 1
                else:
                    tl_counts["other"] += 1

        category_summary = ", ".join(
            f"{name}:{count}" for name, count in sorted(category_counts.items(), key=lambda pair: pair[0])
        )
        if not category_summary:
            category_summary = "none"

        scene_name = scene_metadata.get("scene_name", "")
        roadblock_ids = scene_metadata.get("roadblock_ids", "")
        roadblock_short = roadblock_ids[:64] + ("..." if len(roadblock_ids) > 64 else "")
        time_text.set_text(
            f"t={float(ego_now['t_s']):6.2f}s  frame={frame_idx + 1:03d}/{len(ego_df):03d}\n"
            f"ego speed={float(ego_now.get('speed_mps', 0.0)):5.2f} m/s  accel={float(ego_now.get('accel_mps2', 0.0)):5.2f} m/s^2\n"
            f"ego yaw={math.degrees(float(ego_now.get('yaw', 0.0))):6.1f} deg  agents={0 if frame_agents is None else len(frame_agents)} ({category_summary})\n"
            f"traffic lights: g={tl_counts['green']} y={tl_counts['yellow']} r={tl_counts['red']} o={tl_counts['other']}\n"
            f"scene={scene_name or 'unknown'}\n"
            f"roadblocks={roadblock_short or 'n/a'}"
        )
        return tail_line, ego_box, ego_heading, agent_boxes, agent_velocity, track_trails, highlighted_tl, time_text

    ani = animation.FuncAnimation(fig, update, frames=len(ego_df), interval=1000 / max(fps, 1), blit=False)
    writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
    ani.save(str(output_path), writer=writer, dpi=150)
    plt.close(fig)
    return output_path


def add_recorded_render_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Number of lidar frames to render into the MP4. Use 0 or a negative value to render all available frames.",
    )
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for the output MP4.")
    parser.add_argument("--tail-length", type=int, default=30, help="How many ego samples to retain as a short path tail.")
    parser.add_argument("--track-tail", type=int, default=15, help="How many past samples to retain per tracked agent.")
    parser.add_argument("--margin", type=float, default=40.0, help="Meters of view margin around the ego vehicle.")
    return parser


def run_recorded_scenario_render(
    *,
    data_root: Path,
    db: Path | None,
    output_dir: Path,
    show_schema: bool,
    frames: int,
    fps: int,
    tail_length: int,
    track_tail: int,
    margin: float,
    output_name: str = "11_recorded_scenario.mp4",
) -> Path:
    safe_fps = max(int(fps), 1)
    frame_limit: int | None = None if int(frames) <= 0 else int(frames)

    db_files = find_db_files(data_root, db)
    db = first_db_with_tables(db_files, ["lidar_pc", "ego_pose", "lidar_box"])
    print_db_selection(db.path, db.tables)

    out_dir = ensure_output_dir(output_dir)
    ego_df = load_ego_frames(db.path, frame_limit, show_schema)
    agent_df = load_agent_frames(db.path, frame_limit, show_schema)
    traffic_light_df = load_traffic_light_frames(db.path, frame_limit, show_schema)

    location, map_version = load_log_map_info(db.path)
    local_crs = resolve_local_crs(map_version)
    map_path = resolve_map_path(Path(data_root), map_version, location)
    transformer_local_to_geo = Transformer.from_crs(local_crs, "EPSG:4326", always_xy=True)
    transformer_geo_to_local = Transformer.from_crs("EPSG:4326", local_crs, always_xy=True)
    bbox_lonlat = local_bbox_to_lonlat_bounds(ego_df, max(margin, 5.0), transformer_local_to_geo)
    static_layers, lane_connector_segments = load_map_layers(map_path, bbox_lonlat, transformer_geo_to_local)
    goal_xy = load_route_goal(db.path)
    scene_metadata = load_scene_metadata(db.path)

    mp4_path = out_dir / output_name
    render_video(
        ego_df=ego_df,
        agent_df=agent_df,
        traffic_light_df=traffic_light_df,
        static_layers=static_layers,
        lane_connector_segments=lane_connector_segments,
        goal_xy=goal_xy,
        scene_metadata=scene_metadata,
        output_path=mp4_path,
        fps=safe_fps,
        tail_length=max(tail_length, 1),
        track_tail_length=max(track_tail, 2),
        margin_m=max(margin, 5.0),
    )

    video_duration_s = len(ego_df) / safe_fps
    print(f"[demo] map location: {location}")
    print(f"[demo] map file: {map_path}")
    print(f"[demo] frames rendered: {len(ego_df)} ({video_duration_s:.1f} seconds)")
    print(f"[demo] tracked boxes rendered: {len(agent_df)}")
    print(f"[demo] traffic light rows rendered: {len(traffic_light_df)}")
    if scene_metadata.get("scene_name"):
        print(f"[demo] scene name: {scene_metadata['scene_name']}")
    if goal_xy:
        print(f"[demo] goal location: ({goal_xy[0]:.1f}, {goal_xy[1]:.1f})")
    print(f"[demo] wrote: {mp4_path}")
    print("[demo] component demonstrated: Recorded scenario rendering utility = YES")
    return mp4_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    add_recorded_render_args(parser)
    args = parser.parse_args()
    run_recorded_scenario_render(
        data_root=args.data_root,
        db=args.db,
        output_dir=args.output_dir,
        show_schema=args.show_schema,
        frames=args.frames,
        fps=args.fps,
        tail_length=args.tail_length,
        track_tail=args.track_tail,
        margin=args.margin,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
