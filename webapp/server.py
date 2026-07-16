#!/usr/bin/env python3
"""Flask API and background whale-tracking worker for the local web app."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from collections import defaultdict
from io import BytesIO, StringIO
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from ultralytics import YOLO
# TODO: path to Ultralytics Settings  For help see https://docs.ultralytics.com/quickstart/#ultralytics-settings.

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from whaledrone_hackathon_code.track_and_geolocate import (
    assign_stable_track_id,
    geographic_distance_m,
    load_intrinsics,
    load_telemetry,
    make_tracker,
    reproject_fov,
    smooth_position,
    yaw_rate_deg_s,
    zoom_intrinsics,
)

CLASS_NAMES = {0: "whale", 1: "boat", 2: "dolphin"}


class TrackingCache:
    """Thread-safe, restartable cache for one video analysis session."""

    def __init__(self, path: Path, video_path: Path, metadata: dict, settings: dict):
        self.path = path
        self.video_path = video_path.resolve()
        self.metadata = metadata
        self.settings = settings
        self.lock = threading.RLock()
        self.positions_by_frame: dict[int, list[dict]] = defaultdict(list)
        self.data = self._load_or_create()
        self._index_positions()

    def _empty_data(self):
        return {
            "version": 1,
            "video": str(self.video_path),
            "metadata": self.metadata,
            "settings": self.settings,
            "positions": [],
            "inference": {
                "running": False,
                "done": False,
                "processed_frames": 0,
                "total_frames": self.metadata["total_frames"],
                "error": None,
                "started_at": None,
                "updated_at": None,
            },
        }

    def _load_or_create(self):
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}
            if (
                data.get("video") == str(self.video_path)
                and data.get("settings") == self.settings
            ):
                data.setdefault("positions", [])
                data.setdefault("metadata", self.metadata)
                inference = data.setdefault("inference", {})
                inference.setdefault("processed_frames", 0)
                inference.setdefault("total_frames", self.metadata["total_frames"])
                inference.setdefault("done", False)
                inference.setdefault("error", None)
                if inference["done"]:
                    inference["running"] = False
                    return data
        return self._empty_data()

    def _index_positions(self):
        for position in self.data["positions"]:
            self.positions_by_frame[int(position["frame"])].append(position)

    def _save_locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(self.data, separators=(",", ":")), encoding="utf-8"
        )
        temporary_path.replace(self.path)

    def save(self):
        with self.lock:
            self._save_locked()

    def begin(self):
        with self.lock:
            inference = self.data["inference"]
            if inference["done"] or inference["running"]:
                return False
            inference.update(
                {
                    "running": True,
                    "error": None,
                    "started_at": time.time(),
                    "updated_at": time.time(),
                }
            )
            self._save_locked()
            return True

    def append_positions(self, positions: list[dict]):
        if not positions:
            return
        with self.lock:
            start_sequence = len(self.data["positions"])
            for index, position in enumerate(positions, start=1):
                position["sequence"] = start_sequence + index
                self.data["positions"].append(position)
                self.positions_by_frame[int(position["frame"])].append(position)

    def update_progress(self, processed_frames: int, coordinates_paused: bool):
        with self.lock:
            inference = self.data["inference"]
            inference["processed_frames"] = processed_frames
            inference["updated_at"] = time.time()
            if coordinates_paused:
                inference["paused_coordinate_frames"] = (
                    inference.get("paused_coordinate_frames", 0) + 1
                )

    def finish(self, error: str | None = None):
        with self.lock:
            self.data["inference"].update(
                {
                    "running": False,
                    "done": error is None,
                    "error": error,
                    "updated_at": time.time(),
                }
            )
            self._save_locked()

    def status(self):
        with self.lock:
            return {
                "video": self.video_path.name,
                **self.data["metadata"],
                "inference": dict(self.data["inference"]),
                "position_count": len(self.data["positions"]),
            }

    def positions_after(self, cursor: int):
        with self.lock:
            positions = [
                dict(position) for position in self.data["positions"][max(cursor, 0) :]
            ]
            return positions, len(self.data["positions"])

    def positions_at_frame(self, frame_number: int):
        with self.lock:
            return [
                dict(position)
                for position in self.positions_by_frame.get(frame_number, [])
            ]

    def reset(self):
        with self.lock:
            self.data = self._empty_data()
            self.positions_by_frame.clear()
            self._save_locked()


def probe_video(video_path: Path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open video: {video_path}")
    metadata = {
        "fps": capture.get(cv2.CAP_PROP_FPS) or 30.0,
        "width": round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": round(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    capture.release()
    return metadata


def pipeline_settings(arguments):
    return {
        "confidence": arguments.confidence,
        "classes": arguments.classes,
        "track_high_thresh": arguments.track_high_thresh,
        "track_low_thresh": arguments.track_low_thresh,
        "new_track_thresh": arguments.new_track_thresh,
        "track_buffer_seconds": arguments.track_buffer_seconds,
        "min_track_hits": arguments.min_track_hits,
        "reid_window_seconds": arguments.reid_window_seconds,
        "reid_distance_m": arguments.reid_distance_m,
        "reid_max_speed_mps": arguments.reid_max_speed_mps,
        "coordinate_ema_alpha": arguments.coordinate_ema_alpha,
        "trajectory_gap_seconds": arguments.trajectory_gap_seconds,
        "telemetry_offset_frames": arguments.telemetry_offset_frames,
        "rotation_window_seconds": arguments.rotation_window_seconds,
        "max_yaw_rate_deg_s": arguments.max_yaw_rate_deg_s,
    }


def is_rotation(telemetry, telemetry_frame, fps, arguments):
    yaw_rate = yaw_rate_deg_s(
        telemetry, telemetry_frame, fps, arguments.rotation_window_seconds
    )
    return (
        yaw_rate,
        arguments.max_yaw_rate_deg_s > 0 and yaw_rate > arguments.max_yaw_rate_deg_s,
    )


def estimate_obb_length(corners, K, drone_metadata, distortion):
    """Estimate the longest OBB edge by reprojection of its endpoints."""
    from whaledrone_hackathon_code.geo_projection import pixel_to_gps

    projected = []
    for x, y in corners:
        try:
            projected.append(
                pixel_to_gps(
                    x,
                    y,
                    K,
                    drone_metadata["latitude"],
                    drone_metadata["longitude"],
                    drone_metadata["rel_alt"],
                    drone_metadata["gb_yaw"],
                    drone_metadata["gb_pitch"],
                    drone_metadata["gb_roll"],
                    distortion,
                )
            )
        except ValueError:
            return None
    edge_lengths = [
        geographic_distance_m(
            {"latitude": projected[index][0], "longitude": projected[index][1]},
            {
                "latitude": projected[(index + 1) % 4][0],
                "longitude": projected[(index + 1) % 4][1],
            },
        )
        for index in range(4)
    ]
    return float(max(edge_lengths))


def process_video(cache, arguments, telemetry, K_base, distortion):
    metadata = cache.metadata
    fps = metadata["fps"]
    width = metadata["width"]
    height = metadata["height"]
    model = YOLO(str(arguments.model))
    trackers = {
        class_id: make_tracker(arguments, fps) for class_id in arguments.classes
    }
    identity_states = {
        class_id: {
            "raw_tracks": {},
            "last_positions": {},
            "next_track_id": class_id * 1_000_000 + 1,
        }
        for class_id in arguments.classes
    }
    smoothing_state = {}
    capture = cv2.VideoCapture(str(arguments.video))

    try:
        for frame_number in range(metadata["total_frames"]):
            ok, frame = capture.read()
            if not ok:
                break
            telemetry_frame = frame_number + 1 + arguments.telemetry_offset_frames
            drone_metadata = telemetry.get(telemetry_frame)
            if drone_metadata is None:
                cache.update_progress(frame_number + 1, False)
                continue

            yaw_rate, coordinates_paused = is_rotation(
                telemetry, telemetry_frame, fps, arguments
            )
            result = model.predict(
                frame,
                imgsz=1024,
                conf=arguments.confidence,
                iou=0.5,
                classes=arguments.classes,
                device=arguments.device,
                verbose=False,
                agnostic_nms=True,
            )[0]
            obb = result.obb
            tracks = []
            if obb is not None:
                detections = obb.cpu()
                for class_id, tracker in trackers.items():
                    indices = np.flatnonzero(
                        detections.cls.numpy().astype(int) == class_id
                    )
                    if not len(indices):
                        tracker.update(detections[indices], frame)
                        continue
                    for track in tracker.update(detections[indices], frame):
                        class_detection_index = int(track[-1])
                        if 0 <= class_detection_index < len(indices):
                            tracks.append(
                                (track, class_id, int(indices[class_detection_index]))
                            )
            K = zoom_intrinsics(K_base, drone_metadata, (width, height))
            frame_positions = []

            if K is not None and obb is not None and not coordinates_paused:
                active_track_ids_by_class = defaultdict(set)
                for track, class_id, detection_index in tracks:
                    identity_state = identity_states[class_id]
                    if (
                        int(track[5]) in identity_state["raw_tracks"]
                        and identity_state["raw_tracks"][int(track[5])]["track_id"]
                        is not None
                    ):
                        active_track_ids_by_class[class_id].add(
                            identity_state["raw_tracks"][int(track[5])]["track_id"]
                        )
                for track, class_id, detection_index in tracks:
                    identity_state = identity_states[class_id]
                    active_track_ids = active_track_ids_by_class[class_id]
                    if int(track[5]) not in identity_state["raw_tracks"]:
                        identity_state["raw_tracks"][int(track[5])] = {
                            "hits": 0,
                            "track_id": None,
                        }
                    if (
                        int(track[5]) in identity_state["raw_tracks"]
                        and identity_state["raw_tracks"][int(track[5])]["track_id"]
                        is not None
                    ):
                        active_track_ids.add(
                            identity_state["raw_tracks"][int(track[5])]["track_id"]
                        )
                    if detection_index < 0 or detection_index >= len(obb):
                        continue
                    x, y = (
                        obb.xyxy[detection_index]
                        .cpu()
                        .numpy()[:4]
                        .reshape(2, 2)
                        .mean(axis=0)
                    )
                    try:
                        from whaledrone_hackathon_code.geo_projection import (
                            pixel_to_gps,
                        )

                        latitude, longitude = pixel_to_gps(
                            x,
                            y,
                            K,
                            drone_metadata["latitude"],
                            drone_metadata["longitude"],
                            drone_metadata["rel_alt"],
                            drone_metadata["gb_yaw"],
                            drone_metadata["gb_pitch"],
                            drone_metadata["gb_roll"],
                            distortion,
                        )
                    except ValueError:
                        continue
                    corners = obb.xyxyxyxy[detection_index].cpu().numpy()
                    corners /= np.array([width, height])
                    pixel_corners = corners * np.array([width, height])
                    length_m = (
                        estimate_obb_length(
                            pixel_corners, K, drone_metadata, distortion
                        )
                        if class_id == 1
                        else None
                    )
                    position = {
                        "frame": frame_number,
                        "telemetry_frame": telemetry_frame,
                        "time_s": frame_number / fps,
                        "raw_track_id": int(track[5]),
                        "class_id": class_id,
                        "class_name": CLASS_NAMES.get(class_id, str(class_id)),
                        "pixel": [float(x), float(y)],
                        "corners": np.clip(corners, 0.0, 1.0).round(5).tolist(),
                        "latitude": latitude,
                        "longitude": longitude,
                    }
                    if length_m is not None:
                        position["estimated_length_m"] = length_m
                        position["altitude_m"] = drone_metadata["rel_alt"]
                        position["pitch_deg"] = drone_metadata["gb_pitch"]
                        position["distance_from_image_center_px"] = float(
                            np.linalg.norm(
                                np.array([x, y]) - np.array([width / 2, height / 2])
                            )
                        )
                    stable_position = assign_stable_track_id(
                        position, identity_state, active_track_ids, arguments
                    )
                    if stable_position is not None:
                        frame_positions.append(
                            smooth_position(
                                stable_position,
                                smoothing_state,
                                arguments.coordinate_ema_alpha,
                                arguments.trajectory_gap_seconds,
                            )
                        )

            cache.append_positions(frame_positions)
            cache.update_progress(frame_number + 1, coordinates_paused)
            if frame_number % 30 == 0:
                cache.save()
    finally:
        capture.release()


def start_worker(cache, arguments, telemetry, K_base, distortion):
    if not cache.begin():
        return False

    def worker():
        try:
            process_video(cache, arguments, telemetry, K_base, distortion)
        except Exception as error:
            cache.finish(str(error))
            return
        cache.finish()

    threading.Thread(target=worker, name="whale-tracker", daemon=True).start()
    return True


def create_application(arguments):
    metadata = probe_video(arguments.video)
    telemetry = load_telemetry(arguments.srt)
    K_base, distortion = load_intrinsics(
        arguments.calibration, (metadata["width"], metadata["height"])
    )
    cache_path = arguments.cache_dir / f"{arguments.video.stem}_tracking.json"
    cache = TrackingCache(
        cache_path, arguments.video, metadata, pipeline_settings(arguments)
    )

    application = Flask(__name__)
    CORS(application, resources={r"/api/*": {"origins": "*"}})

    @application.get("/api/session")
    def session_status():
        return jsonify(cache.status())

    @application.get("/api/positions")
    def positions():
        cursor = max(0, request.args.get("after", default=0, type=int))
        values, next_cursor = cache.positions_after(cursor)
        return jsonify({"positions": values, "cursor": next_cursor})

    @application.get("/api/frame-state")
    def frame_state():
        frame_number = max(0, request.args.get("frame", default=0, type=int))
        status = cache.status()
        available = (
            status["inference"]["done"]
            or frame_number < status["inference"]["processed_frames"]
        )
        telemetry_frame = frame_number + 1 + arguments.telemetry_offset_frames
        drone_metadata = telemetry.get(telemetry_frame)
        if drone_metadata is None:
            return jsonify(
                {
                    "available": available,
                    "positions": [],
                    "fov": None,
                    "coordinates_paused": False,
                    "yaw_rate_deg_s": 0.0,
                }
            )
        yaw_rate, coordinates_paused = is_rotation(
            telemetry, telemetry_frame, metadata["fps"], arguments
        )
        K = zoom_intrinsics(
            K_base, drone_metadata, (metadata["width"], metadata["height"])
        )
        fov = (
            reproject_fov(
                (metadata["width"], metadata["height"]), K, drone_metadata, distortion
            )
            if K is not None and not coordinates_paused and available
            else None
        )
        return jsonify(
            {
                "available": available,
                "positions": (
                    cache.positions_at_frame(frame_number) if available else []
                ),
                "fov": fov,
                "coordinates_paused": coordinates_paused,
                "yaw_rate_deg_s": yaw_rate,
            }
        )

    @application.get("/api/boat-metrics")
    def boat_metrics():
        target_length = arguments.boat_length_m
        boats = [
            position
            for position in cache.data["positions"]
            if position.get("class_id") == 1 and "estimated_length_m" in position
        ]
        errors = [position["estimated_length_m"] - target_length for position in boats]
        absolute_errors = [abs(error) for error in errors]
        if not errors:
            summary = {
                "count": 0,
                "median_error_m": None,
                "mean_error_m": None,
                "rmse_m": None,
                "signed_bias_m": None,
                "median_absolute_error_m": None,
                "mean_absolute_error_m": None,
            }
        else:
            summary = {
                "count": len(errors),
                "median_error_m": float(np.median(errors)),
                "mean_error_m": float(np.mean(errors)),
                "rmse_m": float(np.sqrt(np.mean(np.square(errors)))),
                "signed_bias_m": float(np.mean(errors)),
                "median_absolute_error_m": float(np.median(absolute_errors)),
                "mean_absolute_error_m": float(np.mean(absolute_errors)),
            }
        return jsonify(
            {"true_length_m": target_length, "summary": summary, "samples": boats}
        )

    @application.get("/api/boat-metrics.csv")
    def boat_metrics_csv():
        target_length = arguments.boat_length_m
        fieldnames = [
            "sequence",
            "frame",
            "time_s",
            "track_id",
            "estimated_length_m",
            "true_length_m",
            "signed_error_m",
            "absolute_error_m",
            "squared_error_m",
            "altitude_m",
            "pitch_deg",
            "distance_from_image_center_px",
        ]
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        with cache.lock:
            for position in cache.data["positions"]:
                if (
                    position.get("class_id") != 1
                    or "estimated_length_m" not in position
                ):
                    continue
                error = position["estimated_length_m"] - target_length
                writer.writerow(
                    {
                        "sequence": position["sequence"],
                        "frame": position["frame"],
                        "time_s": position["time_s"],
                        "track_id": position["track_id"],
                        "estimated_length_m": position["estimated_length_m"],
                        "true_length_m": target_length,
                        "signed_error_m": error,
                        "absolute_error_m": abs(error),
                        "squared_error_m": error**2,
                        "altitude_m": position["altitude_m"],
                        "pitch_deg": position["pitch_deg"],
                        "distance_from_image_center_px": position[
                            "distance_from_image_center_px"
                        ],
                    }
                )
        return send_file(
            BytesIO(output.getvalue().encode("utf-8")),
            mimetype="text/csv",
            as_attachment=True,
            download_name="boat_length_validation.csv",
        )

    @application.get("/api/video")
    def video():
        return send_file(arguments.video, conditional=True, mimetype="video/mp4")

    @application.post("/api/restart")
    def restart():
        if cache.status()["inference"]["running"]:
            return jsonify({"error": "Analysis is already running."}), 409
        cache.reset()
        started = start_worker(cache, arguments, telemetry, K_base, distortion)
        return jsonify({"started": started, **cache.status()})

    start_worker(cache, arguments, telemetry, K_base, distortion)
    return application


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=ROOT / "video.mp4")
    parser.add_argument("--srt", type=Path, default=ROOT / "metadata.srt")
    parser.add_argument("--model", type=Path, default=ROOT / "model.pt")
    parser.add_argument(
        "--calibration",
        type=Path,
        default=ROOT / "camera_calibration_DJIM3T_RGBwide.json",
    )
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "webapp" / "data")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--classes", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--boat-length-m", type=float, default=7.3)
    parser.add_argument("--track-high-thresh", type=float, default=0.4)
    parser.add_argument("--track-low-thresh", type=float, default=0.2)
    parser.add_argument("--new-track-thresh", type=float, default=0.6)
    parser.add_argument("--track-buffer-seconds", type=float, default=8.0)
    parser.add_argument("--min-track-hits", type=int, default=5)
    parser.add_argument("--reid-window-seconds", type=float, default=15.0)
    parser.add_argument("--reid-distance-m", type=float, default=35.0)
    parser.add_argument("--reid-max-speed-mps", type=float, default=5.0)
    parser.add_argument("--coordinate-ema-alpha", type=float, default=0.25)
    parser.add_argument("--trajectory-gap-seconds", type=float, default=0.5)
    parser.add_argument("--telemetry-offset-frames", type=int, default=0)
    parser.add_argument("--rotation-window-seconds", type=float, default=0.25)
    parser.add_argument("--max-yaw-rate-deg-s", type=float, default=5.0)
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    missing = [
        path
        for path in (
            arguments.video,
            arguments.srt,
            arguments.model,
            arguments.calibration,
        )
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing input files:\n" + "\n".join(f"  - {path}" for path in missing)
        )
    application = create_application(arguments)
    print(f"Whale tracking API: http://{arguments.host}:{arguments.port}")
    application.run(
        host=arguments.host, port=arguments.port, threaded=True, debug=False
    )


if __name__ == "__main__":
    main()
