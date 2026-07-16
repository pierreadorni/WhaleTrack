#!/usr/bin/env python3
"""Track YOLO-OBB whale detections and reproject their centres to GPS."""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
from ultralytics.trackers import BOTSORT

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "WhaleDrone_Hackathon_code" / "src"))

from whaledrone_hackathon_code.geo_projection import pixel_to_gps


TELEMETRY_PATTERN = re.compile(
    r"FrameCnt:\s*(?P<frame>\d+).*?"
    r"focal_len:\s*(?P<focal_len>[\d.]+).*?"
    r"dzoom_ratio:\s*(?P<dzoom_ratio>[\d.]+).*?"
    r"latitude:\s*(?P<latitude>-?[\d.]+).*?"
    r"longitude:\s*(?P<longitude>-?[\d.]+).*?"
    r"rel_alt:\s*(?P<rel_alt>-?[\d.]+).*?"
    r"gb_yaw:\s*(?P<gb_yaw>-?[\d.]+)\s+"
    r"gb_pitch:\s*(?P<gb_pitch>-?[\d.]+)\s+"
    r"gb_roll:\s*(?P<gb_roll>-?[\d.]+)",
    re.DOTALL,
)


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
    parser.add_argument("--device", default="mps", help="YOLO device, e.g. mps, cpu, 0")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--classes", type=int, nargs="+", default=[0], help="YOLO classes to track; 0 is whale")
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
    parser.add_argument("--save-during-rotation", action="store_true")
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--preview-every", type=float, default=2.0)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "whale_positions.jsonl")
    return parser.parse_args()


def require_files(arguments):
    missing = [path for path in (arguments.video, arguments.srt, arguments.model, arguments.calibration) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing input files:\n" + "\n".join(f"  - {path}" for path in missing))
    if not 0 < arguments.coordinate_ema_alpha <= 1:
        raise ValueError("--coordinate-ema-alpha must be in (0, 1].")
    if arguments.trajectory_gap_seconds <= 0 or arguments.rotation_window_seconds <= 0:
        raise ValueError("Trajectory and rotation time windows must be positive.")


def load_telemetry(srt_path):
    telemetry = {}
    for match in TELEMETRY_PATTERN.finditer(srt_path.read_text(encoding="utf-8", errors="replace")):
        values = match.groupdict()
        frame_number = int(values.pop("frame"))
        telemetry[frame_number] = {key: float(value) for key, value in values.items()}
    if not telemetry:
        raise ValueError(f"No DJI telemetry records found in {srt_path}")
    return telemetry


def load_intrinsics(calibration_path, image_size):
    calibration = json.loads(calibration_path.read_text())
    matrix = calibration["camera_matrix"]
    K = np.array([[matrix["fx"], 0, matrix["cx"]], [0, matrix["fy"], matrix["cy"]], [0, 0, 1.0]])
    calibration_size = calibration["image_resolution"]
    scale_x = image_size[0] / calibration_size["width_px"]
    scale_y = image_size[1] / calibration_size["height_px"]
    K[0, :] *= scale_x
    K[1, :] *= scale_y
    distortion = np.array(calibration["distortion_coefficients"]["vector_opencv_order"])
    return K, distortion


def zoom_intrinsics(K_base, telemetry, image_size):
    zoom = telemetry["dzoom_ratio"]
    base_focal_length = telemetry["focal_len"] / zoom
    if abs(base_focal_length - 162.0) < abs(base_focal_length - 24.0):
        return None
    K = K_base.copy()
    centre_x, centre_y = image_size[0] / 2, image_size[1] / 2
    K[0, 0] *= zoom
    K[1, 1] *= zoom
    K[0, 2] = centre_x + (K_base[0, 2] - centre_x) * zoom
    K[1, 2] = centre_y + (K_base[1, 2] - centre_y) * zoom
    return K


def make_tracker(arguments, fps):
    return BOTSORT(
        SimpleNamespace(
            tracker_type="botsort", track_high_thresh=arguments.track_high_thresh,
            track_low_thresh=arguments.track_low_thresh,
            new_track_thresh=arguments.new_track_thresh,
            track_buffer=max(1, round(arguments.track_buffer_seconds * fps)),
            match_thresh=0.8, fuse_score=True,
            gmc_method="sparseOptFlow", proximity_thresh=0.5, appearance_thresh=0.8,
            with_reid=False, model="auto",
        )
    )


def geographic_distance_m(first, second):
    earth_radius_m = 6_371_008.8
    first_latitude, first_longitude = np.radians([first["latitude"], first["longitude"]])
    second_latitude, second_longitude = np.radians([second["latitude"], second["longitude"]])
    delta_latitude = second_latitude - first_latitude
    delta_longitude = second_longitude - first_longitude
    haversine = np.sin(delta_latitude / 2) ** 2 + np.cos(first_latitude) * np.cos(second_latitude) * np.sin(delta_longitude / 2) ** 2
    return float(2 * earth_radius_m * np.arcsin(np.sqrt(haversine)))


def angular_difference_degrees(start_degrees, end_degrees):
    return (end_degrees - start_degrees + 180) % 360 - 180


def yaw_rate_deg_s(telemetry, frame_number, fps, window_seconds):
    previous_frame = frame_number - max(1, round(window_seconds * fps))
    previous = telemetry.get(previous_frame)
    current = telemetry.get(frame_number)
    if previous is None or current is None:
        return 0.0
    return abs(angular_difference_degrees(previous["gb_yaw"], current["gb_yaw"])) * fps / (frame_number - previous_frame)


def smooth_position(position, smoothing_state, alpha, reset_gap_seconds):
    track_id = position["track_id"]
    position["raw_latitude"] = position["latitude"]
    position["raw_longitude"] = position["longitude"]
    previous = smoothing_state.get(track_id)
    if previous is None or position["time_s"] - previous["time_s"] > reset_gap_seconds:
        smoothed_latitude = position["latitude"]
        smoothed_longitude = position["longitude"]
    else:
        smoothed_latitude = alpha * position["latitude"] + (1 - alpha) * previous["latitude"]
        smoothed_longitude = alpha * position["longitude"] + (1 - alpha) * previous["longitude"]
    position["latitude"] = smoothed_latitude
    position["longitude"] = smoothed_longitude
    smoothing_state[track_id] = {"latitude": smoothed_latitude, "longitude": smoothed_longitude, "time_s": position["time_s"]}
    return position


def assign_stable_track_id(position, identity_state, active_track_ids, arguments):
    raw_track_id = position["raw_track_id"]
    raw_track = identity_state["raw_tracks"].setdefault(raw_track_id, {"hits": 0, "track_id": None})
    raw_track["hits"] += 1
    if raw_track["track_id"] is None and raw_track["hits"] >= arguments.min_track_hits:
        eligible_tracks = []
        for track_id, last_position in identity_state["last_positions"].items():
            elapsed_s = position["time_s"] - last_position["time_s"]
            if track_id in active_track_ids or not 0 < elapsed_s <= arguments.reid_window_seconds:
                continue
            distance_m = geographic_distance_m(last_position, position)
            maximum_distance_m = arguments.reid_distance_m + arguments.reid_max_speed_mps * elapsed_s
            if distance_m <= maximum_distance_m:
                eligible_tracks.append((distance_m, track_id))
        if eligible_tracks:
            raw_track["track_id"] = min(eligible_tracks)[1]
        else:
            raw_track["track_id"] = identity_state["next_track_id"]
            identity_state["next_track_id"] += 1
    if raw_track["track_id"] is None:
        return None
    position["track_id"] = raw_track["track_id"]
    identity_state["last_positions"][position["track_id"]] = position
    active_track_ids.add(position["track_id"])
    return position


def draw_preview(frame, detections, positions):
    preview = frame.copy()
    for detection in detections:
        corners = detection.xyxyxyxy.squeeze().cpu().numpy().astype(np.int32)
        cv2.polylines(preview, [corners], True, (255, 120, 0), 2)
    for position in positions:
        x, y = position["pixel"]
        cv2.circle(preview, (round(x), round(y)), 7, (0, 220, 0), -1)
        cv2.putText(preview, f'ID {position["track_id"]}', (round(x) + 8, round(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
    return preview


def reproject_fov(frame_size, K, metadata, distortion):
    width, height = frame_size
    corners = ((0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1))
    try:
        return [
            pixel_to_gps(
                x, y, K, metadata["latitude"], metadata["longitude"], metadata["rel_alt"],
                metadata["gb_yaw"], metadata["gb_pitch"], metadata["gb_roll"], distortion,
            )
            for x, y in corners
        ]
    except ValueError:
        return None


def show_preview(frame, history, fov_polygon, yaw_rate, coordinates_paused, trajectory_gap_seconds, preview_state):
    if preview_state is None:
        plt.ion()
        figure, (video_axis, map_axis) = plt.subplots(1, 2, figsize=(16, 7))
        preview_state = {
            "figure": figure,
            "video_axis": video_axis,
            "map_axis": map_axis,
            "video_image": video_axis.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
        }
        video_axis.set_axis_off()
        figure.tight_layout()
        plt.show(block=False)
    else:
        preview_state["video_image"].set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    map_axis = preview_state["map_axis"]
    map_axis.clear()
    if fov_polygon is not None:
        fov_latitudes, fov_longitudes = zip(*fov_polygon)
        map_axis.fill(
            [*fov_longitudes, fov_longitudes[0]],
            [*fov_latitudes, fov_latitudes[0]],
            color="#4c78a8",
            alpha=0.2,
            label="Drone FOV",
            zorder=1,
        )
        map_axis.plot(
            [*fov_longitudes, fov_longitudes[0]],
            [*fov_latitudes, fov_latitudes[0]],
            color="#4c78a8",
            linewidth=1.5,
            zorder=2,
        )
    colours = plt.get_cmap("tab10")
    for colour_index, (track_id, positions) in enumerate(sorted(history.items())):
        colour = colours(colour_index % colours.N)
        segments = [[]]
        for position in positions:
            if segments[-1] and position["time_s"] - segments[-1][-1]["time_s"] > trajectory_gap_seconds:
                segments.append([])
            segments[-1].append(position)
        for segment_index, segment in enumerate(segments):
            longitudes = [position["longitude"] for position in segment]
            latitudes = [position["latitude"] for position in segment]
            map_axis.plot(longitudes, latitudes, "-o", color=colour, linewidth=1.5, markersize=3, label=f"ID {track_id}" if segment_index == 0 else None)
        longitudes = [position["longitude"] for position in positions]
        latitudes = [position["latitude"] for position in positions]
        map_axis.scatter(longitudes[-1], latitudes[-1], color=colour, edgecolors="black", linewidths=0.7, s=70, zorder=3)
        map_axis.annotate(f"ID {track_id}", (longitudes[-1], latitudes[-1]), xytext=(6, 6), textcoords="offset points")
    video_title = "YOLO OBB tracks"
    if coordinates_paused:
        video_title += f" - GPS paused, yaw {yaw_rate:.1f} deg/s"
    preview_state["video_axis"].set_title(video_title)
    map_axis.set_xlabel("Longitude")
    map_axis.set_ylabel("Latitude")
    map_axis.set_title("Reprojected whale track history" if not coordinates_paused else "Reprojected whale track history - GPS paused during rotation")
    map_axis.ticklabel_format(useOffset=False)
    map_axis.margins(0.1)
    if history:
        map_axis.legend(loc="best")
    preview_state["figure"].canvas.draw_idle()
    preview_state["figure"].canvas.flush_events()
    plt.pause(0.001)
    return preview_state


def main():
    arguments = parse_arguments()
    require_files(arguments)
    telemetry = load_telemetry(arguments.srt)
    capture = cv2.VideoCapture(str(arguments.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {arguments.video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    K_base, distortion = load_intrinsics(arguments.calibration, (width, height))
    model = YOLO(str(arguments.model))
    tracker = make_tracker(arguments, fps)
    preview_interval = max(1, round(arguments.preview_every * fps))
    max_frames = int(arguments.max_seconds * fps) if arguments.max_seconds else None
    source_frame = round(arguments.start_seconds * fps)
    capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
    history = defaultdict(list)
    identity_state = {"raw_tracks": {}, "last_positions": {}, "next_track_id": 1}
    smoothing_state = {}
    preview_state = None
    paused_coordinate_frames = 0

    with arguments.output.open("w", encoding="utf-8") as output:
        processed_frames = 0
        while max_frames is None or processed_frames < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            source_frame += 1
            processed_frames += 1
            telemetry_frame = source_frame + arguments.telemetry_offset_frames
            metadata = telemetry.get(telemetry_frame)
            if metadata is None:
                continue
            current_yaw_rate = yaw_rate_deg_s(telemetry, telemetry_frame, fps, arguments.rotation_window_seconds)
            coordinates_paused = (
                not arguments.save_during_rotation
                and arguments.max_yaw_rate_deg_s > 0
                and current_yaw_rate > arguments.max_yaw_rate_deg_s
            )
            paused_coordinate_frames += coordinates_paused
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
            detections = obb.cpu() if obb is not None else ()
            tracks = tracker.update(detections, frame) if obb is not None else ()
            K = zoom_intrinsics(K_base, metadata, (width, height))
            positions = []
            if K is not None and obb is not None and not coordinates_paused:
                active_track_ids = {
                    identity_state["raw_tracks"][int(track[5])]["track_id"]
                    for track in tracks
                    if int(track[5]) in identity_state["raw_tracks"]
                    and identity_state["raw_tracks"][int(track[5])]["track_id"] is not None
                }
                for track in tracks:
                    detection_index = int(track[-1])
                    if detection_index < 0 or detection_index >= len(obb):
                        continue
                    x, y = obb.xyxy[detection_index].cpu().numpy()[:4].reshape(2, 2).mean(axis=0)
                    try:
                        latitude, longitude = pixel_to_gps(x, y, K, metadata["latitude"], metadata["longitude"], metadata["rel_alt"], metadata["gb_yaw"], metadata["gb_pitch"], metadata["gb_roll"], distortion)
                    except ValueError:
                        continue
                    position = {
                        "frame": source_frame,
                        "telemetry_frame": telemetry_frame,
                        "time_s": source_frame / fps,
                        "raw_track_id": int(track[5]),
                        "pixel": [float(x), float(y)],
                        "latitude": latitude,
                        "longitude": longitude,
                    }
                    stable_position = assign_stable_track_id(position, identity_state, active_track_ids, arguments)
                    if stable_position is not None:
                        smoothed_position = smooth_position(stable_position, smoothing_state, arguments.coordinate_ema_alpha, arguments.trajectory_gap_seconds)
                        positions.append(smoothed_position)
                        history[smoothed_position["track_id"]].append(smoothed_position)
                        output.write(json.dumps(smoothed_position) + "\n")
            if processed_frames % preview_interval == 0:
                preview = draw_preview(frame, detections, positions)
                fov_polygon = reproject_fov((width, height), K, metadata, distortion) if K is not None and not coordinates_paused else None
                preview_state = show_preview(preview, history, fov_polygon, current_yaw_rate, coordinates_paused, arguments.trajectory_gap_seconds, preview_state)
                print(f"frame={source_frame} yaw_rate={current_yaw_rate:.1f} active_positions={len(positions)} tracked_ids={len(history)}")
    capture.release()
    if preview_state is not None:
        plt.close(preview_state["figure"])
    print(f"Wrote {sum(map(len, history.values()))} positions for {len(history)} tracks to {arguments.output}; GPS paused for {paused_coordinate_frames} frames during rotation")


if __name__ == "__main__":
    main()