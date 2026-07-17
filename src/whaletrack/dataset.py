

import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

DATASET = Path("../whale_drone_hack/WhaleDrone_Hackathon_dataset")
CALIB = DATASET / "camera_calibration_DJIM3T_RGBwide.json"
FRAME = (3840, 2160)
CLASSES = ("whale", "boat")
ASPECT = {"whale": 0.25, "boat": 0.30}
NUMERIC = ["frame", "height", "yaw", "pitch", "roll", "length", "image_x", "image_y"]
_TIME = re.compile(r"DJI_(\d{14})_\d+_V\.MP4", re.IGNORECASE)


def load_intrinsics(path=CALIB): # load the camera calibration
    data = json.loads(path.read_text())
    m = data["camera_matrix"]
    k = np.array([[m["fx"], 0, m["cx"]], [0, m["fy"], m["cy"]], [0, 0, 1.0]])
    return k, np.array(data["distortion_coefficients"]["vector_opencv_order"])


def class_of(name):
    name = str(name).upper()
    if "DOLPHIN" in name:
        return None
    return 1 if name.startswith("BOAT") else 0


def project_endpoints(row, k, dist):
    """Tail and rostrum UTM endpoints projected to pixels, for any camera tilt."""
    rot = Rotation.from_euler(
        "ZXZ", [-row["roll"], 90 - row["pitch"], row["yaw"]], degrees=True
    ).as_matrix()
    centroid = np.array([[[row["image_x"], row["image_y"]]]], np.float32)
    ray = rot.T @ np.append(cv2.undistortPoints(centroid, k, dist)[0, 0], 1.0) # apply the correction here 
    tvec = -rot @ (row["height"] / ray[2] * ray)  # camera center vs the centroid
    world = np.array(
        [
            [row["start_east"], row["start_north"], 0.0],
            [row["end_east"], row["end_north"], 0.0],
        ]
    ) - [row["east"], row["north"], 0.0] # we have the word coordinations 
    pixels, _ = cv2.projectPoints(world, cv2.Rodrigues(rot)[0], tvec, k, dist) # project it to image coords 
    return pixels.reshape(-1, 2)


def obb_label(row, k, dist):
    """One YOLO-OBB line (class + 4 normalized corners), or None to skip."""
    cls = class_of(row["name"])
    if cls is None or pd.isna(row["start_east"]) or pd.isna(row["end_east"]):
        return None
    tail, rostrum = project_endpoints(row, k, dist)
    axis = rostrum - tail
    length = float(np.linalg.norm(axis))
    if length == 0:
        return None
    normal = (
        np.array([-axis[1], axis[0]]) / length * (ASPECT[CLASSES[cls]] * length / 2)
    )
    corners = (
        np.array([tail + normal, rostrum + normal, rostrum - normal, tail - normal])
        / FRAME
    )
    if (corners < -0.2).any() or (corners > 1.2).any():
        return None
    return f"{cls} " + " ".join(f"{v:.6f}" for v in np.clip(corners, 0, 1).reshape(-1))


def segment_start(mp4):
    match = _TIME.search(mp4.name)
    if match is None:
        raise ValueError(f"Unexpected segment filename: {mp4.name}")
    return datetime.strptime(match.group(1), "%Y%m%d%H%M%S")


def load_flight(csv_path, flight_dir):
    """Vector rows for one flight, each tagged with its DJI video segment path."""
    df = pd.read_csv(csv_path)
    df["time"] = pd.to_datetime(df["time"], format="mixed")
    df[NUMERIC] = df[NUMERIC].apply(pd.to_numeric, errors="coerce")
    df = df[df["length"].notna()].copy() # filter the rows that do not have length because we wont have start and end
    segments = sorted(flight_dir.glob("DJI_*_V.MP4"), key=segment_start)
    clip_cet = df.groupby("video")["time"].transform("min") + timedelta(hours=1)
    df["segment"] = clip_cet.map(
        lambda t: max(
            (s for s in segments if segment_start(s) <= t),
            key=segment_start,
            default=segments[0],
        )
    )
    return df


def find_closest_time_row(df, target_datetime, time_col='time', offset_s=3600):
    """
    Trouve la ligne du DataFrame `df` où la colonne 'time_col' est la plus proche de target_datetime.

    Parameters:
    - df: DataFrame pandas contenant une colonne de datetime
    - target_datetime: datetime à comparer
    - time_col: nom de la colonne contenant les datetime (par défaut 'time')

    Returns:
    - La ligne du DataFrame la plus proche en temps
    """
    target_datetime += timedelta(seconds=offset_s)
    time_series = pd.to_datetime(df[time_col])
    time_diffs = (time_series - target_datetime).abs()
    closest_idx = time_diffs.idxmin()

    return df.loc[closest_idx]




def read_frame(video, number):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(number))
    ok, image = cap.read()
    cap.release()
    return image if ok else None


