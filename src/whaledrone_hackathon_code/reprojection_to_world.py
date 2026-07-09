from asyncio.log import logger

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional

import cv2
import json
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from whaledrone_hackathon_code.dataset import load_flight

ROOT_DIR = Path("/share/projects/whale_drone_hack")

ROOT_DIR_DATASET = ROOT_DIR / "WhaleDrone_Hackathon_dataset/Annotated"
ROOT_DIR_METADATA = ROOT_DIR / "Whale_Group2/Whale/Annotated_csv"
ALL_FLIGHTS = [d for d in ROOT_DIR_DATASET.iterdir() if d.is_dir()]
CALIB = ROOT_DIR / "WhaleDrone_Hackathon_dataset/camera_calibration_DJIM3T_RGBwide.json"

print(f"Found {len(ALL_FLIGHTS)} flights in the dataset.")

DATA_JSON: list[dict[str, dict[str, dict[str, Any]]]] = []

for flight_dir in ALL_FLIGHTS:
    flight_day = flight_dir.name
    video_files = sorted(flight_dir.glob("*.MP4"))
    srt_files = sorted(flight_dir.glob("*.SRT"))
    csv_files = sorted(flight_dir.glob("*.csv"))

    print(f"\nFlight: {flight_day}")
    print(f"  Video files: {len(video_files)}")
    print(f"  SRT files: {len(srt_files)}")
    print(f"  CSV files: {len(csv_files)}")

    flight_info_dict = {
        "flight_day": flight_day,
        "annotations_file": "",
        "video_files": [],
    }

    if len(video_files) > 0 and len(srt_files) == len(video_files):
        for video_file in video_files:
            flight_name = video_file.stem
            srt_file = video_file.with_suffix(".SRT")
            path_metadata = (
                ROOT_DIR_METADATA / flight_day / f"{flight_name}_telemetry.csv"
            )

            if not srt_file.exists():
                print(f"  Missing SRT for {video_file.name}")
                continue

            if not path_metadata.exists():
                print(f"  Missing metadata for {video_file.name}: {path_metadata}")
                continue

            flight_info_dict["video_files"].append(
                {
                    "video_file": video_file.name,
                    "metadata_file": path_metadata.name,
                    "srt_file": srt_file.name,
                }
            )
            """DATA_JSON.append(
                {"flight_day": flight_day, 
                "annotations_file : "",
                "video_files" : [flight_name] = {
                "video_file": video_file,
                "metadata_file": str(path_metadata
                )"""

    DATA_JSON.append(flight_info_dict)

# print("\nDATA_JSON structure:")
# for flight_info_dict in DATA_JSON:
#     print(f"Flight Day: {flight_info_dict['flight_day']}")
#     for video_info_dict in flight_info_dict["video_files"]:
#         # print(f"  Video Acquisition: {video_info_dict["video_}")
#         print(f"     Video file: {video_info_dict['video_file']}")
#         print(f"     Metadata file: {video_info_dict['metadata_file']}")


def load_intrinsics(path=CALIB):  # load the camera calibration
    data = json.loads(path.read_text())
    m = data["camera_matrix"]
    k = np.array([[m["fx"], 0, m["cx"]], [0, m["fy"], m["cy"]], [0, 0, 1.0]])
    return k, np.array(data["distortion_coefficients"]["vector_opencv_order"])


# read the video per frame
def read_frame(video, number):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(number))
    ok, image = cap.read()
    cap.release()
    return image if ok else None


WIDE_BASE_FL = 24.0
TELE_BASE_FL = 162.0
FRAME = (3840, 2160)


def zoom_intrinsics(k_base, focal_len, zoom_ratio=None):
    """
    See ZOOM_CORRECTION_REPORT.md
    return k_corrected and IS_WIDE_CAMERA falg (boolean)
    """
    dzoom = zoom_ratio if zoom_ratio is not None else focal_len / WIDE_BASE_FL

    base_fl = focal_len / dzoom

    if abs(base_fl - TELE_BASE_FL) < abs(base_fl - WIDE_BASE_FL):
        return k_base, False  # tele camera — no compatible calibration

    z = dzoom
    k = k_base.copy()
    k[0, 0] = k_base[0, 0] * z
    k[1, 1] = k_base[1, 1] * z
    k[0, 2] = FRAME[0] / 2 + (k_base[0, 2] - FRAME[0] / 2) * z
    k[1, 2] = FRAME[1] / 2 + (k_base[1, 2] - FRAME[1] / 2) * z
    return k, True


def pixel_to_gps(
    u,
    v,
    K,
    drone_lat,
    drone_lon,
    altitude_agl,
    yaw,
    pitch,
    roll,
    dist_coeffs=None,
    degrees=True,
    debug=False
):
    """
    Convertit un pixel image (u, v) en latitude/longitude,
    en supposant un sol plat et une altitude AGL connue.

    Parameters
    ----------
    u, v : float
        Coordonnées pixel de l'objet.
    K : np.ndarray, shape (3, 3)
        Matrice intrinsèque caméra.
    drone_lat, drone_lon : float
        Position GPS du drone.
    altitude_agl : float
        Hauteur de la caméra au-dessus du sol, en mètres.
    yaw, pitch, roll : float
        Orientation de la caméra.
    dist_coeffs : optionnel
        Coefficients de distorsion, non utilisés ici.
    degrees : bool
        Angles en degrés si True.

    Returns
    -------
    lat_obj, lon_obj : float
        Position géographique estimée de l'objet.
    """

 
    if debug:
        print(
            f"x={u}\n"
            f"y={v}\n"
            f"drone_lat : {drone_lat}\n"
            f"drone_lon={drone_lon}\n"
            f"altitude_agl={altitude_agl}\n"
            f"yaw={yaw}\n"
            f"pitch={pitch}\n"
            f"roll={roll}"
        )

    # 1. Repère local ENU centré sur le drone
    # E = x vers l'est, N = y vers le nord, U = z vers le haut
    C = np.array([0.0, 0.0, altitude_agl])

    # 2. Pixel homogène
    if dist_coeffs is not None:
        undistorted = cv2.undistortPoints(
            np.array([[[u, v]]], dtype=np.float32), K, dist_coeffs, P=K
        )

        u_corr = undistorted[0, 0, 0]
        v_corr = undistorted[0, 0, 1]
    else:
        u_corr = u
        v_corr = v
    pixel = np.array([u_corr, v_corr, 1.0])
    print(pixel)

    # 3. Rayon dans le repère caméra
    ray_cam = np.linalg.inv(K) @ pixel
    ray_cam = ray_cam / np.linalg.norm(ray_cam)

    # 4. Rotation caméra -> monde local
    R_cam_to_world = R.from_euler(
        "ZYX", [yaw, pitch, roll], degrees=degrees
    ).as_matrix()


    ray_world = R_cam_to_world @ ray_cam
    ray_world = ray_world / np.linalg.norm(ray_world)

    if debug:
        print("Rotation Matrix : ", R_cam_to_world)
        print("Ray cam : ", ray_cam)
        print("Ray world : ", ray_world)

    # 5. Intersection avec le sol z = 0
    if abs(ray_world[2]) < 1e-8:
        raise ValueError("Le rayon est parallèle au sol.")

    t = -C[2] / ray_world[2]

    if t < 0:
        raise ValueError("Le rayon ne coupe pas le sol devant la caméra.")

    P = C + t * ray_world

    if debug:
        print("P:", P)

    east_offset = P[0] # x
    north_offset = P[1] # y

    # 6. Convertir offset local ENU -> lat/lon
    # Approximation locale valable pour petites distances
    R_earth = 6378137.0  # Rayon de la Terre en mètres
    f = 1/298.257223563  # Aplatissement

    lat_rad = np.radians(drone_lat)
    lon_rad = np.radians(drone_lon)

    # Calcul de la nouvelle latitude
    M = R_earth * (1 - f) / (1 - f * np.sin(lat_rad))**2
    lat_ob_rad = lat_rad + north_offset / M

    # Calcul de la nouvelle longitude
    N = R_earth / np.sqrt(1 - f * np.sin(lat_ob_rad)**2)
    lon_obj_rad = lon_rad + east_offset / (N * np.cos(lat_ob_rad))

    #dlat = north_offset / R_earth
    #dlon = east_offset / (R_earth * np.cos(np.radians(drone_lat)))

    lat_obj = np.degrees(lat_ob_rad)
    lon_obj = np.degrees(lon_obj_rad)
    if debug:
        print(f"Résultat: lat={lat_obj:.8f}, lon={lon_obj:.8f}")
        print(f"Différence avec drone: lat_diff={abs(lat_obj - drone_lat):.8f}, lon_diff={abs(lon_obj - drone_lon):.8f}")

    return lat_obj, lon_obj


def show_frame_with_click(image, frame_number, grid_step=100):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w = image_rgb.shape[:2]

    fig, ax = plt.subplots(figsize=(15, 14))
    ax.imshow(image_rgb, origin="upper", extent=(0, w, h, 0))

    ax.set_title(f"Frame {frame_number} - clique sur l'image")
    ax.set_xlabel("x / colonne pixel")
    ax.set_ylabel("y / ligne pixel")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    ax.set_xticks(range(0, w + 1, grid_step))
    ax.set_yticks(range(0, h + 1, grid_step))
    ax.tick_params(axis="x", labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")

    ax.grid(color="yellow", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    text = ax.text(
        10, 30, "", color="yellow", fontsize=12, bbox=dict(facecolor="black", alpha=0.6)
    )
    selected_point = [None]

    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return

        x = int(event.xdata)
        y = int(event.ydata)

        pix_gps = pixel_to_gps(
            u=x,  # Example pixel coordinates
            v=y,
            K=k,
            drone_lat=row["latitude"],
            drone_lon=row["longitude"],
            altitude_agl=row["rel_alt"],
            yaw=row["gb_yaw"],
            pitch=row["gb_pitch"],
            roll=row["gb_roll"],
            dist_coeffs=dist_coeffs,
            degrees=False,
        )

        print(f"Estimated GPS coordinates for pixel ({x}, {y}): {pix_gps}\n")

        print(f"Clicked pixel coordinates: x={x}, y={y}")

        text.set_text(f"x={x}, y={y}; Lat={pix_gps[0]:.8f}, Lon={pix_gps[1]:.8f}")
        if selected_point[0] is not None:
            selected_point[0].remove()
        selected_point[0] = ax.plot(x, y, "ro")[0]
        fig.canvas.draw()

    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.tight_layout()
    plt.show()


def save_frame_with_fixed_point(
    image,
    frame_number,
    x,
    y,
    row,
    k,
    dist_coeffs=None,
    grid_step=100,
    output_dir="frames_with_gps_points",
):
    """
    Affiche/sauvegarde une frame avec un point pixel fixe et calcule sa position GPS.

    Parameters
    ----------
    image : np.ndarray
        Image OpenCV en BGR.
    frame_number : int
        Numéro de la frame.
    x, y : int
        Coordonnées pixel du point fixe.
    row : dict-like
        Ligne contenant latitude, longitude, rel_alt, gb_yaw, gb_pitch, gb_roll.
    k : np.ndarray
        Matrice intrinsèque caméra.
    dist_coeffs : np.ndarray or None
        Coefficients de distorsion.
    grid_step : int
        Espacement de la grille.
    output_dir : str
        Dossier de sauvegarde.

    Returns
    -------
    pix_gps : tuple
        Coordonnées GPS estimées : (lat, lon).
    output_path : pathlib.Path
        Chemin de l'image sauvegardée.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]

    if not (0 <= x < w and 0 <= y < h):
        raise ValueError(
            f"Point pixel invalide: x={x}, y={y}. "
            f"L'image a une taille w={w}, h={h}."
        )

    # pix_gps = pixel_to_gps(
    #     u=x,
    #     v=y,
    #     K=k,
    #     drone_lat=row["latitude"],
    #     drone_lon=row["longitude"],
    #     altitude_agl=row["rel_alt"],
    #     yaw=row["gb_yaw"],
    #     pitch=row["gb_pitch"],
    #     roll=row["gb_roll"],
    #     dist_coeffs=dist_coeffs,
    #     degrees=False,
    # )

    fov = Fov()
    fov.set_image_size(w, h)
    fov.set_camera_params(
        camera_matrix=k,
        dist_coefficients=dist_coeffs,
        horizontal_fov=65.76,
        vertical_fov=39.96,
    )
    
    pix_gps = fov.get_gps_point(
        image_point=(x, y),
        drone_height=row["rel_alt"],
        yaw_pitch_roll=(row["gb_yaw"], row["gb_pitch"], row["gb_roll"]),
        pos=(row["latitude"], row["longitude"]),
    )

    fig, ax = plt.subplots(figsize=(15, 14))
    ax.imshow(image_rgb, origin="upper", extent=(0, w, h, 0))

    ax.set_title(f"Frame {frame_number} - point fixe")
    ax.set_xlabel("x / colonne pixel")
    ax.set_ylabel("y / ligne pixel")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    ax.set_xticks(range(0, w + 1, grid_step))
    ax.set_yticks(range(0, h + 1, grid_step))
    ax.tick_params(axis="x", labelrotation=45)

    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")

    ax.grid(color="yellow", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    ax.plot(x, y, "ro", markersize=8)

    ax.text(
        10,
        30,
        f"x={x}, y={y}; Lat={pix_gps[0]:.8f}, Lon={pix_gps[1]:.8f}",
        color="yellow",
        fontsize=12,
        bbox=dict(facecolor="black", alpha=0.6),
    )

    fig.tight_layout()

    output_path = output_dir / f"frame_{frame_number:06d}_x{x}_y{y}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Pixel: x={x}, y={y}")
    print(f"Estimated GPS: lat={pix_gps[0]:.8f}, lon={pix_gps[1]:.8f}")
    print(f"Image sauvegardée dans: {output_path}")

    return pix_gps, output_path


def get_sub_info_dict(
    data_json_dict: List[Dict[str, Any]],
    key: str,
    value: str,
) -> Optional[Dict[str, Any]]:
    """
    Trouve le premier dictionnaire dans une liste où la clé spécifiée correspond à la valeur donnée.

    Args:
        liste_dicts: Liste de dictionnaires à parcourir.
        cle: Clé à rechercher dans les dictionnaires.
        valeur: Valeur attendue pour la clé.

    Returns:
        Le premier dictionnaire correspondant, ou None si aucun n'est trouvé.
    """
    for info_dict in data_json_dict:
        if info_dict[key] == value:
            return info_dict
    return None


def get_telemetry_file(flight_day, video_segment):
    flight_info_dict = get_sub_info_dict(
        data_json_dict=DATA_JSON, key="flight_day", value=flight_day
    )

    video_info_dict = get_sub_info_dict(
        data_json_dict=flight_info_dict["video_files"],
        key="video_file",
        value=video_segment,
    )

    meta_data = video_info_dict["metadata_file"]
    metadata_path = ROOT_DIR_METADATA / flight_day / meta_data
    return metadata_path


import logging
from collections import defaultdict
from typing import Any

import cv2
import numpy as np
import utm

FRAME = (3840, 2160)

def shift(seq: list[Any], n: int) -> list[Any]:
    return seq[n:] + seq[:n]

class Fov:
    def __init__(self) -> None:
        logger.debug(f"Creating instance of Fov {self}")
        self.image_size: tuple[int, int] # does that corresponds to the image size of the camera? frame?
        self.horizontal_fov: float
        self.vertical_fov: float
        self.camera_matrix: np.ndarray
        self.dist_coefficients: np.ndarray

    def set_image_size(self, width: int, height: int) -> None:
        self.image_size = (width, height)

    def set_camera_params(
        self,
        camera_matrix: np.ndarray,
        dist_coefficients: np.ndarray,
        horizontal_fov: float,
        vertical_fov: float,
        n_images: int | None = None,
        stds: list[float] | None = None,
    ) -> None:
        logger.debug("Setting camera params")
        self.camera_matrix = camera_matrix
        self.dist_coefficients = dist_coefficients
        self.horizontal_fov = horizontal_fov * np.pi / 180
        self.vertical_fov = vertical_fov * np.pi / 180

    @staticmethod
    def roll(roll: float) -> np.ndarray:
        return np.array(
            [
                [np.cos(roll), 0, np.sin(roll)],
                [0, 1, 0],
                [-np.sin(roll), 0, np.cos(roll)],
            ]
        )

    @staticmethod
    def pitch(pitch: float) -> np.ndarray:
        return np.array(
            [
                [1, 0, 0],
                [0, np.cos(pitch), -np.sin(pitch)],
                [0, np.sin(pitch), np.cos(pitch)],
            ]
        )

    @staticmethod
    def yaw(yaw: float) -> np.ndarray:
        return np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])

    def rotation(self, yaw: float, pitch: float, roll: float) -> np.ndarray:
        return np.matmul(self.yaw(yaw), np.matmul(self.pitch(pitch), self.roll(roll)))

    def get_unit_vector(self, image_point: tuple[float, float]) -> np.ndarray:
        if self.camera_matrix is not None:
            undist_point = cv2.undistortPoints(
                np.array([[image_point]], dtype=np.float32),
                self.camera_matrix,
                self.dist_coefficients,
                P=self.camera_matrix,
            )[0][0]
        else:
            undist_point = image_point
        image_center = np.array([self.image_size[0] / 2, self.image_size[1] / 2])
        image_point_from_center = undist_point - image_center
        image_plane_width_in_meters = np.tan(self.horizontal_fov / 2) * 2
        image_plane_height_in_meters = np.tan(self.vertical_fov / 2) * 2
        x = image_point_from_center[0] / self.image_size[0] * image_plane_width_in_meters
        y = 1
        z = -image_point_from_center[1] / self.image_size[1] * image_plane_height_in_meters
        vector = np.array([x, y, z])
        return vector

    def get_horizon_and_world_corners(
        self, world_point_dict: dict[Any, Any], yaw_pitch_roll: tuple[float, float, float]
    ) -> defaultdict[Any, list[dict[str, int]]]:
        margin = 200
        yaw_pitch_roll = (-yaw_pitch_roll[0], yaw_pitch_roll[1], yaw_pitch_roll[2])
        rotation_matrix = self.rotation(*yaw_pitch_roll)
        image_points = defaultdict(list)
        for direction, world_points in world_point_dict.items():
            temp_image_points_x = []
            temp_image_points_y = []
            for world_point in world_points:
                world_rotated_vector = np.matmul(np.transpose(rotation_matrix), world_point)
                if world_rotated_vector[1] >= 0:
                    vector = self.camera_matrix @ self.rotation(0, np.pi / 2, 0) @ world_rotated_vector
                    image_point_x = int(vector[0] / vector[2])
                    image_point_y = int(vector[1] / vector[2])
                    if (
                        -margin <= image_point_x <= self.image_size[0] + margin
                        and -margin <= image_point_y <= self.image_size[1] + margin
                    ):
                        temp_image_points_x.append(image_point_x)
                        temp_image_points_y.append(image_point_y)
            if temp_image_points_x != []:
                # Find last occurrence of the maximum x value
                name_id = len(temp_image_points_x) - 1 - temp_image_points_x[::-1].index(max(temp_image_points_x))
                # Reorder temp_image_points such that the largest x value is the first in the list.
                temp_image_points_x = shift(temp_image_points_x, name_id)
                temp_image_points_y = shift(temp_image_points_y, name_id)
            left = min(temp_image_points_x, default=0)
            top = min(temp_image_points_y, default=0)
            for x, y in zip(temp_image_points_x, temp_image_points_y, strict=False):
                image_points[direction].append({"x": x - left, "y": y - top})
            image_points[direction + "_pos"].append({"top": top, "left": left})
        return image_points

    def get_world_point(
        self,
        image_point: tuple[float, float],
        drone_height: float,
        yaw_pitch_roll: tuple[float, float, float],
        pos: tuple[float, float],
        return_zone: bool = False,
    ) -> tuple[np.ndarray, tuple[int, str]] | np.ndarray:
        unit_vector = self.get_unit_vector(image_point)
        yaw_pitch_roll = (-yaw_pitch_roll[0], yaw_pitch_roll[1], yaw_pitch_roll[2])
        rotation_matrix = self.rotation(*yaw_pitch_roll)
        rotated_vector = np.matmul(rotation_matrix, unit_vector)
        ground_vector = rotated_vector / rotated_vector[2] * -drone_height
        east_north_zone = self.convert_gps(*pos)
        world_point: np.ndarray = ground_vector[:2] + np.array(east_north_zone[:2])
        if return_zone:
            return world_point, east_north_zone[2:]
        else:
            return world_point

    def get_world_points(
        self,
        image_points: list[tuple[int, int]],
        drone_height: float,
        yaw_pitch_roll: tuple[float, float, float],
        pos: tuple[float, float],
    ) -> list[np.ndarray | tuple[np.ndarray, str]]:
        world_points = []
        for image_point in image_points:
            world_point = self.get_world_point(image_point, drone_height, yaw_pitch_roll, pos)
            world_points.append(world_point)
        return world_points

    def get_gps_point(
        self,
        image_point: tuple[int, int],
        drone_height: float,
        yaw_pitch_roll: tuple[float, float, float],
        pos: tuple[float, float],
    ) -> tuple[float, float]:
        world_point, zone = self.get_world_point(image_point, drone_height, yaw_pitch_roll, pos, True)
        lat, lon = self.convert_utm(world_point[0], world_point[1], zone)
        return lat, lon

    @staticmethod
    def convert_gps(lat: float, lon: float) -> tuple[float, float, int, str]:
        east_north_zone: tuple[float, float, int, str] = utm.from_latlon(lat, lon)
        return east_north_zone

    @staticmethod
    def convert_utm(east: float, north: float, zone: tuple[int, str]) -> tuple[float, float]:
        lat, lon = utm.to_latlon(east, north, *zone)
        return lat, lon



if __name__ == "__main__":
    # Example usage of the pixel_to_gps function with a specific flight and video acquisition
    FLIGHT_DAY = "Jan-16th-2026-02-49PM-Flight-Airdata"
    VIDEO_ACQUISITION = "DJI_20260116154954_0001_V"
    flight_info_dict = get_sub_info_dict(
        data_json_dict=DATA_JSON, key="flight_day", value=FLIGHT_DAY
    )

    flight_dir = Path("/share/projects/whale_drone_hack/WhaleDrone_Hackathon_dataset/Annotated") / FLIGHT_DAY
    annotation_path = ROOT_DIR_METADATA / FLIGHT_DAY / flight_info_dict["annotations_file"]

    df = load_flight(annotation_path, flight_dir)

    video_info_dict = get_sub_info_dict(
        data_json_dict=flight_info_dict["video_files"],
        key="video_file",
        value=VIDEO_ACQUISITION + ".MP4",
    )

    meta_data = video_info_dict["metadata_file"]
    video_file = video_info_dict["video_file"]

    segment = video_file

    df_testing_segment = df[df["video_segment"] == segment]

    df_testing_segment.head(10)

    # k, dist_coeffs = load_intrinsics(CALIB)

    # if meta_data is None:
    #     print(
    #         f"No metadata found for flight {FLIGHT_DAY} and video acquisition {VIDEO_ACQUISITION}."
    #     )
    # else:
    #     metadata_path = ROOT_DIR_METADATA / FLIGHT_DAY / meta_data
    #     acquisition_data_csv = pd.read_csv(metadata_path)

    # for index, row in acquisition_data_csv.iterrows():
    #     # frame_number = row["frame_cnt"]
    #     frame_number = 417
    #     x = 984.435
    #     y = 853.580

    #     k, dist_coeffs = load_intrinsics(CALIB)
    #     video_path = ROOT_DIR_DATASET / FLIGHT_DAY / video_file
    #     image = read_frame(video_path, frame_number)

    #     if image is not None:
    #         # show_frame_with_click(image, frame_number)
    #         gps, image_path = save_frame_with_fixed_point(
    #             image=image,
    #             frame_number=frame_number,
    #             x=x,
    #             y=y,
    #             row=row,
    #             k=k,
    #             dist_coeffs=dist_coeffs,
    #             grid_step=100,
    #             output_dir="gps_debug_frames",
    #         )
    #     else:
    #         print(f"Could not read frame {frame_number} from video.\n")

    #     break
