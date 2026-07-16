"""Camera-pixel to WGS84 reprojection helpers used by the streaming pipeline."""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


_DVM_FROM_CV_AXES = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def world_from_cv_rotation(yaw, pitch, roll, degrees=True):
    """Return DVM's camera-to-ENU transform for DJI gimbal orientation."""
    if degrees:
        yaw, pitch, roll = np.deg2rad([yaw, pitch, roll])
    return (
        Rotation.from_euler("z", -yaw).as_matrix()
        @ Rotation.from_euler("x", pitch).as_matrix()
        @ Rotation.from_euler("y", roll).as_matrix()
        @ _DVM_FROM_CV_AXES
    )


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
):
    """Reproject one OpenCV image pixel to WGS84 using DVM's DJI convention."""
    camera_position = np.array([0.0, 0.0, altitude_agl])
    if dist_coeffs is not None:
        undistorted = cv2.undistortPoints(
            np.array([[[u, v]]], dtype=np.float32), K, dist_coeffs, P=K
        )
        u, v = undistorted[0, 0]
    camera_ray = np.linalg.inv(K) @ np.array([u, v, 1.0])
    camera_ray /= np.linalg.norm(camera_ray)
    world_ray = world_from_cv_rotation(yaw, pitch, roll, degrees) @ camera_ray
    world_ray /= np.linalg.norm(world_ray)
    if abs(world_ray[2]) < 1e-8:
        raise ValueError("The camera ray is parallel to the sea surface.")
    distance = -camera_position[2] / world_ray[2]
    if distance < 0:
        raise ValueError("The camera ray does not intersect the sea surface ahead of the camera.")
    east_offset, north_offset, _ = camera_position + distance * world_ray
    earth_radius = 6378137.0
    flattening = 1 / 298.257223563
    latitude = np.radians(drone_lat)
    longitude = np.radians(drone_lon)
    meridional_radius = earth_radius * (1 - flattening) / (1 - flattening * np.sin(latitude)) ** 2
    object_latitude = latitude + north_offset / meridional_radius
    transverse_radius = earth_radius / np.sqrt(1 - flattening * np.sin(object_latitude) ** 2)
    object_longitude = longitude + east_offset / (transverse_radius * np.cos(object_latitude))
    return float(np.degrees(object_latitude)), float(np.degrees(object_longitude))