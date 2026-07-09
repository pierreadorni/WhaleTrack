import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PIXEL TO GPS
    This notebook is to code the proper `pixel_to_gps` function
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from typing import List, Dict, Any, Optional

    import cv2
    import json
    from pathlib import Path
    from scipy.spatial.transform import Rotation as R

    ROOT_DIR = Path("/share/projects/whale_drone_hack")

    CALIB = ROOT_DIR / "WhaleDrone_Hackathon_dataset/camera_calibration_DJIM3T_RGBwide.json"

    return CALIB, R, cv2, np


@app.cell
def _():
    import marimo as mo

    from whaledrone_hackathon_code.reprojection_to_world import load_intrinsics

    return load_intrinsics, mo


@app.cell
def _(CALIB, load_intrinsics):
    FRAME = (3840, 2160)
    k, dist_coeffs = load_intrinsics(CALIB)

    middle_point = (FRAME[0]//2, FRAME[1]//2)


    print("k :" , k.astype(int))
    print("Middle point : ", middle_point)
    return dist_coeffs, k, middle_point


@app.cell
def _(k, middle_point):
    # Parameters
    # ----------
    # Coordonnées pixel de l'objet.
    # u, v : float
    (u, v) =  middle_point

    # K : np.ndarray, shape (3, 3)
    #    Matrice intrinsèque caméra.
    K = k        
    # drone_lat, drone_lon : float
    #    Position GPS du drone.
        
    # altitude_agl : float
    #    Hauteur de la caméra au-dessus du sol, en mètres.

    altitude_agl = 10 

    # yaw, pitch, roll : float
    #    Orientation de la caméra.
    yaw = 0
    roll = 0
    pitch = -90

    # dist_coeffs : optionnel
    #    Coefficients de distorsion, non utilisés ici.
    # degrees : bool
    #    Angles en degrés si True.

    debug = True
        
    return K, altitude_agl, debug, pitch, roll, u, v, yaw


@app.cell
def _(
    K,
    R,
    altitude_agl,
    cv2,
    debug,
    degrees,
    dist_coeffs,
    drone_lat,
    drone_lon,
    np,
    pitch,
    roll,
    u,
    v,
    yaw,
):

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

    east_offset = P[0] # x
    north_offset = P[1] # y

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

    return


if __name__ == "__main__":
    app.run()
