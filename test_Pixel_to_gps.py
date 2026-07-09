import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    #sys.path.append("../Whale_Group2/")  # Ajoute le chemin du dossier
    #
    import pandas as pd

    from whaledrone_hackathon_code.reprojection_to_world import pixel_to_gps, load_intrinsics, ROOT_DIR, ROOT_DIR_DATASET, ROOT_DIR_METADATA, ALL_FLIGHTS, CALIB, DATA_JSON

    from whaledrone_hackathon_code.dataset import load_flight

    k, dist_coeffs = load_intrinsics(CALIB)

    #print(DATA_JSON)
    # Annotation database
    #annotations = {
    #    "video_file" : 
    #    "metadata_file "
    #    "annotations_file" : 
    #}

    #ALL_FLIGHTS
    DATA_JSON
    return ROOT_DIR_DATASET, dist_coeffs, k, load_flight, pd, pixel_to_gps


@app.cell
def _(ROOT_DIR_DATASET, load_flight):

    FLIGHT = "Jan-17th-2026-03-12PM-Flight-Airdata"
    flight_dir = ROOT_DIR_DATASET / FLIGHT
    annotations_file = f"annotations_{FLIGHT}.csv"
    annotation_path = ROOT_DIR_DATASET / annotations_file

    df = load_flight(annotation_path, flight_dir)
    # Add the correspond DJI*.MP4 video file 
    df["segment"].map(lambda s: s.name).value_counts()  # rows per DJI segment
    df
    return FLIGHT, df


@app.cell
def _():
    from whaledrone_hackathon_code.dataset import find_closest_time_row
    from whaledrone_hackathon_code.reprojection_to_world import get_telemetry_file
    """
    time = df.iloc[0].time

    video_segment =  df.iloc[0].segment.name

    metadata_path = get_telemetry_file(flight_day=FLIGHT, video_segment=video_segment)

    df_telemetry = pd.read_csv(metadata_path)
    time_2 = df_telemetry.iloc[0].datetime
    time_row = find_closest_time_row(df=df_telemetry, time_col="datetime", target_datetime=time, offset_s=3600)
    print(time, time_2, time_row.datetime)"""
    return find_closest_time_row, get_telemetry_file


@app.cell
def _(
    FLIGHT,
    df,
    dist_coeffs,
    find_closest_time_row,
    get_telemetry_file,
    k,
    pd,
    pixel_to_gps,
):
    video_segment = None
    metadata_path = None
    skip_segment = False

    from math import sqrt
    from whaledrone_hackathon_code.geo_utils import distance_points
    from whaledrone_hackathon_code.reprojection_to_world import zoom_intrinsics

    distance_error = []
    for row in df.itertuples():

        annotated_lat = row.lat
        annotated_lon = row.lon
        annotated_time = row.time

        # Update telemetry meatadata if video segment change 
        if video_segment != row.segment:
            video_segment = row.segment.name
            metadata_path = get_telemetry_file(flight_day=FLIGHT, video_segment=video_segment)
            df_telemetry = pd.read_csv(metadata_path)

        sync_telemetry_row = find_closest_time_row(df=df_telemetry, time_col="datetime", target_datetime=annotated_time, offset_s=3600)

        focal_length = sync_telemetry_row.focal_len
        zoom_ratio = sync_telemetry_row.dzoom_ratio

        k_zoom, IS_WIDE_CAMERA = zoom_intrinsics(k, focal_len=focal_length, zoom_ratio=zoom_ratio)
    	# TODO handle IS_WIDE_CAMERA
        drone_lon = sync_telemetry_row.longitude
        drone_lat = sync_telemetry_row.latitude
        try:
            lat_obj, lon_obj  = pixel_to_gps(
                u=row.image_x,  # Example pixel coordinates
                v=row.image_y,
                K=k_zoom,
                drone_lat=drone_lat,
                drone_lon=drone_lon,
                altitude_agl=sync_telemetry_row.rel_alt,
                yaw=sync_telemetry_row.gb_yaw,
                pitch=sync_telemetry_row.gb_pitch,
                roll=sync_telemetry_row.gb_roll,
                dist_coeffs=dist_coeffs,
                degrees=True
            )
        except ValueError:
            continue
        coord_error = distance_points(annotated_lat, annotated_lon, lat_obj, lon_obj)
        distance_drone_annotated_centroid = distance_points(annotated_lat, annotated_lon, drone_lat, drone_lon)
        distance_drone_estimated_centroid = distance_points(drone_lat, drone_lon, lat_obj, lon_obj)
        distance_error.append((annotated_time, coord_error, distance_drone_annotated_centroid, distance_drone_estimated_centroid))
    return (
        distance_error,
        distance_points,
        drone_lat,
        drone_lon,
        sync_telemetry_row,
    )


@app.cell
def _(distance_error, pd):
    df_error = pd.DataFrame(distance_error, columns=['datetime', 'error', 'distance_to_drone_annotated', 'distance_to_drone']) 
    return (df_error,)


@app.cell
def _(df_error):
    import marimo as mo
    mo.mpl.interactive(df_error.plot(x="datetime"))
    #mo.mpl.interactive(df_error.plot(x="datetime", y='distance_to_drone_annotated'))
    return (mo,)


@app.cell
def _(df_error, mo):
    mo.mpl.interactive(df_error.plot(x="datetime", y='distance_to_drone'))
    return


@app.cell
def _():
    return


@app.cell
def _(sync_telemetry_row):
    sync_telemetry_row
    return


@app.cell
def _(distance_points, drone_lat, drone_lon, k, pixel_to_gps):
    FRAME = (3840, 2160)
    import numpy as np

    # Paramètres de la caméra (centre de l'image)
    K = np.array([[3000, 0, 1920], [0, 3000, 1080], [0, 0, 1]])

    lat_obj2, lon_obj2 = pixel_to_gps(
        u=FRAME[0] / 2,  # Example pixel coordinates
        v=FRAME[1] / 2,
        K=k,
        drone_lat=drone_lat,
        drone_lon=drone_lon,
        altitude_agl=100,  # sync_telemetry_row.rel_alt,
        yaw=0,  # sync_telemetry_row.gb_yaw,
        pitch=-90,  # sync_telemetry_row.gb_pitch,
        roll=0,  # sync_telemetry_row.gb_roll,
        dist_coeffs=None,  # dist_coeffs,
        degrees=True,
        debug=True,
    )
    print(lat_obj2, lon_obj2)

    print(
        f"Différence avec drone: lat_diff={abs(lat_obj2 - drone_lat):.8f}, lon_diff={abs(lon_obj2 - drone_lon):.8f}"
    )

    coord_distance = distance_points(drone_lat, drone_lon, lat_obj2, lon_obj2)
    print(f"Distance : {coord_distance} m")
    return


@app.cell
def _(distance_points, drone_lat, drone_lon):


    print(distance_points(drone_lat, drone_lon, drone_lat+0.00000001, drone_lon+0.00000060))
    return


if __name__ == "__main__":
    app.run()
