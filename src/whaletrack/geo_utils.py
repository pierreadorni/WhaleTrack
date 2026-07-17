from geopy.point import Point  # Import Point class to represent geographic coordinates
from geopy.distance import geodesic  # Import geodesic function to calculate distances

def distance_points(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two geographic points in meters
    using geopy's Point objects and geodesic distance.

    Parameters:
    lat1, lon1: Latitude and longitude of the first point (in degrees)
    lat2, lon2: Latitude and longitude of the second point (in degrees)

    Returns:
    Distance in meters between the two points
    """

    # Create Point objects for both locations
    # Point(latitude, longitude) - stores coordinates in degrees
    point1 = Point(lat1, lon1)  # First geographic point
    point2 = Point(lat2, lon2)  # Second geographic point

    # Calculate geodesic distance between the two points
    # geodesic() returns a Distance object with various units
    # .meters extracts the distance in meters
    distance_meters = geodesic(point1, point2).meters

    return distance_meters