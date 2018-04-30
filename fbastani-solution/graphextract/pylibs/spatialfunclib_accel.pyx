import math

cdef double METERS_PER_DEGREE_LATITUDE, METERS_PER_DEGREE_LONGITUDE

METERS_PER_DEGREE_LATITUDE = 111070.34306591158
METERS_PER_DEGREE_LONGITUDE = 83044.98918812413

#
# Returns the distance in meters between two points specified in degrees, using an approximation method.
#
def fast_distance(double a_lat, double a_lon, double b_lat, double b_lon):
    if a_lat == b_lat and a_lon==b_lon:
        return 0.0
    
    cdef double y_dist, x_dist
    y_dist = METERS_PER_DEGREE_LATITUDE * (a_lat - b_lat)
    x_dist = METERS_PER_DEGREE_LONGITUDE * (a_lon - b_lon)
    
    return math.sqrt((y_dist * y_dist) + (x_dist * x_dist))

