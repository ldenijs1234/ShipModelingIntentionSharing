"""
case02_config.py
Confined Fairway Encounter with Turning TS (Trimmed Exit).
Coordinates are in NED [North (X), East (Y)].
"""

import numpy as np
from shapely.geometry import Polygon
import shapely.ops as so

poly_port = Polygon([
    (-35.0, -18.0),
    (35.0, -18.0),
    (35.0, -6.0),
    (-35.0, -6.0)
])

poly_starboard_north = Polygon([
    (5.0, 15.0),
    (35.0, 15.0),
    (35.0, 6.0),
    (5.0, 6.0)
])

poly_starboard_south = Polygon([
    (-35.0, 15.0),
    (-25.0, 15.0),
    (-25.0, 6.0),
    (-35.0, 6.0)
])

poly_canal_full = so.unary_union([poly_port, poly_starboard_north, poly_starboard_south])

CASE02_CONFIG = {
    "name": "case02",
    "description": "Akdag Confined Fairway Encounter with Turning TS (Trimmed Exit)",

    "canal_polygons": poly_canal_full,
    "canal_bounds": {
        "y_min": -4.0,
        "y_max": 4.0,
        "x_min": -35.0,
        "x_max": 35.0
    },

    # Own Ship (Blue) 
    # Starts at North = -20.0, heads straight North
    "os_initial_state": np.array([-20.0, 0.0, 0.0, 0.45, 0.0, 0.0], dtype=np.float64),
    "os_nominal_speed": 0.45,
    "os_mission_wps": np.array([
        [-20.0,  0.0],
        [ 20.0,  0.0]
    ], dtype=np.float64),

    # Target Ship (Magenta/Red)
    # Starts at North = 20.0. 
    # Easting is -0.5m (Deep inside OS's 1.0m safety domain to trigger immediate evasion)
    "ts_initial_state": np.array([20.0, -1.5, np.pi, 0.45, 0.0, 0.0], dtype=np.float64),
    "ts_nominal_speed": 0.45,
    "ts_mission_wps": np.array([
        [ 20.0, -1.5],   # Start (Matches Akdag [-100, 2000])
        [ 3.0, -1.5],   # Turn Point (Matches Akdag [-100, 500])
        [-20.0, 5.5]    # End Point (Matches Akdag [1000, -2000] retaining the 23.7 deg angle)
    ], dtype=np.float64),
    
    "t_sim": 120.0
}