"""
case06_config.py
Imazu / Inland Canal Overtaking Encounter (COLREGs Rule 13).
Coordinates are in NED [North (X), East (Y)].
"""

import numpy as np
from shapely.geometry import Polygon
import shapely.ops as so

poly_port = Polygon([
    (-45.0, -15.0),
    (75.0, -15.0),
    (75.0, -3.0),
    (-45.0, -3.0)
])

poly_starboard = Polygon([
    (-45.0, 3.0),
    (75.0, 3.0),
    (75.0, 15.0),
    (-45.0, 15.0)
])

poly_canal_full = so.unary_union([poly_port, poly_starboard])

CASE06_CONFIG = {
    "name": "case06",
    "description": "Inland Canal Overtaking Encounter (Rule 13) with Low Closing Speed",

    "canal_polygons": poly_canal_full,
    "canal_bounds": {
        "y_min": -3.0,
        "y_max": 3.0,
        "x_min": -40.0,
        "x_max": 70.0
    },

    # Own Ship (Faster vessel: 0.52 m/s, starts behind at X = -35.0 m)
    "os_initial_state": np.array([-35.0, -0.8, 0.0, 0.52, 0.0, 0.0], dtype=np.float64),
    "os_nominal_speed": 0.52,
    "os_mission_wps": np.array([
        [-35.0, -0.8],
        [ 65.0, -0.8]
    ], dtype=np.float64),

    # Target Ship (Slower vessel: 0.28 m/s, starts ahead at X = -10.0 m)
    # Initial separation = 25.0 m. Relative closing speed = 0.24 m/s.
    # R_IS is clamped to R_min (10.0 m) -> Starts safely out of range (25.0 m > 10.0 m)
    "ts_initial_state": np.array([-10.0, -0.8, 0.0, 0.28, 0.0, 0.0], dtype=np.float64),
    "ts_nominal_speed": 0.28,
    "ts_mission_wps": np.array([
        [-10.0, -0.8],
        [ 65.0, -0.8]
    ], dtype=np.float64),
    "t_sim": 220.0
}