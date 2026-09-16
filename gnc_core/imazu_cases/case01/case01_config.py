"""
case01_config.py
Configuration parameters for Imazu Case 01 (Head-On Encounter) with Canal Boundaries.
Coordinates are in NED [North (X), East (Y)].
"""

import numpy as np
from shapely.geometry import Polygon
import shapely.ops as so

# Define canal banks (fairway width = 6.0m, Y between -3.0 and +3.0)
# Port (West) Bank: Y in [-15.0, -3.0]
poly_port = Polygon([
    (-10.0, -15.0),
    (50.0, -15.0),
    (50.0, -4.0),
    (-10.0, -4.0)
])

# Starboard (East) Bank: Y in [3.0, 15.0]
poly_starboard = Polygon([
    (-10.0, 4.0),
    (50.0, 4.0),
    (50.0, 15.0),
    (-10.0, 15.0)
])

poly_canal_full = so.unary_union([poly_port, poly_starboard])

CASE01_CONFIG = {
    "name": "case01",
    "description": "Imazu Case 01: Symmetrical Head-On Encounter with Canal Walls",
    
    # Canal polygon definition
    "canal_polygons": poly_canal_full,
    "canal_bounds": {
        "y_min": -4.0,
        "y_max": 4.0,
        "x_min": -5.0,
        "x_max": 45.0
    },

    # Own Ship (OS) Initial Conditions & Route
    "os_initial_state": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float64),
    "os_nominal_speed": 0.5,
    "os_mission_wps": np.array([
        [0.0, 0.0],
        [12.5, 0.0],
        [25.0, 0.0]
    ], dtype=np.float64),

    # Target Ship (TS) Initial Conditions & Route
    "ts_initial_state": np.array([25.0, 0.0, np.pi, 0.0, 0.0, 0.5], dtype=np.float64),
    "ts_nominal_speed": 0.5,
    "ts_mission_wps": np.array([
        [25.0, 0.0],
        [12.5, 0.0],
        [0.0, 0.0]
    ], dtype=np.float64),
}