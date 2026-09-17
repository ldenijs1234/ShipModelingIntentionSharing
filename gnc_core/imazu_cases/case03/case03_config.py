"""
case03_config.py
Configuration parameters for Imazu Case 03 (4-Way Canal Crossroads / Intersection).
Coordinates are in NED [North (X), East (Y)].
"""

import numpy as np
from shapely.geometry import Polygon
import shapely.ops as so

# -----------------------------------------------------------------------------
# 4-Way Crossroads Canal Definition (Fairway width = 8.0m, Branch width = 8.0m)
# Main Fairway (North-South): Y between -4.0 and +4.0
# Cross Fairway (East-West):   X between -4.0 and +4.0 (extending to Y = -20.0 and Y = +26.0)
# -----------------------------------------------------------------------------

# 1. South-West Corner (South of West branch, West of main canal)
poly_sw_corner = Polygon([
    (-25.0, -20.0),
    (-4.0, -20.0),
    (-4.0, -4.0),
    (-25.0, -4.0)
])

# 2. North-West Corner (North of West branch, West of main canal)
poly_nw_corner = Polygon([
    (4.0, -20.0),
    (30.0, -20.0),
    (30.0, -4.0),
    (4.0, -4.0)
])

# 3. South-East Corner (South of East branch, East of main canal)
poly_se_corner = Polygon([
    (-25.0, 4.0),
    (-4.0, 4.0),
    (-4.0, 26.0),
    (-25.0, 26.0)
])

# 4. North-East Corner (North of East branch, East of main canal)
poly_ne_corner = Polygon([
    (4.0, 4.0),
    (30.0, 4.0),
    (30.0, 26.0),
    (4.0, 26.0)
])

# Merge the four corner quadrants into a single obstacle polygon
poly_canal_full = so.unary_union([
    poly_sw_corner,
    poly_nw_corner,
    poly_se_corner,
    poly_ne_corner
])

CASE03_CONFIG = {
    "name": "case03",
    "description": "Imazu Case 03: 4-Way Crossroads Intersection with Turning Target Ship",

    # Canal polygon definition (read directly by live_plotter and decision layer)
    "canal_polygons": poly_canal_full,
    "canal_bounds": {
        "y_min": -15.0,
        "y_max": 25.0,
        "x_min": -22.0,
        "x_max": 28.0
    },

    # Own Ship (OS) Initial Conditions & Route (Northbound through main canal)
    "os_initial_state": np.array([-20.0, 0.0, 0.0, 0.42, 0.0, 0.0], dtype=np.float64),
    "os_nominal_speed": 0.42,
    "os_mission_wps": np.array([
        [-20.0, 0.0],
        [25.0, 0.0]
    ], dtype=np.float64),

    # Target Ship (TS) Initial Conditions & Route (Heads toward west opening, steers right into main)
    "ts_initial_state": np.array([0.0, 22.0, -np.pi / 2.0, 0.45, 0.0, 0.0], dtype=np.float64),
    "ts_nominal_speed": 0.45,
    "ts_mission_wps": np.array([
        [0.0, 22.0],
        [0.0, 2.5],
        [25.0, 2.5]
    ], dtype=np.float64),
}