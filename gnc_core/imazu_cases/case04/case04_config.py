"""
case04_config.py
Curved Fairway Crossing Encounter with Canal Banks.
Coordinates are in NED [North (X), East (Y)].
"""

import math
import numpy as np
from shapely.geometry import Polygon
import shapely.ops as so

# -------------------------------------------------------------------------
# Waypoints
# OS sails generally South-East along the main curved channel
# TS crosses from the North-East and enters through the southern fairway opening
# -------------------------------------------------------------------------
os_wps = np.array(
    [
        [ 25.0, -35.0],
        [ 15.0, -15.0],
        [  5.0,   5.0],   # Crossing area
        [  8.0,  20.0],
        [ 15.0,  35.0],
    ],
    dtype=np.float64,
)

ts_wps = np.array(
    [
        [ 30.0,  30.0],
        [ 15.0,  15.0],
        [  5.0,  -2.0],   # Turning point into the south opening
        [ -5.0,  -5.0],
        [-33.0,   0.0],   # Terminal position in south harbor/branch
    ],
    dtype=np.float64,
)

psi_os_init = math.atan2(os_wps[1, 1] - os_wps[0, 1], os_wps[1, 0] - os_wps[0, 0])
psi_ts_init = math.atan2(ts_wps[1, 1] - ts_wps[0, 1], ts_wps[1, 0] - ts_wps[0, 0])

# -------------------------------------------------------------------------
# Curved Canal Banks with Southern Opening
# -------------------------------------------------------------------------

# 1. North Bank (Curved shoreline enclosing the top edge of the fairway)
poly_north_bank = Polygon([
    ( 40.0, -45.0),
    ( 32.0, -25.0),
    ( 20.0,  -5.0),
    ( 24.0,  15.0),
    ( 30.0,  32.0),
    ( 38.0,  45.0),
    ( 50.0,  45.0),
    ( 50.0, -45.0)
])

# 2. South-West Bank (Curving boundary left of the OS approach)
# Stops at Y = -10.0 to create the south-western side of the opening
poly_southwest_bank = Polygon([
    ( 12.0, -45.0),
    (  2.0, -28.0),
    ( -6.0, -10.0),
    (-45.0, -10.0),
    (-45.0, -45.0)
])

# 3. South-East Bank (Curving boundary right of the TS exit path)
# Starts at Y = +8.0, leaving an 18-meter navigable gap (Y in [-10, +8]) for TS to exit South
poly_southeast_bank = Polygon([
    (-45.0,   8.0),
    ( -8.0,   8.0),
    ( -2.0,  20.0),
    (  3.0,  35.0),
    ( 10.0,  45.0),
    (-45.0,  45.0)
])

poly_canal_full = so.unary_union([poly_north_bank, poly_southwest_bank, poly_southeast_bank])

CASE04_CONFIG = {
    "name": "case04",
    "description": "Multi-waypoint curved crossing encounter with curved canal and southern opening",

    # Canal polygon definition (Compatible with both Shapely and MplPolygon)
    "canal_polygons": poly_canal_full,
    "canal_bounds": {
        "y_min": -45.0,
        "y_max":  45.0,
        "x_min": -45.0,
        "x_max":  50.0
    },

    # Own Ship (Blue)
    "os_initial_state": np.array(
        [os_wps[0, 0], os_wps[0, 1], psi_os_init, 0.45, 0.0, 0.0], dtype=np.float64
    ),
    "os_nominal_speed": 0.45,
    "os_mission_wps": os_wps,

    # Target Ship (Red)
    "ts_initial_state": np.array(
        [ts_wps[0, 0], ts_wps[0, 1], psi_ts_init, 0.40, 0.0, 0.0], dtype=np.float64
    ),
    "ts_nominal_speed": 0.40,
    "ts_mission_wps": ts_wps,
    "t_sim": 180.0
}