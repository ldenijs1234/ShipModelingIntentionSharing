"""
case04_config.py
Curved Fairway Crossing Encounter with Canal Banks (Trimmed Exit).
Coordinates are in NED [North (X), East (Y)].
"""

import math
import numpy as np
from shapely.geometry import Polygon
import shapely.ops as so

os_wps = np.array(
    [
        [ 25.0, -35.0],
        [ 15.0, -15.0],
        [  5.0,   5.0],   # Crossing area
        [  8.0,  20.0],   # Trimmed exit
    ],
    dtype=np.float64,
)

ts_wps = np.array(
    [
        [ 30.0,  30.0],
        [ 15.0,  15.0],
        [  5.0,  -2.0],   # Turning into south exit
        [-12.0,  -5.0],   # Trimmed exit inside channel opening
    ],
    dtype=np.float64,
)

psi_os_init = math.atan2(os_wps[1, 1] - os_wps[0, 1], os_wps[1, 0] - os_wps[0, 0])
psi_ts_init = math.atan2(ts_wps[1, 1] - ts_wps[0, 1], ts_wps[1, 0] - ts_wps[0, 0])

poly_north_bank = Polygon([
    ( 40.0, -45.0),
    ( 32.0, -25.0),
    ( 20.0,  -5.0),
    ( 24.0,  15.0),
    ( 30.0,  30.0),
    ( 45.0,  30.0),
    ( 45.0, -45.0)
])

poly_southwest_bank = Polygon([
    ( 12.0, -45.0),
    (  2.0, -28.0),
    ( -6.0, -10.0),
    (-25.0, -10.0),
    (-25.0, -45.0)
])

poly_southeast_bank = Polygon([
    (-25.0,   8.0),
    ( -8.0,   8.0),
    ( -2.0,  20.0),
    (  3.0,  30.0),
    ( 10.0,  30.0),
    (-25.0,  30.0)
])

poly_canal_full = so.unary_union([poly_north_bank, poly_southwest_bank, poly_southeast_bank])

CASE04_CONFIG = {
    "name": "case04",
    "description": "Multi-waypoint curved crossing encounter with canal banks (Trimmed Exit)",

    "canal_polygons": poly_canal_full,
    "canal_bounds": {
        "y_min": -45.0,
        "y_max":  30.0,
        "x_min": -25.0,
        "x_max":  45.0
    },

    "os_initial_state": np.array(
        [os_wps[0, 0], os_wps[0, 1], psi_os_init, 0.45, 0.0, 0.0], dtype=np.float64
    ),
    "os_nominal_speed": 0.45,
    "os_mission_wps": os_wps,

    "ts_initial_state": np.array(
        [ts_wps[0, 0], ts_wps[0, 1], psi_ts_init, 0.40, 0.0, 0.0], dtype=np.float64
    ),
    "ts_nominal_speed": 0.40,
    "ts_mission_wps": ts_wps,
    "t_sim": 95.0
}