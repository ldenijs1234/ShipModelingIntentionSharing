import math
import numpy as np

# -------------------------------------------------------------------------
# Waypoint Definitions (Defined first so they can be referenced cleanly)
# -------------------------------------------------------------------------
os_wps = np.array(
    [
        [20.0, -30.0],
        [10.0, -10.0],
        [5.0, 5.0],  # Crossing area
        [8.0, 20.0],
        [15.0, 35.0],
    ],
    dtype=np.float64,
)

ts_wps = np.array(
    [
        [25.0, 25.0],
        [15.0, 15.0],
        [5.0, -2.0],
        [-5.0, -5.0],
        [-33.0, 0.0],
    ],
    dtype=np.float64,
)

# Initial course angles pointing toward the first leg: atan2(dy, dx)
psi_os_init = math.atan2(os_wps[1, 1] - os_wps[0, 1], os_wps[1, 0] - os_wps[0, 0])
psi_ts_init = math.atan2(ts_wps[1, 1] - ts_wps[0, 1], ts_wps[1, 0] - ts_wps[0, 0])

# -------------------------------------------------------------------------
# Scenario Configuration Dictionary
# -------------------------------------------------------------------------
CASE04_CONFIG = {
    "name": "case04",
    "description": "Multi-waypoint curved crossing based on sketch",

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

    # Boundary banks
    "left_bank": np.array([[45.0, -45.0], [45.0, 45.0]], dtype=np.float64),
    "right_bank": np.array([[-45.0, -45.0], [-45.0, 45.0]], dtype=np.float64),
}