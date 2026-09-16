import numpy as np

CASE02_CONFIG = {
    # -------------------------------------------------------------------------
    # Scenario Info
    # -------------------------------------------------------------------------
    "name": "case02",
    "description": "Akdag Confined Fairway Encounter with Turning TS (Scale 1:100)",

    # -------------------------------------------------------------------------
    # Own Ship (Blue) Configuration
    # -------------------------------------------------------------------------
    # Initial state: [x, y, psi, u, v, r]
    # Starts at x = -20 m, y = 0.0 m, heading North (psi = 0 rad)
    "os_initial_state": np.array([-20.0, 0.0, 0.0, 0.0, 0.0, 0.45], dtype=np.float64),
    "os_nominal_speed": 0.45,  # [m/s]
    "os_mission_wps": np.array([
        [-20.0, 0.0],
        [-10.0, 0.0],
        [ -2.0, 0.0],  # Pre-conflict anchor
        [  5.0, 0.0],  # Immediate recovery anchor (just past TS turn)
        [ 12.0, 0.0],
        [ 20.0, 0.0]
    ], dtype=np.float64),

    # -------------------------------------------------------------------------
    # Target Ship (Magenta / Red) Configuration
    # -------------------------------------------------------------------------
    # Initial state: [x, y, psi, u, v, r]
    # Starts at x = +20 m, y = -1.0 m, heading South (psi = pi rad)
    "ts_initial_state": np.array([20.0, -1.0, np.pi, 0.0, 0.0, 0.45], dtype=np.float64),
    "ts_nominal_speed": 0.45,  # [m/s]
    # TS route: South along west lane, then turns southeast at x = 5.0 m
    "ts_mission_wps": np.array([
        [ 20.0, -1.0],
        [  6.0, -1.0],
        [-20.0, 10.0]
    ], dtype=np.float64),

    # -------------------------------------------------------------------------
    # Fairway Canal Boundaries (Akdag Shorelines)
    # [North (x), East (y)]
    # -------------------------------------------------------------------------
    "left_bank": np.array([
        [-25.0, -15.0],
        [-15.0, -21.0],
        [  0.0, -23.0],
        [ 15.0, -21.0],
        [ 25.0, -15.0]
    ], dtype=np.float64),

    "right_bank": np.array([
        [-25.0, 25.0],
        [ -5.0, 20.0],
        [  5.0, 26.0],
        [ 25.0, 26.0]
    ], dtype=np.float64)
}