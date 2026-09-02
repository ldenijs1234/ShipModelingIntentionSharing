import numpy as np


class VesselParams:
    # Hull specs (Tito Neri 1:30, Table 2.3)
    L = 0.98
    B = 0.30
    m = 16.90
    scale = 30.0

    # Safety thresholds (Section 2.5)
    DCPA_safe = 1.0  # [m] (Eq. 2.5)
    TCPA_safe = 20.0  # [s] (Eq. 2.4)

    # Domain semi-axes (Eq. 2.6)
    R_long = 0.98
    R_lateral = 0.30

    # Actuator limits & gains
    tau_max = 1.5
    u_max = 1.2
    u_min = -0.5
    mu_los = 1.5  # Lookahead factor
    D_m = 0.40  # Waypoint acceptance radius
    Kp_psi = 2.5  # Autopilot proportional gain
    Kd_psi = 0.8  # Autopilot derivative gain