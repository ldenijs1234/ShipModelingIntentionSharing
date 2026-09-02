from typing import Tuple
import numpy as np
from gnc_core.config.vessel_params import VesselParams


class Autopilot:

    @staticmethod
    def compute_control(
        x_os: np.ndarray, psi_wp: float, psi_ca_reactive: float, u_nominal: float = 0.5
    ) -> Tuple[float, float, float]:
        """PD steering controller with yaw rate damping and speed output (Eq. 3.47 - 3.48)."""
        psi, r = x_os[2], x_os[5]

        psi_cmd = (psi_wp + psi_ca_reactive + np.pi) % (2.0 * np.pi) - np.pi
        heading_err = (psi_cmd - psi + np.pi) % (2.0 * np.pi) - np.pi

        # Commanded steering effort tau_c
        tau_c = VesselParams.Kp_psi * heading_err - VesselParams.Kd_psi * r
        tau_c = float(np.clip(tau_c, -VesselParams.tau_max, VesselParams.tau_max))
        u_c = float(np.clip(u_nominal, VesselParams.u_min, VesselParams.u_max))

        return u_c, tau_c, float(psi_cmd)