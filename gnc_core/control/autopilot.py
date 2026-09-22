from typing import Tuple
import numpy as np
from gnc_core.config.vessel_params import VesselParams


class Autopilot:
    _last_tau: float = 0.0

    @classmethod
    def reset(cls):
        """Reset internal actuator state at the beginning of each run."""
        cls._last_tau = 0.0

    @classmethod
    def compute_control(
        cls,
        x_os: np.ndarray,
        psi_wp: float,
        psi_ca_reactive: float,
        u_nominal: float = 0.5,
        dt: float = 0.05,
    ) -> Tuple[float, float, float]:
        """PD steering controller with actuator rate limiting (Eq. 3.47 - 3.48)."""
        psi, r = x_os[2], x_os[5]

        psi_cmd = (psi_wp + psi_ca_reactive + np.pi) % (2.0 * np.pi) - np.pi
        heading_err = (psi_cmd - psi + np.pi) % (2.0 * np.pi) - np.pi

        # Unconstrained commanded steering effort
        tau_desired = VesselParams.Kp_psi * heading_err - VesselParams.Kd_psi * r
        tau_desired = float(np.clip(tau_desired, -VesselParams.tau_max, VesselParams.tau_max))

        # --- Actuator Slew Rate Limiter ---
        delta_tau = tau_desired - cls._last_tau
        max_delta = VesselParams.tau_dot_max * dt
        tau_c = cls._last_tau + float(np.clip(delta_tau, -max_delta, max_delta))
        cls._last_tau = tau_c

        u_c = float(np.clip(u_nominal, VesselParams.u_min, VesselParams.u_max))

        return u_c, tau_c, float(psi_cmd)