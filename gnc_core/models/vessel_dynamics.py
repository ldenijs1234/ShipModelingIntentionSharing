"""
vessel_dynamics.py
Vessel dynamics and RK4 numerical integrator following Sarhadi (2022).
"""

import numpy as np


class VesselDynamics:

    @staticmethod
    def vessel_dynamics(x_0: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """
        Calculate the vessel dynamics derivatives (Sarhadi 2022).
        
        Parameters:
            x_0: Current state [x, y, psi, r, b, u]
            inputs: Control inputs [tau_c, u_c]
            
        Returns:
            x_dot: State derivatives [x_dot, y_dot, psi_dot, r_dot, b_dot, u_dot]
        """
        x, y, psi, r, b, u = x_0
        tau_c, u_c = inputs

        # Scaled parameters for Tito Neri model
        k_psi = 1.0       # Scaled steering gain (was 0.01)
        t_psi = 0.5       # Fast yaw responsiveness (was 30.0)
        k_v = 1.0
        t_v = 1.0         # 1-second surge acceleration (was 50.0)
        t_b = 20.0 * t_psi

        x_dot = u_c * np.cos(psi)
        y_dot = u_c * np.sin(psi)
        psi_dot = r

        w_r = 0.0
        w_b = 0.0  # Set to 0.5 * np.random.randn() for stochastic drift

        # Nomoto steering and surge dynamics
        r_dot = -(1.0 / t_psi) * r + (1.0 / t_psi) * k_psi * (tau_c - b) + w_r
        b_dot = -(1.0 / t_b) * b + w_b
        u_dot = -(1.0 / t_v) * u + (1.0 / t_v) * k_v * u_c

        return np.array([x_dot, y_dot, psi_dot, r_dot, b_dot, u_dot], dtype=np.float64)

    @staticmethod
    def rk4(x: np.ndarray, inputs: np.ndarray, dt: float = 0.02) -> np.ndarray:
        """
        Advances the vessel state by dt using 4th-order Runge-Kutta.
        """
        k1 = VesselDynamics.vessel_dynamics(x, inputs)
        k2 = VesselDynamics.vessel_dynamics(x + 0.5 * dt * k1, inputs)
        k3 = VesselDynamics.vessel_dynamics(x + 0.5 * dt * k2, inputs)
        k4 = VesselDynamics.vessel_dynamics(x + dt * k3, inputs)

        x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Wrap heading angle psi to [-pi, pi]
        x_next[2] = (x_next[2] + np.pi) % (2.0 * np.pi) - np.pi

        return x_next