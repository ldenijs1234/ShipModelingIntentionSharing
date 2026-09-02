import numpy as np


class StateEstimation:

    @staticmethod
    def estimate_own_state(raw_state: np.ndarray) -> np.ndarray:
        """
        Maps internal state [X, Y, psi, r, b, u] to standardized 6-state [X, Y, psi, u, v, r] with v=0.
        """
        X, Y, psi, r, _, u = raw_state
        return np.array([X, Y, psi, u, 0.0, r], dtype=np.float64)

    @staticmethod
    def estimate_target_from_ais(
        ais_current: np.ndarray,
        ais_prev: np.ndarray | None,
        dt_ais: float = 1.0,
    ) -> np.ndarray:
        """
        Reconstructs full 6-state vector X_TS from raw AIS inputs [X, Y, psi, U].
        
        Parameters:
            ais_current: [X, Y, psi, U] at current step
            ais_prev:    [X, Y, psi, U] at previous AIS update (or None)
            dt_ais:      Time elapsed between AIS packets [s]
            
        Returns:
            X_TS: [X, Y, psi, u, v, r] with v=0 and estimated yaw rate r
        """
        X, Y, psi, U = ais_current

        if ais_prev is not None and dt_ais > 1e-4:
            psi_prev = ais_prev[2]
            # Wrap delta heading to [-pi, pi] before dividing by dt
            d_psi = (psi - psi_prev + np.pi) % (2.0 * np.pi) - np.pi
            r_est = float(d_psi / dt_ais)
        else:
            r_est = 0.0

        u_ts = float(U)
        v_ts = 0.0  # Non-holonomic assumption

        return np.array([X, Y, psi, u_ts, v_ts, r_est], dtype=np.float64)