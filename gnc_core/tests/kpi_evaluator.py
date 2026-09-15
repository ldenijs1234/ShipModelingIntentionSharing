import numpy as np


class ScenarioKPIEvaluator:

  def __init__(
      self,
      d_safe: float,
      nominal_waypoints: np.ndarray,
      w_cte: float = 0.7,
      w_ctrl: float = 0.3,
  ):
    """nominal_waypoints: Array of shape (N, 2) containing [X, Y] coordinates of the

    original pre-encounter mission track.
    """
    self.d_safe = d_safe
    self.nominal_wps = nominal_waypoints
    self.w_cte = w_cte
    self.w_ctrl = w_ctrl

  def calculate_cte(
      self, x: np.ndarray, y: np.ndarray, wp_a: np.ndarray, wp_b: np.ndarray
  ) -> np.ndarray:
    """Computes signed cross-track error using Chapter 3 projection logic."""
    dx = wp_b[0] - wp_a[0]
    dy = wp_b[1] - wp_a[1]
    alpha_k = np.arctan2(dy, dx)
    # Orthogonal projection relative to leg bearing
    return -(x - wp_a[0]) * np.sin(alpha_k) + (y - wp_a[1]) * np.cos(alpha_k)

  def evaluate_single_run(
        self,
        t: np.ndarray,
        os_pos: np.ndarray,
        os_psi: np.ndarray,
        os_r: np.ndarray,
        ts_pos: np.ndarray,
    ) -> dict:
        """Evaluates raw metrics for a single simulation run."""
        # 0. Sanitize duplicate timestamps to prevent gradient divide-by-zero
        t_clean, unique_indices = np.unique(t, return_index=True)
        os_pos = os_pos[unique_indices]
        os_psi = os_psi[unique_indices]
        os_r = os_r[unique_indices]
        ts_pos = ts_pos[unique_indices]
        t = t_clean

        # Select compatible trapezoidal integrator for NumPy 1.x and 2.x
        trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))

        duration = max(t[-1] - t[0], 1.0)
         
        # 1. Safety Constraint: Range R(t)
        ranges = np.linalg.norm(ts_pos - os_pos, axis=1)
        r_min = float(np.min(ranges))
        breached = r_min < self.d_safe
        j_safe = np.inf if breached else 0.0

        # 2. Control Effort: Integral of squared yaw acceleration r_dot^2
        if len(t) > 1:
            r_dot = np.gradient(os_r, t)
            j_ctrl = float(trapz_fn(r_dot**2, t)) / duration
        else:
            j_ctrl = 0.0

        # 3. Mission Tracking: Cross-Track Error relative to nominal track
        e_cte = self.calculate_cte(
            os_pos[:, 0], os_pos[:, 1], self.nominal_wps[0], self.nominal_wps[1]
        )
        j_cte = float(trapz_fn(np.abs(e_cte), t)) / duration if len(t) > 1 else 0.0

        return {
            "r_min": r_min,
            "breached": breached,
            "j_safe": j_safe,
            "j_ctrl": j_ctrl,
            "j_cte": j_cte,
        }

  def evaluate_comparison(self, raw_ra: dict, raw_is: dict) -> dict:
    """Computes relative baseline normalization and final delta J."""
    # Safety gate
    if raw_is["breached"]:
      return {
          "status": "IS_BREACH",
          "j_total_is": np.inf,
          "delta_j": np.inf,
          "delta_j_pct": -np.inf,
      }

    if raw_ra["breached"] and not raw_is["breached"]:
      return {
          "status": "SAFETY_ENHANCEMENT",
          "j_total_is": 0.0,
          "delta_j": -np.inf,
          "delta_j_pct": 100.0,
      }

    # Operational evaluation (Both safe)
    norm_ctrl = raw_is["j_ctrl"] / raw_ra["j_ctrl"]
    norm_cte = raw_is["j_cte"] / raw_ra["j_cte"]

    j_total_is = self.w_ctrl * norm_ctrl + self.w_cte * norm_cte
    delta_j = j_total_is - 1.0
    delta_j_pct = (1.0 - j_total_is) * 100.0

    return {
        "status": "SAFE_COMPARISON",
        "norm_ctrl": norm_ctrl,
        "norm_cte": norm_cte,
        "j_total_is": j_total_is,
        "delta_j": delta_j,
        "delta_j_pct": delta_j_pct,
    }