import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path

def plot_control_effort(scenario="case02"):
    # Automatically resolve the root directory (2 levels up from gnc_core/tests/)
    repo_root = Path(__file__).resolve().parents[2]
    log_dir = repo_root / "simulation_logs"
    
    print(f"Looking for logs in: {log_dir}")
    npz_files = glob.glob(str(log_dir / f"*{scenario}*.npz"))
    
    ra_file = None
    is_file = None
    
    # 1. Find the exact Baseline (RA) and IS (tau=0.0) files
    for f in npz_files:
        data = np.load(f, allow_pickle=True)
        mode = str(data.get("mode", "RA"))
        latency = float(data.get("latency", 0.0))
        
        if mode == "RA":
            ra_file = f
        elif mode == "IS" and latency == 0.0:
            is_file = f
            
    if not ra_file or not is_file:
        print("Error: Could not find both RA and IS (tau=0.0) logs in the folder.")
        print(f"Found {len(npz_files)} files matching '{scenario}'.")
        return

    # 2. Extract Data
    ra_data = np.load(ra_file)
    is_data = np.load(is_file)
    
    t_ra, r_ra, psi_ra = ra_data["t"], ra_data["os_r"], ra_data["os_psi"]
    t_is, r_is, psi_is = is_data["t"], is_data["os_r"], is_data["os_psi"]
    
    # 3. Calculate Cumulative Control Effort (Integral of r^2 dt)
    dt_ra = np.diff(t_ra, prepend=0.0)
    dt_is = np.diff(t_is, prepend=0.0)
    
    j_ctrl_ra = np.cumsum(r_ra**2 * dt_ra)
    j_ctrl_is = np.cumsum(r_is**2 * dt_is)

    # 4. Plotting
    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    
    # --- Plot A: Heading ---
    axs[0].plot(t_ra, np.degrees(psi_ra), label="Mode B (RA Baseline)", color='red')
    axs[0].plot(t_is, np.degrees(psi_is), label="Mode A (IS, tau=0.0)", color='blue')
    axs[0].set_ylabel("Heading [deg]")
    axs[0].set_title(f"Heading & Control Effort Diagnostic: {scenario.upper()}")
    axs[0].legend()
    axs[0].grid(True, linestyle=":", alpha=0.7)
    
    # --- Plot B: Yaw Rate ---
    axs[1].plot(t_ra, np.degrees(r_ra), color='red')
    axs[1].plot(t_is, np.degrees(r_is), color='blue', alpha=0.7)
    axs[1].set_ylabel("Yaw Rate [deg/s]")
    axs[1].grid(True, linestyle=":", alpha=0.7)
    
    # --- Plot C: Cumulative J_ctrl ---
    # The 'r' before the string fixes the SyntaxWarning
    axs[2].plot(t_ra, j_ctrl_ra, label=f"RA Final Score: {j_ctrl_ra[-1]:.4f}", color='red')
    axs[2].plot(t_is, j_ctrl_is, label=f"IS Final Score: {j_ctrl_is[-1]:.4f}", color='blue')
    axs[2].set_ylabel(r"Cumulative $J_{ctrl} = \int r^2 dt$")
    axs[2].set_xlabel("Simulation Time [s]")
    axs[2].legend()
    axs[2].grid(True, linestyle=":", alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{scenario}_jctrl_diagnostic.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_control_effort(scenario="case02")