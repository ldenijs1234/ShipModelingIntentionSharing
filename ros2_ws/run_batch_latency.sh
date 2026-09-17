#!/bin/bash
# set -e

# Clear or archive old logs if needed (optional)
# rm -rf simulation_logs/*.npz

echo "=========================================================="
echo "Starting Automated Simulation Batch"
echo "=========================================================="

SCENARIOS=("case05")
LATENCIES=(0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0)

for SCENARIO in "${SCENARIOS[@]}"; do
    echo ">>> Running Baseline (RA) for ${SCENARIO}..."
    # Mode B: SHARE_INTENT=False, TAU=0.0
    # Added "True" at the end for AUTO_CLOSE
    ./run_scenario.sh "${SCENARIO}" 0.0 False 0.0 5.0 True 15.0 True

    # Mode A: Intention Sharing swept across latency values
    for TAU in "${LATENCIES[@]}"; do
        echo ">>> Running Intention Sharing (IS) for ${SCENARIO} with tau=${TAU}s..."
        # Added "True" at the end for AUTO_CLOSE
        ./run_scenario.sh "${SCENARIO}" "${TAU}" True 5.0 5.0 True 15.0 True
    done
done

echo "=========================================================="
echo "All simulations completed! Running KPI evaluation..."
echo "=========================================================="

# 1. Resolve and navigate to the project root directory
REPO_ROOT="/mnt/c/Users/lars/Documents/Ship dynamics - Thesis Lars de Nijs"
cd "${REPO_ROOT}"

# 2. Run offline batch evaluation from the repo root
python3 -m gnc_core.tests.batch_runner