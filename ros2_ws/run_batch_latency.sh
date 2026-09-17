#!/bin/bash
# set -e

# Clear or archive old logs if needed (optional)
# rm -rf simulation_logs/*.npz

# Start Master Stopwatch
START_TIME=$(date +%s)

echo "=========================================================="
echo "Starting Automated Simulation Batch"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================="

SCENARIOS=("case05")
INTERVALS=(1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0)
LATENCIES=(0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0)

TOTAL_RUNS=$(( ${#SCENARIOS[@]} * (1 + ${#INTERVALS[@]} * ${#LATENCIES[@]}) ))
CURRENT_RUN=0

for SCENARIO in "${SCENARIOS[@]}"; do
    CURRENT_RUN=$((CURRENT_RUN + 1))
    echo ">>> Running Baseline (RA) for ${SCENARIO}..."
    # Mode B: Baseline Reactive Avoidance (Run once per scenario)
    # Parameters: SCENARIO, TAU, SHARE_INTENT, INTERVAL, SPEED, AUTO_CLOSE, INTENT_RANGE, HEADLESS
    ./run_scenario.sh "${SCENARIO}" 0.0 False 0.0 5.0 True 15.0 True

    for INTERVAL in "${INTERVALS[@]}"; do
        for TAU in "${LATENCIES[@]}"; do
            CURRENT_RUN=$((CURRENT_RUN + 1))
            echo ">>> Running IS for ${SCENARIO} | Interval=${INTERVAL}s | Tau=${TAU}s..."
            ./run_scenario.sh "${SCENARIO}" "${TAU}" True "${INTERVAL}" 5.0 True 15.0 True
        done
    done
done

# Stop Master Stopwatch
END_TIME=$(date +%s)
ELAPSED_SEC=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_SEC / 3600))
MINUTES=$(((ELAPSED_SEC % 3600) / 60))
SECONDS=$((ELAPSED_SEC % 60))

echo "=========================================================="
printf "All %d simulations completed in: %02dh:%02dm:%02ds\n" "${TOTAL_RUNS}" "${HOURS}" "${MINUTES}" "${SECONDS}"
echo "Running KPI evaluation..."
echo "=========================================================="

# 1. Resolve and navigate to the project root directory
REPO_ROOT="/mnt/c/Users/lars/Documents/Ship dynamics - Thesis Lars de Nijs"
cd "${REPO_ROOT}"

# 2. Run offline batch evaluation from the repo root
python3 -m gnc_core.tests.batch_runner