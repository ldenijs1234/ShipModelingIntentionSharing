#!/bin/bash
# ==============================================================================
# Batch Runner: Broadcast Interval Sensitivity Sweep
# Keeps Latency (tau) fixed at 0.0s to isolate the effect of route update rates.
# Usage: ./run_batch_interval.sh [scenario] [intent_range] [speed_factor]
# Example: ./run_batch_interval.sh case05 15.0 3.0
# ==============================================================================

SCENARIO=("case05")
INTENT_RANGE=10.0
SPEED_FACTOR=5.0
TAU_FIXED=0.0
INTERVAL_LIST=(1.1 3.0 6.0 11.0 16.0 21.9)

WS_DIR="/mnt/c/Users/lars/Documents/Ship dynamics - Thesis Lars de Nijs/ros2_ws"
REPO_DIR="/mnt/c/Users/lars/Documents/Ship dynamics - Thesis Lars de Nijs"
LOG_DIR="${REPO_DIR}/simulation_logs"

echo "=========================================================="
echo "Starting Interval Sensitivity Sweep for: ${SCENARIO}"
echo "Fixed Latency (tau) : ${TAU_FIXED} s"
echo "Intent Comms Range  : ${INTENT_RANGE} m"
echo "Simulation Speed    : ${SPEED_FACTOR}x"
echo "Intervals to Test   : ${INTERVAL_LIST[*]} s"
echo "=========================================================="

mkdir -p "${LOG_DIR}"

for SCENARIO in "${SCENARIO[@]}"; do
    echo -e "\n\033[92m>>> Running Scenario: ${SCENARIO} <<<\033[0m"
    
    # 1. Run Reactive Avoidance (RA) Baseline
    echo -e "\n\033[96m>>> Running Baseline: Reactive Avoidance (WITHOUT Intent) <<<\033[0m"
    "${WS_DIR}/run_scenario.sh" "${SCENARIO}" "${TAU_FIXED}" False 0.0 "${SPEED_FACTOR}" True "${INTENT_RANGE}" False
    sleep 1

    # 2. Sweep over Broadcast Intervals with Intent Sharing (IS) Active
    for INTERVAL in "${INTERVAL_LIST[@]}"; do
        echo -e "\n\033[93m>>> Running IS Scenario: Interval = ${INTERVAL}s | Latency = ${TAU_FIXED}s <<<\033[0m"
        "${WS_DIR}/run_scenario.sh" "${SCENARIO}" "${TAU_FIXED}" True "${INTERVAL}" "${SPEED_FACTOR}" True "${INTENT_RANGE}" False
        sleep 1
    done
done

# 1. Resolve and navigate to the project root directory
REPO_ROOT="/mnt/c/Users/lars/Documents/Ship dynamics - Thesis Lars de Nijs"
cd "${REPO_ROOT}"

# 2. Run offline batch evaluation from the repo root
python3 -m gnc_core.tests.batch_runner