#!/bin/bash
# set -e

SCENARIO=${1:-"case01"}
TAU=${2:-0.5}
SHARE_INTENT=${3:-True}
T_ADVANCE=${4:-15.0}
SPEED_FACTOR=${5:-1.0}
AUTO_CLOSE=${6:-False}  # Default is False for single manual runs

WS_DIR="/mnt/c/Users/lars/Documents/Ship dynamics - Thesis Lars de Nijs/ros2_ws"

echo "=========================================================="
echo "Scenario     : ${SCENARIO}"
echo "Latency (tau): ${TAU} s"
echo "Share Intent : ${SHARE_INTENT}"
echo "T_advance    : ${T_ADVANCE} s before TCPA"
echo "Auto Close   : ${AUTO_CLOSE}"
echo "=========================================================="

source "${WS_DIR}/install/setup.bash"

export AMENT_PREFIX_PATH="${WS_DIR}/install/own_ship:${WS_DIR}/install/communication_layer:${WS_DIR}/install/target_ship:${AMENT_PREFIX_PATH}"

ros2 launch own_ship run_scenario.launch.py \
    scenario:=$SCENARIO \
    tau:=$TAU \
    share_intent:=$SHARE_INTENT \
    t_advance:=$T_ADVANCE \
    speed_factor:=$SPEED_FACTOR \
    auto_close:=$AUTO_CLOSE