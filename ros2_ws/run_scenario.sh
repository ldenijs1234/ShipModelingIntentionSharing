#!/bin/bash
# set -e

SCENARIO=${1:-"case01"}
TAU=${2:-0.5}
SHARE_INTENT=${3:-True}
ROUTE_INTERVAL=${4:-3.0}  # Replaced T_ADVANCE with ROUTE_INTERVAL
SPEED_FACTOR=${5:-1.0}
AUTO_CLOSE=${6:-False}  # Default is False for single manual runs
MIN_INTENT_RANGE=${7:-10.0} # Minimum communication range for intent sharing
HEADLESS=${8:-False}    # Default is False for single manual runs

WS_DIR="/mnt/c/Users/lars/Documents/Ship dynamics - Thesis Lars de Nijs/ros2_ws"

echo "=========================================================="
echo "Scenario     : ${SCENARIO}"
echo "Latency (tau): ${TAU} s"
echo "Share Intent : ${SHARE_INTENT}"
echo "Broadcast Int: ${ROUTE_INTERVAL} s"
echo "Auto Close   : ${AUTO_CLOSE}"
echo "Minimum Intent Range : ${MIN_INTENT_RANGE} m"
echo "=========================================================="

source "${WS_DIR}/install/setup.bash"

export AMENT_PREFIX_PATH="${WS_DIR}/install/own_ship:${WS_DIR}/install/communication_layer:${WS_DIR}/install/target_ship:${AMENT_PREFIX_PATH}"
export PYTHONUNBUFFERED=1

ros2 launch own_ship run_scenario.launch.py \
    scenario:=$SCENARIO \
    tau:=$TAU \
    share_intent:=$SHARE_INTENT \
    route_interval:=$ROUTE_INTERVAL \
    speed_factor:=$SPEED_FACTOR \
    auto_close:=$AUTO_CLOSE \
    min_intent_range:=$MIN_INTENT_RANGE \
    headless:=$HEADLESS