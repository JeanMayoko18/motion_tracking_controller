#!/usr/bin/env bash
# ============================================================
# Launcher for ROS2 + MuJoCo + Policy (WandB / ONNX)
# ============================================================

set -e

echo "[INFO] Script running under: $(ps -p $$ -o comm=)"
echo "[INFO] Login shell (SHELL): $SHELL"

# ------------------------------------------------------------
# 1) Source ROS 2 Humble (BASH version)
# ------------------------------------------------------------
echo "[INFO] Sourcing ROS2 Humble (setup.bash)"
source /opt/ros/humble/setup.bash

# ------------------------------------------------------------
# 2) Source ROS workspace
# ------------------------------------------------------------
WS="$HOME/ros2rob_ws"
echo "[INFO] Sourcing workspace: $WS"

if [ -d "$WS/install" ]; then
    source "$WS/install/setup.bash"
else
    echo "[ERROR] Workspace is not built. Please run:"
    echo "  cd $WS && colcon build"
    exit 1
fi

# ------------------------------------------------------------
# 3) Activate the Python virtual environment (mujoco-rl)
#    (this MUST be AFTER ROS so that venv's python is first in PATH)
# ------------------------------------------------------------
VENV_PATH="$HOME/mujoco-rl"

if [ -d "$VENV_PATH" ]; then
    echo "[INFO] Activating venv: $VENV_PATH"
    source "$VENV_PATH/bin/activate"
else
    echo "[ERROR] Virtual environment not found at: $VENV_PATH"
    exit 1
fi

echo "[INFO] Python interpreter used by nodes: $(which python3)"

# ------------------------------------------------------------
# 4) LeggedGym Python path (for deploy_mujoco_ros.py)
# ------------------------------------------------------------
export LEGGM_PATH="$HOME/RL/LeggGym/unitree_rl_gym"
export PYTHONPATH="$LEGGM_PATH:$PYTHONPATH"
echo "[INFO] PYTHONPATH configured for LeggedGym: $LEGGM_PATH"

# ------------------------------------------------------------
# 5) Forward arguments to ros2 launch
# ------------------------------------------------------------
echo "[INFO] Running ros2 launch motion_tracking_controller mujoco.launch.py $@"
ros2 launch motion_tracking_controller mujoco.launch.py "$@"

