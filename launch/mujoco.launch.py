from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, ExecuteProcess
from launch.substitutions import (
    LaunchConfiguration,
    EnvironmentVariable,
    PathJoinSubstitution,
)
from launch_ros.actions import Node

import os
import sys


def generate_launch_description():
    # --------------------------------------------------------------
    # Launch arguments
    # The user must specify EITHER:
    #   - a Weights & Biases run (wandb_path)
    #   - OR a local ONNX file (policy_path)
    # Exactly one must be non-empty (XOR).
    # --------------------------------------------------------------

    wandb_arg = DeclareLaunchArgument(
        "wandb_path",
        default_value="",
        description=(
            "Weights & Biases run path "
            "(entity/project/run_id OR entity/project/run_id/file.onnx)."
        ),
    )

    policy_arg = DeclareLaunchArgument(
        "policy_path",
        default_value="",
        description="Path to a local ONNX policy file.",
    )

    # --------------------------------------------------------------
    # ROS topics used for MuJoCo <-> ROS2 communication
    # --------------------------------------------------------------

    state_topic_arg = DeclareLaunchArgument(
        "state_topic",
        default_value="/mujoco/joint_states",
        description="The JointState topic coming from the MuJoCo simulator.",
    )

    action_topic_arg = DeclareLaunchArgument(
        "action_topic",
        default_value="/mujoco/joint_torque_cmd",
        description="Topic used to publish RL actions (torques) back to MuJoCo.",
    )

    # --------------------------------------------------------------
    # Location of the MuJoCo → ROS2 bridge script
    # This script launches:
    #   - the MuJoCo simulation,
    #   - the viewer,
    #   - and publishes ROS topics.
    # Default location is inside the LeggedGym repository.
    # --------------------------------------------------------------

    mujoco_bridge_script_arg = DeclareLaunchArgument(
        "mujoco_bridge_script",
        default_value=PathJoinSubstitution(
            [
                EnvironmentVariable("HOME"),
                "RL",
                "LeggGym",
                "unitree_rl_gym",
                "deploy",
                "deploy_mujoco",
                "deploy_mujoco_ros.py",
            ]
        ),
        description="Python script that launches the MuJoCo simulator and bridges it to ROS2.",
    )

    # --------------------------------------------------------------
    # The YAML configuration file used by MuJoCo
    # This file lives inside: deploy_mujoco/configs/
    # --------------------------------------------------------------

    mujoco_config_arg = DeclareLaunchArgument(
        "mujoco_config",
        default_value="g1_29.yaml",
        description="MuJoCo config file used by deploy_mujoco_ros.py (e.g. g1_29.yaml).",
    )

    # ==================================================================
    #  Custom launch setup function
    # ==================================================================
    def launch_setup(context, *args, **kwargs):

        # Retrieve all arguments evaluated by ROS launch
        wandb_path = LaunchConfiguration("wandb_path").perform(context)
        policy_path = LaunchConfiguration("policy_path").perform(context)
        state_topic = LaunchConfiguration("state_topic").perform(context)
        action_topic = LaunchConfiguration("action_topic").perform(context)
        mujoco_bridge_script = LaunchConfiguration("mujoco_bridge_script").perform(context)
        mujoco_config = LaunchConfiguration("mujoco_config").perform(context)

        # --------------------------------------------------------------
        # XOR CHECK:
        # The user MUST provide exactly one of:
        #   - wandb_path (download .pt/.onnx from WandB)
        #   - policy_path (load local .onnx)
        # --------------------------------------------------------------
        if (wandb_path and policy_path) or (not wandb_path and not policy_path):
            raise RuntimeError(
                "\n[MUJOCO LAUNCH] You must set EXACTLY ONE of:\n"
                "  - wandb_path\n"
                "  - policy_path\n\n"
                f"Got: wandb_path='{wandb_path}', policy_path='{policy_path}'\n"
            )

        nodes = []

        # --------------------------------------------------------------
        # 0) Choose which Python interpreter should run policy_node
        #
        # - Prefer ROS_PYTHON_EXECUTABLE (exported by run_mujoco_ros.sh)
        # - If not found, fallback to sys.executable (current interpreter)
        #
        # This ensures the policy node ALWAYS runs inside the mujoco-rl venv,
        # so 'wandb', 'onnxruntime', IsaacLab, etc. are available.
        # --------------------------------------------------------------
        python_exec = os.environ.get("ROS_PYTHON_EXECUTABLE", sys.executable)

        # --------------------------------------------------------------
        # 0bis) Resolve the path to export_rslrl_to_onnx_from_ckpt.py dynamically
        #
        # Priority:
        #   1) Environment variable EXPORT_RSLRL_TO_ONNX (if set)
        #   2) RL_ROOT/export_rslrl_to_onnx_from_ckpt.py (if RL_ROOT is set)
        #   3) Common locations under $HOME (RL, Documents/RL, Desktop/RL)
        #   4) Fallback to $HOME/RL/export_rslrl_to_onnx_from_ckpt.py
        # --------------------------------------------------------------
        export_script = os.environ.get("EXPORT_RSLRL_TO_ONNX", "")

        if not export_script:
            # Try RL_ROOT first, if the user has set it
            rl_root = os.environ.get("RL_ROOT", "")
            candidates = []

            if rl_root:
                candidates.append(os.path.join(rl_root, "export_rslrl_to_onnx_from_ckpt.py"))

            home = os.environ.get("HOME", "")
            if home:
                candidates.extend(
                    [
                        os.path.join(home, "RL", "export_rslrl_to_onnx_from_ckpt.py"),
                        os.path.join(home, "Documents", "RL", "export_rslrl_to_onnx_from_ckpt.py"),
                        os.path.join(home, "Documentos", "RL", "export_rslrl_to_onnx_from_ckpt.py"),
                        os.path.join(home, "Desktop", "RL", "export_rslrl_to_onnx_from_ckpt.py"),
                    ]
                )

            # Default fallback: ~/RL/export_rslrl_to_onnx_from_ckpt.py
            fallback = os.path.join(home, "RL", "export_rslrl_to_onnx_from_ckpt.py") if home else ""

            export_script = fallback
            for cand in candidates:
                if os.path.isfile(cand):
                    export_script = cand
                    break

        # Log the chosen export script path
        print(f"[MUJOCO LAUNCH] export_from_ckpt_script resolved to: {export_script}")

        # --------------------------------------------------------------
        # 1) Start MuJoCo simulation and ROS bridge
        #
        #    The command uses "python" because the run script activates the
        #    mujoco-rl venv before launching this file.
        # --------------------------------------------------------------
        sim_cmd = f"python {mujoco_bridge_script} {mujoco_config}"

        nodes.append(
            ExecuteProcess(
                cmd=["/bin/bash", "-c", sim_cmd],
                output="screen",
            )
        )

        # --------------------------------------------------------------
        # 2) Start policy_node (the ONNX policy inference node)
        #
        # - prefix=[python_exec, " "] forces the executable to be launched
        #   using the virtual environment Python rather than the system one.
        #
        # - wandb_path/policy_path select how the ONNX file is obtained
        #
        # - extra parameters enable automatic .pt → .onnx export
        #   with the correct IsaacLab task name.
        # --------------------------------------------------------------
        nodes.append(
            Node(
                package="motion_tracking_controller",
                executable="policy_node",
                name="mujoco_policy_node",
                output="screen",
                # Force execution inside mujoco-rl virtualenv
                prefix=[python_exec, " "],
                parameters=[
                    {
                        "wandb_path": wandb_path,
                        "policy_path": policy_path,
                        "state_topic": state_topic,
                        "action_topic": action_topic,
                        "onnx_input_name": "obs",
                        "control_rate": 200.0,
                        # Additional parameters for automatic checkpoint export
                        "export_from_ckpt_script": export_script,
                        # IsaacLab task name used for env + RSL-RL config
                        "export_task_name": "Tracking-Flat-G1-v0",
                        "export_num_envs": 1,
                    }
                ],
            )
        )

        return nodes

    # ==================================================================
    # Return the LaunchDescription that ROS2 will execute
    # ==================================================================
    return LaunchDescription(
        [
            wandb_arg,
            policy_arg,
            state_topic_arg,
            action_topic_arg,
            mujoco_bridge_script_arg,
            mujoco_config_arg,
            OpaqueFunction(function=launch_setup),
        ]
    )
