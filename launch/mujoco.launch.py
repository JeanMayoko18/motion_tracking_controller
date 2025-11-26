from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, ExecuteProcess
from launch.substitutions import LaunchConfiguration, EnvironmentVariable, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():

    # Source of the policy: WandB run path OR local ONNX file (XOR)
    wandb_arg = DeclareLaunchArgument(
        "wandb_path",
        default_value="",
        description="Weights & Biases run path (entity/project/run_id or entity/project/run_id/file.onnx).",
    )

    policy_arg = DeclareLaunchArgument(
        "policy_path",
        default_value="",
        description="Local ONNX policy path.",
    )

    # ROS topics for MuJoCo <-> ROS bridge
    state_topic_arg = DeclareLaunchArgument(
        "state_topic",
        default_value="/mujoco/joint_states",
        description="JointState topic from MuJoCo simulator.",
    )

    action_topic_arg = DeclareLaunchArgument(
        "action_topic",
        default_value="/mujoco/joint_torque_cmd",
        description="Action topic sent to MuJoCo (RL actions -> PD controller).",
    )

    # Path to the MuJoCo bridge script (deploy_mujoco_ros.py)
    # Default: $HOME/RL/LeggGym/unitree_rl_gym/deploy/deploy_mujoco/deploy_mujoco_ros.py
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
        description="Python script that runs MuJoCo env + viewer and bridges to ROS.",
    )

    # MuJoCo YAML config file name (inside deploy_mujoco/configs/)
    mujoco_config_arg = DeclareLaunchArgument(
        "mujoco_config",
        default_value="g1.yaml",
        description="YAML config for the MuJoCo environment (e.g. g1.yaml).",
    )

    def launch_setup(context, *args, **kwargs):
        wandb_path = LaunchConfiguration("wandb_path").perform(context)
        policy_path = LaunchConfiguration("policy_path").perform(context)
        state_topic = LaunchConfiguration("state_topic").perform(context)
        action_topic = LaunchConfiguration("action_topic").perform(context)
        mujoco_bridge_script = LaunchConfiguration("mujoco_bridge_script").perform(context)
        mujoco_config = LaunchConfiguration("mujoco_config").perform(context)

        # Enforce XOR: exactly one of wandb_path or policy_path
        if (wandb_path and policy_path) or (not wandb_path and not policy_path):
            raise RuntimeError(
                "\n[MUJOCO LAUNCH] You must set EXACTLY ONE of:\n"
                "  - wandb_path\n"
                "  - policy_path\n\n"
                f"Got: wandb_path='{wandb_path}', policy_path='{policy_path}'\n"
            )

        nodes = []

        # 1) Start MuJoCo simulation + viewer + ROS bridge
        #    NOTE: 'python' here should be the interpreter from your mujoco-rl venv.
        sim_cmd = f"python {mujoco_bridge_script} {mujoco_config}"
        nodes.append(
            ExecuteProcess(
                cmd=["/bin/bash", "-c", sim_cmd],
                output="screen",
            )
        )

        # 2) Start the policy node (ONNX from WandB or local file)
        nodes.append(
            Node(
                package="motion_tracking_controller",
                executable="policy_node",
                name="mujoco_policy_node",
                output="screen",
                parameters=[
                    {
                        "wandb_path": wandb_path,
                        "policy_path": policy_path,
                        "state_topic": state_topic,
                        "action_topic": action_topic,
                        "onnx_input_name": "obs",  # change if your ONNX input name is different
                        "control_rate": 200.0,
                    }
                ],
            )
        )

        return nodes

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
