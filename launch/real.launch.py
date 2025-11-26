from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # Robot IP (where unitree_sdk2 is installed)
    robot_ip_arg = DeclareLaunchArgument(
        "robot_ip",
        default_value="192.168.1.117",
        description="IP address of the Unitree G1 robot (user: unitree).",
    )

    # Network interface name on the robot (e.g. eth0)
    network_if_arg = DeclareLaunchArgument(
        "network_interface",
        default_value="eth0",
        description="Network interface used by g1_loco_client on the robot (e.g. eth0).",
    )

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

    # ROS topics used on the PC side (bridge to the robot)
    state_topic_arg = DeclareLaunchArgument(
        "state_topic",
        default_value="/real_robot/joint_states",
        description="JointState topic from the real robot (PC side).",
    )

    action_topic_arg = DeclareLaunchArgument(
        "action_topic",
        default_value="/real_robot/joint_torque_cmd",
        description="Action topic to the real robot (PC side).",
    )

    # g1_loco_client mode: zero_torque, debug, damp, etc.
    loco_mode_arg = DeclareLaunchArgument(
        "loco_mode",
        default_value="zero_torque",
        description="Mode passed to g1_loco_client on the robot (e.g. zero_torque, debug, damp).",
    )

    def launch_setup(context, *args, **kwargs):
        robot_ip = LaunchConfiguration("robot_ip").perform(context)
        network_interface = LaunchConfiguration("network_interface").perform(context)
        wandb_path = LaunchConfiguration("wandb_path").perform(context)
        policy_path = LaunchConfiguration("policy_path").perform(context)
        state_topic = LaunchConfiguration("state_topic").perform(context)
        action_topic = LaunchConfiguration("action_topic").perform(context)
        loco_mode = LaunchConfiguration("loco_mode").perform(context)

        # Enforce XOR: exactly one of wandb_path or policy_path
        if (wandb_path and policy_path) or (not wandb_path and not policy_path):
            raise RuntimeError(
                "\n[REAL LAUNCH] You must set EXACTLY ONE of:\n"
                "  - wandb_path\n"
                "  - policy_path\n\n"
                f"Got: wandb_path='{wandb_path}', policy_path='{policy_path}'\n"
            )

        nodes = []

        # 1) Start g1_loco_client on the robot via SSH
        #
        #    This assumes:
        #      - SSH keys are configured (no password prompt)
        #      - unitree_sdk2 is located in ~/unitree_sdk2/build/bin

        # ------------------------------------------------------------
        # Launch the Unitree SDK low-level controller on the robot
        # via SSH. This runs only on the robot, while the ONNX policy
        # runs on the local PC.
        # ------------------------------------------------------------

        # Build the remote command executed directly on the robot.
        # We must `cd` into the SDK directory because g1_loco_client
        # is NOT in the robot's PATH.
        #
        # Example remote execution:
        #   cd ~/unitree_sdk2/build/bin &&
        #   ./g1_loco_client --network_interface=eth0 --zero_torque
        #
        loco_cmd_remote = (
            f"cd ~/unitree_sdk2/build/bin && "
            f"./g1_loco_client --network_interface={network_interface} --{loco_mode}"
        )

        # Build the SSH command that will be executed from the PC.
        # - StrictHostKeyChecking=no prevents SSH from blocking if
        #   this is the first time connecting to the robot.
        # - The entire remote command is placed inside quotes so that
        #   bash executes it as a single block.
        #
        ssh_cmd = f'ssh -o StrictHostKeyChecking=no unitree@{robot_ip} "{loco_cmd_remote}"'

        # Create the ROS2 ExecuteProcess node that launches the SSH
        # command. This will run the robot’s low-level controller.
        #
        # Options:
        #   - shell=True:
        #       Ensures that complex commands (with &&, quotes, etc.)
        #       are interpreted correctly by the shell.
        #
        #   - respawn=False:
        #       VERY IMPORTANT for real hardware.
        #       If the SSH session dies or g1_loco_client exits, we
        #       do NOT want ROS to automatically relaunch it. This
        #       avoids unsafe behavior on the real robot.
        #
        #   - output='screen':
        #       Streams SSH output to the ROS terminal for debugging.
        #
        nodes.append(
            ExecuteProcess(
                cmd=["/bin/bash", "-c", ssh_cmd],
                shell=True,
                respawn=False,
                output="screen",
            )
        )

        # 2) Start the ONNX policy node on the PC
        nodes.append(
            Node(
                package="motion_tracking_controller",
                executable="policy_node",
                name="real_policy_node",
                output="screen",
                parameters=[
                    {
                        "wandb_path": wandb_path,
                        "policy_path": policy_path,
                        "state_topic": state_topic,
                        "action_topic": action_topic,
                        "onnx_input_name": "obs",
                        "control_rate": 200.0,
                    }
                ],
            )
        )

        return nodes

    return LaunchDescription(
        [
            robot_ip_arg,
            network_if_arg,
            wandb_arg,
            policy_arg,
            state_topic_arg,
            action_topic_arg,
            loco_mode_arg,
            OpaqueFunction(function=launch_setup),
        ]
    )
