#!/usr/bin/env python3
"""
ROS 2 node that runs an ONNX policy for motion control.

- Loads the policy either from Weights & Biases (wandb_path)
  or from a local ONNX file (policy_path).
- Subscribes to joint states (sensor_msgs/JointState).
- Builds an observation vector from positions and velocities.
- Runs ONNX inference at a fixed control rate.
- Publishes actions to Float64MultiArray on an action topic.

Exactly one of 'wandb_path' or 'policy_path' must be set.
"""

import os
import sys
import pathlib
from typing import Optional, List

import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class PolicyNode(Node):
    def __init__(self) -> None:
        super().__init__("policy_node")

        # ------------------ Parameters ------------------
        # Path to W&B run: "entity/project/run_id"
        self.declare_parameter("wandb_path", "")
        # Local ONNX policy path
        self.declare_parameter("policy_path", "")
        # Joint state topic (simulation or real robot)
        self.declare_parameter("state_topic", "/mujoco/joint_states")
        # Action topic (normalized RL actions)
        self.declare_parameter("action_topic", "/mujoco/joint_torque_cmd")
        # ONNX input name (usually "obs", but can vary)
        self.declare_parameter("onnx_input_name", "obs")
        # Control rate in Hz
        self.declare_parameter("control_rate", 200.0)

        self.wandb_path: str = self.get_parameter("wandb_path").get_parameter_value().string_value
        self.policy_path: str = self.get_parameter("policy_path").get_parameter_value().string_value
        self.state_topic: str = self.get_parameter("state_topic").get_parameter_value().string_value
        self.action_topic: str = self.get_parameter("action_topic").get_parameter_value().string_value
        self.onnx_input_name: str = self.get_parameter("onnx_input_name").get_parameter_value().string_value
        self.control_rate: float = self.get_parameter("control_rate").get_parameter_value().double_value

        # ------------------ Check ONNX runtime ------------------
        if ort is None:
            self.get_logger().error(
                "onnxruntime is not installed. Please install it in this environment:\n"
                "  pip install onnxruntime"
            )
            # Fatal error: node cannot operate without ONNX runtime
            rclpy.shutdown()
            sys.exit(1)

        # ------------------ Enforce XOR wandb_path / policy_path ------------------
        if (self.wandb_path and self.policy_path) or (not self.wandb_path and not self.policy_path):
            self.get_logger().error(
                "You must set EXACTLY ONE of the following parameters:\n"
                "  - 'wandb_path' (Weights & Biases run path)\n"
                "  - 'policy_path' (local ONNX file)\n"
                f"Current values:\n  wandb_path='{self.wandb_path}'\n  policy_path='{self.policy_path}'"
            )
            rclpy.shutdown()
            sys.exit(1)

        # ------------------ Load ONNX policy ------------------
        self.session: Optional[ort.InferenceSession] = None
        self.obs_dim: Optional[int] = None
        self._load_policy()

        # ------------------ ROS I/O ------------------
        self.latest_joint_state: Optional[JointState] = None

        self.state_sub = self.create_subscription(
            JointState,
            self.state_topic,
            self.joint_state_callback,
            10,
        )

        self.action_pub = self.create_publisher(
            Float64MultiArray,
            self.action_topic,
            10,
        )

        # Timer for periodic control loop
        if self.control_rate <= 0.0:
            self.get_logger().warn("Control rate <= 0.0, forcing to 100.0 Hz")
            self.control_rate = 100.0
        self.dt = 1.0 / self.control_rate
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(
            "✅ PolicyNode initialized\n"
            f"  wandb_path={self.wandb_path}\n"
            f"  policy_path={self.policy_path}\n"
            f"  state_topic={self.state_topic}\n"
            f"  action_topic={self.action_topic}\n"
            f"  onnx_input_name={self.onnx_input_name}\n"
            f"  control_rate={self.control_rate} Hz"
        )

    # ---------------------------------------------------------
    # Policy loading utilities
    # ---------------------------------------------------------

    def _load_policy(self) -> None:
        """Load ONNX policy either from WandB or from a local path."""
        if self.wandb_path:
            policy_file = self._download_policy_from_wandb(self.wandb_path)
            if policy_file is None:
                self.get_logger().error("Failed to download ONNX policy from WandB. Aborting.")
                rclpy.shutdown()
                sys.exit(1)
            self.policy_path = str(policy_file)

        # At this point, self.policy_path must be a valid ONNX file path
        if not os.path.isfile(self.policy_path):
            self.get_logger().error(f"Policy file does not exist: {self.policy_path}")
            rclpy.shutdown()
            sys.exit(1)

        self.get_logger().info(f"Loading ONNX policy from: {self.policy_path}")
        try:
            self.session = ort.InferenceSession(self.policy_path, providers=["CPUExecutionProvider"])
        except Exception as e:
            self.get_logger().error(f"Failed to create ONNXRuntime session: {e}")
            rclpy.shutdown()
            sys.exit(1)

        # Infer observation dimension from ONNX model input shape if possible
        try:
            inputs = self.session.get_inputs()
            input_names = [i.name for i in inputs]
            if self.onnx_input_name not in input_names:
                self.get_logger().warn(
                    f"Input name '{self.onnx_input_name}' not found in ONNX model inputs: {input_names}. "
                    f"Using first input '{inputs[0].name}' instead."
                )
                self.onnx_input_name = inputs[0].name

            input_tensor = None
            for i in inputs:
                if i.name == self.onnx_input_name:
                    input_tensor = i
                    break

            if input_tensor is not None:
                # Typical ONNX input shape: [batch, obs_dim]
                # Some models may use dynamic dimensions (-1 or 'None')
                shape = list(input_tensor.shape)
                if len(shape) == 2:
                    # batch x obs_dim
                    self.obs_dim = int(shape[1]) if isinstance(shape[1], int) else None
                elif len(shape) == 1:
                    # obs_dim only
                    self.obs_dim = int(shape[0]) if isinstance(shape[0], int) else None
                else:
                    self.get_logger().warn(
                        f"Unexpected input shape for '{self.onnx_input_name}': {shape}. "
                        "Observation dimension will be inferred at runtime."
                    )
            else:
                self.get_logger().warn(
                    f"Could not find input tensor for name '{self.onnx_input_name}'. "
                    "Observation dimension will be inferred at runtime."
                )
        except Exception as e:
            self.get_logger().warn(f"Could not infer observation shape from ONNX model: {e}")
            self.obs_dim = None

    def _download_policy_from_wandb(self, wandb_path: str) -> Optional[pathlib.Path]:
        """
        Download the latest ONNX policy file from a WandB run.

        wandb_path format:
            'entity/project/run_id'       -> auto-detect ONNX file
            'entity/project/run_id/file'  -> download specific file

        Returns:
            Path to downloaded ONNX file, or None on failure.
        """
        try:
            import wandb
        except ImportError:
            self.get_logger().error(
                "wandb is not installed. Install it with:\n  pip install wandb\n"
                "or use a local 'policy_path' instead."
            )
            return None

        api = wandb.Api()

        # If wandb_path includes a file name (4 components), split it
        parts = wandb_path.split("/")
        file_name = None
        if len(parts) == 4:
            run_path = "/".join(parts[:3])
            file_name = parts[3]
        else:
            run_path = wandb_path

        try:
            run = api.run(run_path)
        except Exception as e:
            self.get_logger().error(f"Failed to access W&B run '{run_path}': {e}")
            return None

        # Find ONNX file if none specified
        if file_name is None:
            candidates = [
                f.name for f in run.files()
                if f.name.endswith(".onnx") or "policy" in f.name.lower() or "model" in f.name.lower()
            ]
            if not candidates:
                self.get_logger().error(
                    f"No ONNX-like file found in W&B run '{run_path}'. "
                    "Please specify the exact file name in wandb_path (entity/project/run/file.onnx)."
                )
                return None
            # Pick the last one (often the latest)
            candidates.sort()
            file_name = candidates[-1]

        self.get_logger().info(f"Downloading ONNX policy '{file_name}' from W&B run '{run_path}'...")

        cache_dir = pathlib.Path.home() / ".cache" / "motion_tracking_controller"
        cache_dir.mkdir(parents=True, exist_ok=True)
        dest_path = cache_dir / file_name

        try:
            wandb_file = run.file(file_name)
            wandb_file.download(root=str(cache_dir), replace=True)
        except Exception as e:
            self.get_logger().error(f"Failed to download file '{file_name}' from WandB: {e}")
            return None

        self.get_logger().info(f"Policy downloaded to: {dest_path}")
        return dest_path

    # ---------------------------------------------------------
    # ROS Callbacks and control loop
    # ---------------------------------------------------------

    def joint_state_callback(self, msg: JointState) -> None:
        """Store the latest joint state message."""
        self.latest_joint_state = msg

    def control_loop(self) -> None:
        """
        Periodic control loop:
        - Build observation from latest joint state.
        - Run ONNX policy.
        - Publish Float64MultiArray actions.
        """
        if self.session is None:
            # Policy not loaded
            return

        if self.latest_joint_state is None:
            # No joint state received yet
            return

        # Build observation from joint state
        obs = self._build_observation(self.latest_joint_state)
        if obs is None:
            return

        # Add batch dimension: [1, obs_dim]
        obs_batch = obs[np.newaxis, :].astype(np.float32)

        # ONNX inference
        try:
            outputs = self.session.run(None, {self.onnx_input_name: obs_batch})
        except Exception as e:
            self.get_logger().error(f"ONNX inference failed: {e}")
            return

        if not outputs:
            self.get_logger().error("ONNX model returned no outputs.")
            return

        actions = outputs[0]
        # Expected shape: [1, action_dim] or [action_dim]
        actions = np.asarray(actions).squeeze()
        if actions.ndim != 1:
            self.get_logger().warn(f"Unexpected action shape: {actions.shape}. Flattening.")
            actions = actions.flatten()

        # Publish actions as Float64MultiArray
        msg = Float64MultiArray()
        msg.data = actions.astype(float).tolist()
        self.action_pub.publish(msg)

    def _build_observation(self, js: JointState) -> Optional[np.ndarray]:
        """
        Build a simple observation vector from JointState.

        Current implementation:
            obs = [positions, velocities]

        If obs_dim is known from ONNX, the vector is padded or truncated to match.
        """
        positions: List[float] = list(js.position) if js.position else []
        velocities: List[float] = list(js.velocity) if js.velocity else []

        obs_vec = np.array(positions + velocities, dtype=np.float32)

        if obs_vec.size == 0:
            self.get_logger().warn("JointState has empty position and velocity; cannot build observation.")
            return None

        if self.obs_dim is not None:
            if obs_vec.size < self.obs_dim:
                # Pad with zeros
                padded = np.zeros(self.obs_dim, dtype=np.float32)
                padded[: obs_vec.size] = obs_vec
                obs_vec = padded
            elif obs_vec.size > self.obs_dim:
                # Truncate
                obs_vec = obs_vec[: self.obs_dim]

        return obs_vec


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
