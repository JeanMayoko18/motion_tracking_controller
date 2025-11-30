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

This version also supports ONNX models that require an additional
input (e.g. `time_step`). If the ONNX model has an input named
'time_step', the node will automatically feed a step counter.
"""

import os
import sys
import pathlib
import subprocess
from typing import Optional, List, Dict

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
        # Path to W&B run: "entity/project/run_id" or "entity/project/run_id/file"
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

        # ---------------------------------------------------------
        # Export-from-checkpoint parameters (dynamic)
        # ---------------------------------------------------------

        # Build default export script path dynamically
        default_export_script = str(pathlib.Path.home() / "RL" / "export_rslrl_to_onnx_from_ckpt.py")

        self.declare_parameter(
            "export_from_ckpt_script",
            default_export_script,
        )

        # Task name MUST match actual IsaacLab training task
        self.declare_parameter(
            "export_task_name",
            "Tracking-Flat-G1-v0",
        )

        # Dummy number of envs for export
        self.declare_parameter("export_num_envs", 1)


        # Read parameters
        self.wandb_path: str = self.get_parameter("wandb_path").get_parameter_value().string_value
        self.policy_path: str = self.get_parameter("policy_path").get_parameter_value().string_value
        self.state_topic: str = self.get_parameter("state_topic").get_parameter_value().string_value
        self.action_topic: str = self.get_parameter("action_topic").get_parameter_value().string_value
        self.onnx_input_name: str = self.get_parameter("onnx_input_name").get_parameter_value().string_value
        self.control_rate: float = self.get_parameter("control_rate").get_parameter_value().double_value

        self.export_from_ckpt_script: str = (
            self.get_parameter("export_from_ckpt_script").get_parameter_value().string_value
        )
        self.export_task_name: str = (
            self.get_parameter("export_task_name").get_parameter_value().string_value
        )
        self.export_num_envs: int = (
            self.get_parameter("export_num_envs").get_parameter_value().integer_value
        )

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

        # ------------------ ONNX input bookkeeping ------------------
        # Will store ONNX input NodeArg objects by name.
        self.onnx_inputs: Dict[str, "ort.NodeArg"] = {}
        # Name of the auxiliary time-step input if the model requires it.
        self.time_step_name: Optional[str] = None
        # Internal step counter used when feeding 'time_step'.
        self.step_counter: int = 0

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
            f"  control_rate={self.control_rate} Hz\n"
            f"  export_from_ckpt_script={self.export_from_ckpt_script}\n"
            f"  export_task_name={self.export_task_name}\n"
            f"  export_num_envs={self.export_num_envs}"
        )

    # ---------------------------------------------------------
    # Policy loading utilities
    # ---------------------------------------------------------

    def _load_policy(self) -> None:
        """Load ONNX policy either from WandB or from a local path."""
        if self.wandb_path:
            policy_file = self._download_policy_from_wandb(self.wandb_path)
            if policy_file is None:
                self.get_logger().error("Failed to obtain ONNX policy from WandB. Aborting.")
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

        # -----------------------------------------------------
        # Infer observation dimension and detect extra inputs
        # (such as 'time_step') from the ONNX model signature.
        # -----------------------------------------------------
        try:
            inputs = self.session.get_inputs()
            # Keep a dict of all input NodeArg objects
            self.onnx_inputs = {i.name: i for i in inputs}
            input_names = list(self.onnx_inputs.keys())

            # --- Handle main observation input (e.g. 'obs') ---
            if self.onnx_input_name not in input_names:
                self.get_logger().warn(
                    f"Input name '{self.onnx_input_name}' not found in ONNX model inputs: {input_names}. "
                    f"Using first input '{inputs[0].name}' instead."
                )
                self.onnx_input_name = inputs[0].name

            input_tensor = self.onnx_inputs[self.onnx_input_name]

            # Typical ONNX input shape: [batch, obs_dim] or [obs_dim]
            shape = list(input_tensor.shape)
            if len(shape) == 2:
                # [batch, obs_dim]
                self.obs_dim = int(shape[1]) if isinstance(shape[1], int) else None
            elif len(shape) == 1:
                # [obs_dim]
                self.obs_dim = int(shape[0]) if isinstance(shape[0], int) else None
            else:
                self.get_logger().warn(
                    f"Unexpected input shape for '{self.onnx_input_name}': {shape}. "
                    "Observation dimension will be inferred at runtime."
                )

            # --- Detect auxiliary input: 'time_step' (if present) ---
            if "time_step" in self.onnx_inputs:
                self.time_step_name = "time_step"
                self.step_counter = 0
                self.get_logger().info(
                    f"ONNX model expects an additional input '{self.time_step_name}'. "
                    "A step counter will be provided automatically."
                )
            else:
                self.get_logger().info(
                    f"ONNX model inputs: {input_names}. No 'time_step' input detected."
                )

        except Exception as e:
            self.get_logger().warn(f"Could not infer input shapes from ONNX model: {e}")
            self.obs_dim = None
            self.onnx_inputs = {}
            self.time_step_name = None

    def _auto_find_export_script(self) -> Optional[pathlib.Path]:
        """
        Automatically search for export_rslrl_to_onnx_from_ckpt.py
        anywhere in the user's home directory.

        This allows the script to be placed in different locations
        (e.g. ~/RL, ~/Documents/RL) and still be discovered.
        """

        home = pathlib.Path.home()
        script_name = "export_rslrl_to_onnx_from_ckpt.py"

        # ░░░ Candidate folders to speed-up search ░░░
        candidates = [
            home / "RL",
            home / "Documents",
            home / "Documents/RL",
            home / "Desktop",
            home / "Desktop/RL",
        ]

        # 1) Fast search in known locations
        for folder in candidates:
            path = folder / script_name
            if path.is_file():
                return path

        # 2) Full recursive search (slower, but done only once)
        for root, dirs, files in os.walk(home):
            if script_name in files:
                return pathlib.Path(root) / script_name

        return None

    def _download_policy_from_wandb(self, wandb_path: str) -> Optional[pathlib.Path]:
        """
        Download a policy checkpoint from WandB and ensure we end up with an ONNX file.

        Behavior:
        - If the W&B run already contains an .onnx file, we download it and return it.
        - If the W&B run only has a .pt checkpoint:
            * We download the .pt into ~/.cache/motion_tracking_controller/
            * We export it to .onnx using an external script (export_rslrl_to_onnx_from_ckpt.py)
            * We cache the resulting .onnx and return its path.
        - On next launches, if the .onnx already exists, we reuse it directly
          (no new export).

        wandb_path formats:
            'entity/project/run_id'
            'entity/project/run_id/file_name.pt' or .onnx
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

        # Split wandb_path into run path and optionally a file name
        parts = wandb_path.split("/")
        file_name: Optional[str] = None
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

        cache_dir = pathlib.Path.home() / ".cache" / "motion_tracking_controller"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # -------------------------------------------------------------
        # 1) If file_name is not specified, try to choose a good candidate
        # -------------------------------------------------------------
        if file_name is None:
            # Prefer ONNX files if they exist
            onnx_candidates = [
                f.name for f in run.files()
                if f.name.endswith(".onnx")
            ]
            if onnx_candidates:
                onnx_candidates.sort()
                file_name = onnx_candidates[-1]  # last one is usually latest
            else:
                # Otherwise look for model/policy checkpoints (e.g. .pt)
                pt_candidates = [
                    f.name for f in run.files()
                    if f.name.endswith(".pt") or "model" in f.name.lower()
                ]
                if not pt_candidates:
                    self.get_logger().error(
                        f"No ONNX or checkpoint (.pt) files found in W&B run '{run_path}'."
                    )
                    return None
                pt_candidates.sort()
                file_name = pt_candidates[-1]

        # At this point, file_name is chosen
        self.get_logger().info(
            f"Selected W&B file '{file_name}' from run '{run_path}'."
        )

        # Local path for the downloaded file
        dest_path = cache_dir / file_name

        # -------------------------------------------------------------
        # 2) If it is already an ONNX file → just download & return
        # -------------------------------------------------------------
        if file_name.endswith(".onnx"):
            # Download ONNX directly if not already present
            if not dest_path.is_file():
                self.get_logger().info(
                    f"Downloading ONNX policy '{file_name}' from W&B to '{dest_path}'..."
                )
                try:
                    wandb_file = run.file(file_name)
                    wandb_file.download(root=str(cache_dir), replace=True)
                except Exception as e:
                    self.get_logger().error(
                        f"Failed to download ONNX file '{file_name}' from W&B: {e}"
                    )
                    return None
            else:
                self.get_logger().info(
                    f"ONNX file already in cache: '{dest_path}', reusing it."
                )
            self.get_logger().info(f"Policy ready at: {dest_path}")
            return dest_path

        # -------------------------------------------------------------
        # 3) Otherwise, assume it's a checkpoint (.pt) → export to ONNX
        # -------------------------------------------------------------
        # The ONNX file will be placed next to the .pt file with same stem
        onnx_path = dest_path.with_suffix(".onnx")

        # If ONNX already exists from a previous export, reuse it
        if onnx_path.is_file():
            self.get_logger().info(
                f"Found existing ONNX export for '{file_name}': '{onnx_path}'. Reusing it."
            )
            return onnx_path

        # Download checkpoint (.pt) if not already present
        if not dest_path.is_file():
            self.get_logger().info(
                f"Downloading checkpoint '{file_name}' from W&B to '{dest_path}'..."
            )
            try:
                wandb_file = run.file(file_name)
                wandb_file.download(root=str(cache_dir), replace=True)
            except Exception as e:
                self.get_logger().error(
                    f"Failed to download checkpoint '{file_name}' from W&B: {e}"
                )
                return None
        else:
            self.get_logger().info(
                f"Checkpoint already in cache: '{dest_path}', skipping download."
            )

        # -------------------------------------------------------------
        # 4) Export .pt → .onnx using external script
        # -------------------------------------------------------------
        export_script = pathlib.Path(self.export_from_ckpt_script)

        # If the path does not exist → search automatically
        if not export_script.is_file():
            self.get_logger().warn(
                f"Export script not found at declared path: {export_script}\n"
                "Attempting automatic discovery in home directory..."
            )

            auto = self._auto_find_export_script()
            if auto is None:
                self.get_logger().error(
                    "Automatic search failed: Could not find export_rslrl_to_onnx_from_ckpt.py\n"
                    "Please place it somewhere in your HOME directory."
                )
                return None

            self.get_logger().info(f"Found export script automatically at: {auto}")
            export_script = auto

        self.get_logger().info(
            f"Exporting checkpoint '{dest_path}' to ONNX using script:\n"
            f"  {export_script}\n"
            f"Target ONNX path: {onnx_path}"
        )

        cmd = [
            sys.executable,                # use the same Python interpreter that runs this node
            str(export_script),
            "--checkpoint", str(dest_path),
            "--output", str(onnx_path),
            "--task", self.export_task_name,
            "--num_envs", str(self.export_num_envs),
        ]

        self.get_logger().info(f"Running export command: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.get_logger().info(
                "Export script output (stdout):\n" + result.stdout
            )
            if result.stderr.strip():
                self.get_logger().warn(
                    "Export script warnings/errors (stderr):\n" + result.stderr
                )
        except subprocess.CalledProcessError as e:
            self.get_logger().error(
                "Export script failed with non-zero exit code:\n"
                f"Command: {' '.join(cmd)}\n"
                f"Return code: {e.returncode}\n"
                f"Stdout:\n{e.stdout}\n"
                f"Stderr:\n{e.stderr}"
            )
            return None
        except Exception as e:
            self.get_logger().error(f"Failed to run export script: {e}")
            return None

        # Check that ONNX file was actually created
        if not onnx_path.is_file():
            self.get_logger().error(
                f"Export script completed but ONNX file not found at '{onnx_path}'."
            )
            return None

        self.get_logger().info(f"✅ ONNX policy successfully exported to: {onnx_path}")
        return onnx_path

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

        # ---------------- Build ONNX input feed ----------------
        # Always feed the observation tensor
        inputs_feed = {self.onnx_input_name: obs_batch}

        # If the model expects an additional 'time_step' input,
        # automatically provide a monotonically increasing counter.
        if self.time_step_name is not None:
            self.step_counter += 1
            ts_info = self.onnx_inputs.get(self.time_step_name, None)

            # Default dtype for the time_step: int64
            ts_dtype = np.int64
            if ts_info is not None:
                # Try to infer the dtype from the ONNX type string, e.g. 'tensor(int64)'
                type_str = ts_info.type or ""
                if "float" in type_str:
                    ts_dtype = np.float32
                elif "int32" in type_str:
                    ts_dtype = np.int32
                elif "int64" in type_str:
                    ts_dtype = np.int64

            # Use shape [1, 1] by default; ONNX can usually broadcast
            time_step_arr = np.array([[self.step_counter]], dtype=ts_dtype)
            inputs_feed[self.time_step_name] = time_step_arr

        # ---------------- ONNX inference ----------------
        try:
            outputs = self.session.run(None, inputs_feed)
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
