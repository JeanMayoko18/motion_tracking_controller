# motion_tracking_controller

ROS 2 Humble package for running reinforcement learning policies for the Unitree G1 humanoid in:

- **MuJoCo simulation** (LeggedGym/unitree_rl_gym)
- **Real robot execution** (via SSH + g1_loco_client)

Policies can be loaded from:

- **Weights & Biases** (`wandb_path`)
- **Local ONNX checkpoint** (`policy_path`)

Exactly **one** must be specified.

---

## 1. Package Structure

```
motion_tracking_controller/
  package.xml
  setup.py
  setup.cfg
  CMakeLists.txt
  resource/
    motion_tracking_controller
  motion_tracking_controller/
    __init__.py
    policy_node.py
  launch/
    mujoco.launch.py
    real.launch.py
```

`policy_node.py`:
- Subscribes to joint state information (real or simulated)
- Loads a policy from WandB or ONNX
- Runs ONNX inference
- Publishes normalized RL actions to a torque topic

---

## 2. Requirements

### 2.1 MuJoCo + LeggedGym environment (simulation only)

Activate your MuJoCo RL virtual environment:

```
source ~/mujoco-rl/bin/activate
export PYTHONPATH=$PYTHONPATH:$HOME/RL/LeggGym/unitree_rl_gym
```

### 2.2 ROS 2 workspace

```
source /opt/ros/humble/setup.bash
cd ~/ros2rob_ws
colcon build --packages-select motion_tracking_controller
source install/setup.bash
```

### 2.3 WandB (if using wandb_path)

```
pip install wandb onnxruntime
export WANDB_API_KEY=YOUR_KEY
```

---

## 3. MuJoCo Simulation

### Run with a WandB policy

```
ros2 launch motion_tracking_controller mujoco.launch.py   wandb_path:=your-entity/your-project/your-runid
```

### Run with a local ONNX policy

```
ros2 launch motion_tracking_controller mujoco.launch.py   policy_path:=$HOME/RL/policies/policy.onnx
```

This will:

1. Start `deploy_mujoco_ros.py` using your MuJoCo + LeggedGym installation  
2. Launch `policy_node` that computes actions  
3. Actions are sent through `/mujoco/joint_torque_cmd`

---

## 4. Real Robot Execution (Unitree G1)

You must be able to SSH into the robot:

```
ssh unitree@ROBOT_IP
```

### Run with WandB policy

```
ros2 launch motion_tracking_controller real.launch.py   robot_ip:=192.168.123.10   network_interface:=eth0   loco_mode:=zero_torque   wandb_path:=your-entity/your-project/your-runid
```

### Run with local ONNX policy

```
ros2 launch motion_tracking_controller real.launch.py   robot_ip:=192.168.123.10   network_interface:=eth0   loco_mode:=zero_torque   policy_path:=$HOME/RL/policies/policy.onnx
```

During real execution, the launch file:

1. Opens an SSH session to the robot  
2. Runs `g1_loco_client` in the mode you specify (`zero_torque`, `debug`, etc.)  
3. Runs the ONNX policy on your PC  
4. Sends joint commands to the robot in real time  

---

## 5. Notes

- Ensure that MuJoCo viewer works on your machine (NVIDIA drivers recommended)
- The policy expects the same observation structure as during training  
- If needed, adjust `policy_node.py` to match your exact observation pipeline  
- Make sure SSH keys are configured to avoid password prompts  

---

## 6. License

MIT License
