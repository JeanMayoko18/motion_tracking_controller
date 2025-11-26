# motion_tracking_controller

ROS 2 Humble package for deploying reinforcement-learning (RL) policies to the Unitree G1 humanoid robot in **MuJoCo simulation** and on **real hardware**.  
Policies trained in **IsaacLab** (and optionally stored in **Weights & Biases**) can be executed seamlessly using a unified ROS interface.

## Features
- Load policies from **WandB** or **local ONNX**
- Run policies in **LeggedGym/unitree_rl_gym** MuJoCo simulation
- Deploy policies on the **real Unitree G1** through SSH + `g1_loco_client`
- Real-time action publishing (joint torques / commands)
- Consistent sim-to-real pipeline

## Repository Structure
```
motion_tracking_controller/
  package.xml
  setup.py
  CMakeLists.txt
  launch/
    mujoco.launch.py
    real.launch.py
  motion_tracking_controller/
    policy_node.py
```

## Usage (Simulation)
```bash
ros2 launch motion_tracking_controller mujoco.launch.py wandb_path:=ENTITY/PROJECT/RUN
```
or
```bash
ros2 launch motion_tracking_controller mujoco.launch.py policy_path:=path/to/policy.onnx
```

## Usage (Real Robot)
```bash
ros2 launch motion_tracking_controller real.launch.py robot_ip:=ROBOT_IP wandb_path:=ENTITY/PROJECT/RUN
```
or
```bash
ros2 launch motion_tracking_controller real.launch.py robot_ip:=ROBOT_IP policy_path:=path/to/policy.onnx
```

## Requirements
- ROS 2 Humble
- Python 3.10+
- MuJoCo + LeggedGym (`unitree_rl_gym`) for simulation
- ONNX Runtime and WandB (if using WandB policies)

## License
MIT License
