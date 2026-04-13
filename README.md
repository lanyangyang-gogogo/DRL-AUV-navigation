# DRL-AUV-Navigation: Deep Reinforcement Learning for AUV Navigation

基于 ROS Gazebo 仿真平台的自主水下机器人（AUV）无地图避障导航方案。

本项目利用**双延迟深度确定性策略梯度（TD3）**神经网络，使 AUV 能够在完全未知的模拟水下环境（二维平面方向）中学习避开障碍物，并自主导航至随机生成的目标点。模型基于 PyTorch 框架开发，并在 Ubuntu 20.04 + ROS Noetic 环境下测试通过。

![AUV Navigation Demo](https://placehold.co/800x400/eeeeee/4a90e2?text=Replace+with+your+AUV+Navigation+GIF)

---

## 🌟 项目亮点

- **架构解耦 (Decoupled Design)**：仿真环境 (Gazebo) 与算法逻辑 (PyTorch) 完全物理分离，通过 ROS Topic 机制通信。彻底规避了 Conda 虚拟环境与系统 ROS 底层 C++ 库冲突导致的段错误（Segmentation Fault）。
- **连续动作空间控制**：基于 TD3 算法输出连续的线速度与角速度指令，实现 20Hz 的高频平滑控制。
- **无头训练模式 (Headless Training)**：支持关闭 Gazebo 图形渲染界面进行物理推演，大幅降低显存和 CPU 占用，成倍提升强化学习采样效率。

---

## 📁 目录结构

本项目将 ROS 仿真空间与算法代码进行了清晰的分离：

```text
DRL-AUV-Navigation/
│
├── simulation/                 # ROS 仿真端 (ROS Workspace 源码)
│   ├── uuv_simulator/          # AUV 模型、水动力学插件及传感器定义
│   └── custom_msgs/            # 自定义 ROS 通信消息
│
├── algorithm/                  # 深度强化学习算法端 (PyTorch)
│   ├── TD3/
│   │   ├── train_velodyne_td3.py  # TD3 算法训练主程序
│   │   ├── test_velodyne_td3.py   # 模型评估与测试程序
│   │   ├── velodyne_env.py        # ROS-Gym 环境交互接口封装
│   │   └── assets/                # Launch 启动配置文件
│   ├── pytorch_models/         # 存放预训练好的 Actor/Critic 模型权重 (.pth)
│   └── results/                # 存放评估数据与 Reward 记录 (.npy)
│
└── README.md                   # 本说明文档
```

---

## 🛠️ 快速复现指南

### 1. 环境依赖 (Prerequisites)
- OS: Ubuntu 20.04
- ROS: Noetic
- Python: 3.8+
- ML Framework: PyTorch

### 2. 仿真环境准备 (Terminal 1)
首先，配置并编译 ROS 仿真工作空间：

```bash
# 创建并进入工作空间
mkdir -p ~/uuv_ws/src
# 将本项目的 simulation 内容拷贝至 src 目录下
cp -r simulation/* ~/uuv_ws/src/
cd ~/uuv_ws

# 编译工作空间
catkin_make
source devel/setup.bash

# 启动 AUV 仿真 (推荐追加 gui:=false 以无头模式启动，极大提升训练速度)
roslaunch uuv_gazebo start_pid_demo_with_teleop.launch gui:=false
```

### 3. 算法环境准备与运行 (Terminal 2)
建议使用 Anaconda 创建独立的 Python 虚拟环境，以避免与 ROS 默认环境冲突：

```bash
# 创建并激活 conda 环境
conda create -n auv_nav python=3.8
conda activate auv_nav

# 安装算法所需依赖
pip install torch numpy tensorboard squaternion

# 进入算法目录并启动训练
cd algorithm/TD3
python3 train_velodyne_td3.py
```

如果你想测试预训练好的模型，请运行：
```bash
python3 test_velodyne_td3.py
```

---

## 📊 训练结果与监控

在训练过程中，你可以使用 TensorBoard 实时查看 Loss 和 Average Reward 的收敛曲线：

```bash
cd algorithm/TD3
tensorboard --logdir=runs/
```

打开浏览器访问 `http://localhost:6006` 即可查看。

---

## 📜 致谢与引用 (Acknowledgments & References)

本项目的顺利完成离不开以下优秀开源项目的支持与启发，特此致谢：

- **reiniscimurs / DRL-robot-navigation**  
  本项目核心的深度强化学习框架（TD3 算法实现）以及 ROS-Gym 风格的环境交互逻辑（velodyne_env.py），主要参考并适配自该项目针对陆地移动机器人的导航研究。

- **uuvsimulator / uuv_simulator**  
  本项目中水下机器人（AUV）的高保真物理模型、水动力学插件（Hydrodynamics plugins）以及底层推进器控制逻辑，均依赖于该开源水下仿真组件。

---

## 📄 协议 (License)

本项目基于 MIT License 开源。
