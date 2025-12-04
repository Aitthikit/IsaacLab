# Isaac Lab Installation Guide (DPPO version)

This guide will walk you through the installation process of Isaac Lab version 4.5.0 with DPPO algorithm using pip installation method.

## Prerequisites

- Python 3.10
- CUDA-compatible GPU
- On Windows: GPU driver version 552.86 (for CUDA 12)
- [Miniconda](https://docs.anaconda.com/miniconda/miniconda-other-installer-links/) (recommended if using Conda)

## Installation Steps

### 1. Create and Activate Virtual Environment

```bash
conda create -n env_isaaclab python=3.10
conda activate env_isaaclab
```

### 2. Install PyTorch

Install CUDA-enabled PyTorch 2.5.1 (required for Windows, optional for Linux):

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Update pip

```bash
pip install --upgrade pip
```

### 4. Install Isaac Sim

Install Isaac Sim packages:

```bash
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
```

### 5. Install Isaac Lab (DPPO version)

Recommend Create workspace before Install
```bash
mkdir workspace_name
cd workspace_name
```

1. Clone the Isaac Lab repository:
```bash
git clone https://github.com/Aitthikit/IsaacLab.git
```

2. Install dependencies (Ubuntu):
```bash
sudo apt install cmake build-essential
```

3. Install Isaac Lab extensions:
```bash
cd IsaacLab
./isaaclab.sh --install  # Installs all learning frameworks
```

### 6. Rsl_rl Install 

Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

1. Clone the Rsl_rl repository:
```bash
cd .. # back to workspace
git clone https://github.com/Aitthikit/rsl_rl.git -b my-fix
```

2. Install dependencies (Ubuntu):
```bash
cd rsl_rl
pip install -e .
```


## Special Notes

### For 50 Series GPUs
If you're using 50 series GPUs, use the latest PyTorch nightly build instead:
```bash
pip install --upgrade --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

### First Run Information
- The first run will download all dependent extensions (may take ~10 minutes)
- You'll need to accept the NVIDIA Software License Agreement
- Extensions will be cached for subsequent runs

## Verification

To verify your installation:

1. Run the simulator:
```bash
isaacsim
```

2. Test with a sample script:
```bash
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
```

3. Verify environment :
```bash
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
```

You should see a simulator window with a black viewport. If this appears, your installation was successful!

## Training Examples 

Once installed, you can try these example training commands:

Train an anymal c to walk with DPPO algorithm:
```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Rough-DPPO-Anymal-C-Direct-v0 --num_envs 1024
```

Train an anymal c to distillation Teacher policy:
```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Distill-DPPO-Anymal-C-Direct-v0 --num_envs 256
```

## Troubleshooting

If you encounter any issues:
- Check the [Isaac Sim documentation](https://docs.omniverse.nvidia.com/dev-guide/latest/linux-troubleshooting.html)
- Visit the [Isaac Sim Forums](https://docs.isaacsim.omniverse.nvidia.com//latest/isaac_sim_forums.html)

For more information and detailed guides, refer to the [official Isaac Lab documentation](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/setup/installation/pip_installation.html)

For more information and detailed guides about algorithm, refer to the [RSL_RL Github](https://github.com/Aitthikit/rsl_rl.git)
