# LMM Environment Setup

This directory contains the setup scripts for configuring the development environment for the Large Mind Model (LMM) project.

## System Requirements

- Windows 10 or newer
- NVIDIA RTX 3070 GPU or compatible
- CUDA 12.1 installed
- Python 3.9 or newer

## Installation Instructions

### 1. Install CUDA 12.1

If you don't have CUDA 12.1 installed:
1. Download CUDA 12.1 from the [NVIDIA CUDA Archive](https://developer.nvidia.com/cuda-12-1-0-download-archive)
2. Follow the installation instructions for your system
3. Verify installation by running `nvcc --version` in a terminal

### 2. Run Setup Script

From this directory, run the PowerShell setup script:

```powershell
.\setup.ps1
```

This script will:
- Create a Python virtual environment
- Install required packages with CUDA support
- Test the GPU setup to verify everything is working correctly

### 3. Activate the Environment

After installation, activate the environment:

```powershell
..\lmm_env\Scripts\Activate.ps1
```

## Verifying the Installation

The setup process includes a GPU test that verifies:
- PyTorch with CUDA support
- TensorFlow with GPU acceleration
- FAISS with GPU support

If all tests pass, you'll see the message: "✅ GPU setup is complete and working correctly!"

## Manual Installation

If the automatic setup fails, you can manually install components:

1. Create a virtual environment:
   ```powershell
   python -m venv ..\lmm_env
   ..\lmm_env\Scripts\Activate.ps1
   ```

2. Install PyTorch with CUDA support:
   ```powershell
   pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
   ```

3. Install remaining packages:
   ```powershell
   pip install -r ..\requirements.txt
   ```

4. Test GPU setup:
   ```powershell
   python gpu_test.py
   ```

## Troubleshooting

Common issues and solutions:

- **CUDA not found**: Ensure CUDA 12.1 is properly installed and CUDA_PATH environment variable is set
- **PyTorch not finding CUDA**: Make sure you installed the CUDA-enabled version of PyTorch
- **TensorFlow GPU issues**: Check that you have compatible CUDA and cuDNN versions for TensorFlow 2.13.0

For more detailed troubleshooting, refer to the [PyTorch](https://pytorch.org/get-started/locally/) and [TensorFlow](https://www.tensorflow.org/install/gpu) GPU setup guides. 