#!/usr/bin/env python
# environment_setup.py - LMM Development Environment Setup
# Sets up the Python environment for the Large Mind Model project

import os
import sys
import subprocess
import platform
from pathlib import Path

class EnvironmentSetup:
    """
    Sets up the Python environment for the Large Mind Model project
    with CUDA 12.1 support for RTX 3070 GPU acceleration.
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.absolute()
        self.env_name = "lmm_env"
        self.python_version = "3.9"  # Target Python version
        self.cuda_version = "12.1"
        self.gpu_model = "RTX 3070"
        self.is_windows = platform.system() == "Windows"
        
    def check_cuda_availability(self):
        """Check if CUDA is available on the system and report version"""
        print("Checking CUDA availability...")
        
        if self.is_windows:
            # Windows-specific CUDA check
            nvcc_path = os.path.join(os.environ.get('CUDA_PATH', ''), 'bin', 'nvcc.exe')
            if os.path.exists(nvcc_path):
                try:
                    result = subprocess.run([nvcc_path, '--version'], 
                                           capture_output=True, 
                                           text=True, 
                                           check=True)
                    print(f"CUDA is available: {result.stdout.strip()}")
                    return True
                except Exception as e:
                    print(f"Error checking CUDA version: {e}")
            else:
                print(f"CUDA not found at path: {nvcc_path}")
                print("Please install CUDA 12.1 from NVIDIA website")
                return False
        else:
            # This shouldn't be reached on Windows, but keeping it for future compatibility
            print("Non-Windows OS detected, please manually ensure CUDA 12.1 is installed")
            return False
    
    def install_dependencies(self):
        """Install required Python packages with CUDA support"""
        print("\nInstalling required Python packages...")
        
        # Core dependencies with specific versions
        packages = [
            # Core ML frameworks
            "torch==2.1.0+cu121",
            "torchvision==0.16.0+cu121",
            "torchaudio==2.1.0+cu121",
            "tensorflow==2.13.0",  # TF version compatible with CUDA 12.1
            
            # FAISS for embedding search
            "faiss-gpu==1.7.2", 
            
            # Data processing and validation
            "pydantic==2.4.2",
            "numpy==1.24.3",
            "pandas==2.1.0",
            "pillow==10.0.1",
            "scipy==1.11.3",
            
            # Graph neural networks
            "torch-geometric==2.4.0",
            "networkx==3.1",
            
            # Visualization
            "matplotlib==3.7.3",
            "seaborn==0.13.0",
            "dash==2.13.0",
            "plotly==5.16.1",
            
            # Utilities
            "tqdm==4.66.1",
            "psutil==5.9.5",
            "h5py==3.9.0",
            
            # Testing
            "pytest==7.4.2",
        ]
        
        install_commands = []
        
        # PyTorch installation from PyTorch website with CUDA 12.1
        if self.is_windows:
            # Windows-specific installation commands
            install_commands.append([
                "pip", "install", 
                "--index-url", "https://download.pytorch.org/whl/cu121",
                "torch", "torchvision", "torchaudio"
            ])
            
            # Install remaining packages
            for package in packages:
                if not package.startswith("torch"):  # Skip torch packages already installed
                    install_commands.append(["pip", "install", package])
        
        # Execute installation commands
        for cmd in install_commands:
            try:
                print(f"Running: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error installing package: {e}")
                print("Please try installing manually.")
    
    def test_gpu_setup(self):
        """Test if CUDA and GPU are properly configured for PyTorch and TensorFlow"""
        print("\nTesting GPU configuration...")
        
        # Create test script
        test_script = self.project_root / "environment" / "gpu_test.py"
        with open(test_script, "w") as f:
            f.write("""
import os
import sys
import torch
import tensorflow as tf
import numpy as np

def test_pytorch_gpu():
    print("\\nPyTorch GPU Test:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        
        # Simple tensor operation on GPU
        x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        y = x * 2
        print(f"GPU tensor operation result: {y}")
        return True
    else:
        print("PyTorch cannot use GPU!")
        return False

def test_tensorflow_gpu():
    print("\\nTensorFlow GPU Test:")
    print(f"TensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"TensorFlow GPU devices: {gpus}")
    
    if gpus:
        # Simple operation on GPU
        with tf.device('/GPU:0'):
            x = tf.constant([1.0, 2.0, 3.0])
            y = x * 2
            print(f"GPU tensor operation result: {y}")
        return True
    else:
        print("TensorFlow cannot use GPU!")
        return False

def test_faiss_gpu():
    try:
        print("\\nFAISS GPU Test:")
        import faiss
        print(f"FAISS version: {faiss.__version__}")
        
        try:
            # Simple FAISS GPU test
            res = faiss.StandardGpuResources()
            dim = 768  # Typical embedding dimension
            index = faiss.IndexFlatL2(dim)
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            print("FAISS GPU index created successfully")
            return True
        except Exception as e:
            print(f"FAISS GPU test failed: {e}")
            return False
    except ImportError:
        print("FAISS not installed correctly")
        return False

if __name__ == "__main__":
    pytorch_ok = test_pytorch_gpu()
    tensorflow_ok = test_tensorflow_gpu()
    faiss_ok = test_faiss_gpu()
    
    if pytorch_ok and tensorflow_ok and faiss_ok:
        print("\\n✅ GPU setup is complete and working correctly!")
        sys.exit(0)
    else:
        print("\\n❌ GPU setup has issues. Please check the output above.")
        sys.exit(1)
""")
        
        print(f"Created GPU test script at {test_script}")
        print("Run this script after installation to verify GPU setup.")
    
    def create_requirements_file(self):
        """Generate requirements.txt file for the project"""
        requirements_path = self.project_root / "requirements.txt"
        
        with open(requirements_path, "w") as f:
            f.write("""# LMM Project Requirements
# CUDA 12.1 optimized for RTX 3070

# Core ML frameworks
torch==2.1.0+cu121
torchvision==0.16.0+cu121
torchaudio==2.1.0+cu121
tensorflow==2.13.0

# Embedding search
faiss-gpu==1.7.2

# Data validation
pydantic==2.4.2

# Data processing
numpy==1.24.3
pandas==2.1.0
pillow==10.0.1
scipy==1.11.3

# Graph neural networks
torch-geometric==2.4.0
networkx==3.1

# Visualization
matplotlib==3.7.3
seaborn==0.13.0
dash==2.13.0
plotly==5.16.1

# Utilities
tqdm==4.66.1
psutil==5.9.5
h5py==3.9.0

# Testing
pytest==7.4.2
""")
        
        print(f"Created requirements.txt at {requirements_path}")
    
    def create_setup_script(self):
        """Create PowerShell setup script for Windows"""
        setup_path = self.project_root / "environment" / "setup.ps1"
        
        with open(setup_path, "w") as f:
            f.write("""# LMM Project Environment Setup for Windows
# PowerShell script to set up virtual environment with CUDA support

# Check admin status
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Warning "For the best experience, run this script as Administrator"
}

# Create virtual environment
Write-Host "Creating Python virtual environment..." -ForegroundColor Green
python -m venv ..\lmm_env

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
..\lmm_env\Scripts\Activate.ps1

# Install required packages
Write-Host "Installing required packages..." -ForegroundColor Green
python environment_setup.py

# Test GPU setup
Write-Host "Testing GPU setup..." -ForegroundColor Green
python gpu_test.py

Write-Host "Setup complete! Use '..\lmm_env\Scripts\Activate.ps1' to activate the environment for future sessions." -ForegroundColor Green
""")
        
        print(f"Created setup.ps1 at {setup_path}")
    
    def setup(self):
        """Run the full setup process"""
        print(f"Setting up LMM development environment for {self.gpu_model} with CUDA {self.cuda_version}")
        
        # Check CUDA
        cuda_available = self.check_cuda_availability()
        if not cuda_available and self.is_windows:
            print("\nCUDA 12.1 is required for this project.")
            print("Please install CUDA 12.1 from: https://developer.nvidia.com/cuda-12-1-0-download-archive")
            print("After installation, run this script again.")
            return
        
        # Create requirements file
        self.create_requirements_file()
        
        # Generate GPU test script
        self.test_gpu_setup()
        
        # Create setup script for Windows
        if self.is_windows:
            self.create_setup_script()
            print("\nSetup preparation complete!")
            print(f"To complete setup, run: {self.project_root}/environment/setup.ps1 in PowerShell")
        
if __name__ == "__main__":
    setup = EnvironmentSetup()
    setup.setup() 