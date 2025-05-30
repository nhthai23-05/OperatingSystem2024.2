# 🚀 Operating Systems 2024.2 - Installation Guide

## Quick Setup

### 1. Create Virtual Environment (if not already done)
```powershell
cd "d:\Projects\Hust\Operating system 2024.2"
python -m venv setup
```

### 2. Activate Virtual Environment
```powershell
.\setup\Scripts\Activate.ps1
```

### 3. Install All Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Verify Installation
```powershell
python -c "import stable_baselines3, gym, torch, pandas, psutil; print('✅ All dependencies installed successfully!')"
```

## Step-by-Step Installation (Alternative)

If you encounter issues with the full requirements.txt, install dependencies by step:

### Step 1 Dependencies (Data Collection)
```powershell
pip install psutil pandas numpy
```

### Step 2 Dependencies (Machine Learning)
```powershell
pip install scikit-learn xgboost imbalanced-learn onnx onnxruntime
```

### Step 3 Dependencies (RL Environment)
```powershell
pip install gym==0.26.2
```

### Step 4 Dependencies (RL Training)
```powershell
pip install stable-baselines3==1.8.0 torch tensorboard matplotlib
```

## System Requirements

- **Python**: 3.8+ (recommended: 3.9)
- **OS**: Windows 10/11, macOS, or Linux
- **RAM**: Minimum 8GB (recommended: 16GB+)
- **Storage**: ~2GB for dependencies

## Troubleshooting

### Common Issues:

1. **PyTorch Installation**:
   ```powershell
   # For CPU-only (lighter)
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   
   # For CUDA GPU support (if you have NVIDIA GPU)
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

2. **Gym Version Conflicts**:
   ```powershell
   pip uninstall gym gymnasium
   pip install gym==0.26.2
   ```

3. **Stable-Baselines3 Compatibility**:
   ```powershell
   pip install stable-baselines3==1.8.0
   ```

## Project Structure After Installation

```
Operating system 2024.2/
├── requirements.txt          # Main dependencies file
├── SETUP.md                 # This installation guide
├── setup/                   # Virtual environment
├── Step1/                   # Data collection
├── Step2/                   # Baseline ML model
├── Step3/                   # RL environment
└── Step4/                   # RL agent training
```

## Next Steps

1. **Test Step 3 Environment**:
   ```powershell
   cd Step4
   python test.py
   ```

2. **Run RL Training**:
   ```powershell
   python train_rl_agent.py
   ```

3. **Evaluate Results**:
   ```powershell
   python evaluate_model.py
   ```

## Additional Tools (Optional)

### For Step 1 Data Collection:
- **stress-ng** (system stress testing)
  - Ubuntu: `sudo apt install stress-ng`
  - macOS: `brew install stress-ng`
  - Windows: Use alternative stress tools

### For Enhanced Development:
```powershell
pip install jupyter tqdm plotly seaborn
```

---

## 🎯 Project Goals Reminder

This project creates an AI-powered CPU scheduler using:
1. **Real system data collection** (Step 1)
2. **Baseline ML model** (Step 2) 
3. **RL environment simulation** (Step 3)
4. **AI agent training** (Step 4)

**Current Status**: ✅ All steps completed successfully!
**Your Results**: 271.9% better than random scheduling!
