# Tavatar

This repository provides the inference code and pre-trained checkpoint for Tavatar.

## Requirements

- Python >= 3.9
- CUDA 11.8
- [uv](https://github.com/astral-sh/uv) package manager

### Platform-Specific Requirements

**Windows:**
- Visual Studio 2022 with C++ build tools (required for compiling extensions)

**Ubuntu:**
- GCC compiler (usually pre-installed)

## Installation

It's so easy !! install dependencies:

```bash
uv sync
```

This will automatically:
- Create a virtual environment
- Install all required dependencies including PyTorch, PyTorch3D, and other packages
- Build necessary extensions (diff-gaussian-rasterization, tinycudann, etc.)

**Note:** The first installation may take some time as it needs to compile several extensions from source.

## Data Preparation

### SMPL Models

Download the SMPL model files and place them in the following directory structure:

```
./dataset/smpl_models/smpl/
  ├── SMPL_FEMALE.pkl
  ├── SMPL_MALE.pkl
  └── SMPL_NEUTRAL.pkl
```

You can obtain SMPL models from the [official SMPL website](https://smpl.is.tue.mpg.de/) (registration required).

## Inference

We provide a pre-trained checkpoint in `output/tavatar/people_snapshot/male-3-casual/`.

### Run Animation

**Ubuntu/Linux:**
```bash
bash scripts/eval.sh
```

**Windows (PowerShell):**
```powershell
.\scripts\eval.ps1
```

Or run directly with Python:
```bash
python main.py --resume_dir "./output/tavatar/people_snapshot/male-3-casual" --animate
```

### Output

The animation results will be saved in:
```
output/tavatar/people_snapshot/male-3-casual/predict_20/
```

## Troubleshooting

### Windows Build Issues

If you encounter build errors on Windows, ensure:
1. Visual Studio 2022 is installed with "Desktop development with C++" workload
2. CUDA 11.8 is properly installed and in PATH