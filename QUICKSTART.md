# Quick Start Installation

## The Problem
Craftax requires Python 3.9+, but the default install may create a Python 3.8 environment.

## The Solution (Apple Silicon)

**One command to rule them all:**

```bash
./INSTALL.sh
```

That's it! This will:
1. Remove the old `craftax-recording` environment
2. Create a fresh environment with Python 3.9
3. Install JAX and jaxlib correctly for Apple Silicon
4. Install Craftax
5. Install this package

## Manual Installation (if script doesn't work)

```bash
# 1. Remove old environment
conda env remove -n craftax-recording

# 2. Create new environment with Python 3.9
conda create -n craftax-recording python=3.9 -y
conda activate craftax-recording

# 3. Install dependencies
pip install numpy pygame scipy ml-dtypes opt-einsum

# 4. Install JAX (Apple Silicon specific)
pip install jax
pip install jaxlib --find-links https://storage.googleapis.com/jax-releases/jax_releases.html

# 5. Install Craftax
git clone https://github.com/MichaelTMatthews/Craftax.git
cd Craftax
pip install -e .
cd ..

# 6. Install this package
pip install -e .

# 7. Test it
python -m recorder.play --help
```

## For Other Platforms

### Linux (CPU)
```bash
conda create -n craftax-recording python=3.9 -y
conda activate craftax-recording
pip install -r requirements.txt
pip install "jax[cpu]"
git clone https://github.com/MichaelTMatthews/Craftax.git
cd Craftax && pip install -e . && cd ..
pip install -e .
```

### Linux/Windows (CUDA)
```bash
conda create -n craftax-recording python=3.9 -y
conda activate craftax-recording
pip install -r requirements.txt
pip install "jax[cuda12]"  # or jax[cuda11]
git clone https://github.com/MichaelTMatthews/Craftax.git
cd Craftax && pip install -e . && cd ..
pip install -e .
```

## Verifying Your Install

```bash
conda activate craftax-recording
python -c "import jax, jaxlib, craftax; print('✓ All good!')"
python -m recorder.play --help
```
