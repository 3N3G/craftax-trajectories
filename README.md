# Craftax Trajectories

Record human gameplay trajectories in [Craftax](https://github.com/MichaelTMatthews/Craftax) for training RL agents or analyzing gameplay patterns.

## Installation

### Apple Silicon (M1/M2/M3)

```bash
git clone https://github.com/3N3G/craftax-trajectories.git
cd craftax-trajectories
./INSTALL.sh
```

### Linux/Windows

```bash
git clone https://github.com/3N3G/craftax-trajectories.git
cd craftax-trajectories

# Create environment
conda create -n craftax-recording python=3.9 -y
conda activate craftax-recording

# Install dependencies
pip install -r requirements.txt

# Install JAX (choose one):
pip install "jax[cpu]"              # CPU only
pip install "jax[cuda12]"           # CUDA 12
pip install "jax[cuda11]"           # CUDA 11

# Install Craftax
git clone https://github.com/MichaelTMatthews/Craftax.git
cd Craftax && pip install -e . && cd ..

# Install this package
pip install -e .
```

## Usage

### Record a trajectory

```bash
conda activate craftax-recording
python -m recorder.play
```

**Controls:**
- Arrow keys: Move
- Space: Do (mine, attack, place)
- 1-9: Select inventory items
- Q: Quit and save

### Visualize a trajectory

```bash
python visualize_trajectory.py
```

Creates an interactive HTML file with:
- Visual game states (rendered images)
- Text observations (map, inventory, stats)
- Timestep slider and playback controls
- Keyboard shortcuts (arrows, spacebar)

### Command-line options

```bash
python -m recorder.play --help

# Examples:
python -m recorder.play --god_mode              # Invincibility
python -m recorder.play --fps 30                # Lower framerate
python -m recorder.play --continue_after_done   # Multi-episode recording
```

## Output Files

Each recording creates a timestamped directory:

```
trajectory_recordings/traj_YYYYMMDD_HHMMSS/
├── metadata.json           # Session info
├── obs_vectors.npy         # (T, D) observation vectors
├── text_obs.jsonl          # Human-readable observations
├── trajectory.npz          # Actions, rewards, dones
├── trajectory_pairs.pbz2   # Complete trajectory data
├── env_states/             # Full environment states
└── visualization.html      # Interactive visualization
```

## Using Trajectories

### Load data

```python
import numpy as np

# Load observations and actions
obs = np.load("traj_XXX/obs_vectors.npy")
traj = np.load("traj_XXX/trajectory.npz")
actions = traj['action_ids']
rewards = traj['rewards']
```

### Train an agent

```python
# Filter trailing state (action_id=-1)
valid = actions >= 0
obs_train = obs[valid]
actions_train = actions[valid]

# Behavioral cloning
policy.fit(obs_train, actions_train)
```

## Troubleshooting

**"No module named 'jaxlib'"** (Apple Silicon)
```bash
pip install jax
pip install jaxlib --find-links https://storage.googleapis.com/jax-releases/jax_releases.html
```

**"No module named 'craftax'"**
```bash
git clone https://github.com/MichaelTMatthews/Craftax.git
cd Craftax && pip install -e . && cd ..
```

**"Package 'craftax' requires a different Python"**
```bash
# Need Python 3.9+
conda create -n craftax-recording python=3.9 -y
```

See [QUICKSTART.md](QUICKSTART.md) for detailed installation instructions.

## Features

- ✓ Full trajectory recording (observations, actions, rewards)
- ✓ Environment state snapshots for video rendering
- ✓ Text observations for debugging
- ✓ Metadata tracking (git commit, timestamps)
- ✓ Interactive HTML visualization
- ✓ Auto-archiving to zip files

## License

MIT License - see [LICENSE](LICENSE) file

## Citation

If you use this tool, please cite the Craftax paper:

```bibtex
@article{matthews2024craftax,
  title={Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning},
  author={Matthews, Michael and Samvelyan, Mikayel and Parker-Holder, Jack and Grefenstette, Edward and Rockt{\"a}schel, Tim},
  journal={arXiv preprint arXiv:2402.16801},
  year={2024}
}
```
