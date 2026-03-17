# Craftax Trajectories

A Python package for recording human gameplay trajectories in [Craftax](https://github.com/MichaelTMatthews/Craftax), a procedurally-generated Minecraft-inspired gridworld environment. These trajectories can be used for training RL agents, imitation learning, or analyzing human gameplay patterns.

## Features

- **Full trajectory recording**: Captures observations, actions, rewards, and done flags
- **Environment state snapshots**: Saves complete JAX environment states for video rendering
- **Text observations**: Exports human-readable text observations for debugging
- **Metadata tracking**: Records git commit, environment params, and timestamps
- **Auto-archiving**: Creates zip files for easy sharing and storage
- **Flexible recording**: Stop on episode end or continue across multiple episodes

## Installation

### Prerequisites

- Python 3.8+
- JAX (with CUDA support for GPU)
- Craftax environment

### Quick Install

```bash
# Clone the repository
git clone https://github.com/3N3G/craftax-trajectories.git
cd craftax-trajectories

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Installing Craftax

If you don't have Craftax installed:

```bash
git clone https://github.com/MichaelTMatthews/Craftax.git
cd Craftax
pip install -e .
```

## Usage

### Basic Recording

Record a gameplay session and save the trajectory:

```bash
python -m recorder.play
```

This will:
1. Launch a playable Craftax window
2. Record every timestep (observation, action, reward, state)
3. Save the trajectory when you close the window
4. Create a zip archive for easy sharing

### Command Line Options

```bash
python -m recorder.play --help
```

Available options:

- `--god_mode`: Enable god mode (invincibility)
- `--debug`: Disable JIT compilation for debugging
- `--fps <N>`: Set frames per second (default: 60)
- `--continue_after_done`: Keep recording after episode ends
- `--output_dir <DIR>`: Specify output directory (default: `trajectory_recordings`)
- `--no_zip`: Don't create zip archive

### Example Commands

```bash
# Record in god mode at 30 FPS
python -m recorder.play --god_mode --fps 30

# Record multiple episodes in one session
python -m recorder.play --continue_after_done

# Save to custom directory without zip
python -m recorder.play --output_dir my_trajectories --no_zip
```

## Output Format

Each recording session creates a timestamped directory with the following structure:

```
trajectory_recordings/
└── traj_20260317_143022/
    ├── metadata.json              # Session metadata
    ├── obs_vectors.npy            # (T, D) observation vectors
    ├── text_obs.jsonl             # Human-readable observations
    ├── trajectory.npz             # episode_ids, action_ids, rewards, dones
    ├── trajectory_pairs.pbz2      # Paired (obs, action, reward) data
    └── env_states/                # Full environment states for rendering
        ├── t_00000.pbz2
        ├── t_00001.pbz2
        └── ...
```

### Data Files

| File | Description | Format |
|------|-------------|--------|
| `metadata.json` | Session info, env config, timestamps | JSON |
| `obs_vectors.npy` | Symbolic observation vectors | NumPy (T, D) float32 |
| `text_obs.jsonl` | Human-readable text observations | JSONL |
| `trajectory.npz` | Actions, rewards, dones, episode IDs | NumPy compressed |
| `trajectory_pairs.pbz2` | Complete trajectory with paired data | Pickle + bz2 |
| `env_states/` | Full JAX environment states | Pickle + bz2 per timestep |

### Data Alignment

All data is aligned by timestep `t`:
- `obs[t]` = observation at time t
- `action[t]` = action taken at time t
- `reward[t]` = reward received at time t
- `done[t]` = episode termination flag at time t

When recording stops by user quit (not episode end), a final trailing state is written with `action_id=-1` and `reward=NaN`.

## Using Recorded Trajectories

### Loading Trajectories

```python
import numpy as np
import json

# Load observation vectors
obs_vectors = np.load("traj_20260317_143022/obs_vectors.npy")

# Load trajectory data
traj_data = np.load("traj_20260317_143022/trajectory.npz")
actions = traj_data['action_ids']
rewards = traj_data['rewards']
dones = traj_data['dones']

# Load metadata
with open("traj_20260317_143022/metadata.json") as f:
    metadata = json.load(f)

print(f"Recorded {metadata['num_timesteps']} timesteps")
print(f"Observation dimension: {metadata['obs_dim']}")
```

### Loading Environment States

```python
import bz2
import pickle

def load_env_state(path):
    """Load a compressed environment state."""
    with bz2.open(path, 'rb') as f:
        return pickle.load(f)

# Load state at timestep 42
state = load_env_state("traj_20260317_143022/env_states/t_00042.pbz2")
```

### Text Observations

```python
import json

# Read JSONL file
with open("traj_20260317_143022/text_obs.jsonl") as f:
    for line in f:
        entry = json.loads(line)
        print(f"Timestep {entry['t']}: {entry['action_name']}")
        print(entry['raw_text_obs'])
```

## Controls

Default keyboard controls for playing Craftax:

- Arrow keys: Move up/down/left/right
- Space: Do (mine, attack, place, etc.)
- `1-9`: Select inventory items
- Tab: Toggle inventory
- `Q`: Quit and save trajectory

(See `craftax.craftax.play_craftax.KEY_MAPPING` for full controls)

## Use Cases

### Training Imitation Learning Agents

```python
# Use obs_vectors and actions for behavioral cloning
obs = np.load("trajectory/obs_vectors.npy")
actions = np.load("trajectory/trajectory.npz")['action_ids']

# Filter out trailing state (action_id=-1)
valid_mask = actions >= 0
obs_train = obs[valid_mask]
actions_train = actions[valid_mask]

# Train your policy
policy.fit(obs_train, actions_train)
```

### Analyzing Human Strategies

```python
import json

# Analyze action distribution
with open("trajectory/text_obs.jsonl") as f:
    actions = [json.loads(line)['action_name'] for line in f]

from collections import Counter
action_counts = Counter(actions)
print("Most common actions:", action_counts.most_common(10))
```

### Generating Training Videos

Use the saved environment states to render gameplay videos:

```python
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.renderer import render_craftax_pixels

env = make_craftax_env_from_name("Craftax-Symbolic-v1")

# Load and render each state
frames = []
for t in range(num_timesteps):
    state = load_env_state(f"env_states/t_{t:05d}.pbz2")
    frame = render_craftax_pixels(state, env.default_params)
    frames.append(frame)

# Save as video with your preferred library (imageio, cv2, etc.)
```

## Project Structure

```
craftax-trajectories/
├── recorder/
│   ├── __init__.py       # Package exports
│   ├── play.py           # Main recording logic
│   └── utils.py          # Utility functions
├── examples/             # Example trajectories (empty initially)
├── README.md             # This file
├── requirements.txt      # Python dependencies
├── pyproject.toml        # Package metadata
└── LICENSE               # MIT License
```

## Development

### Running Tests

```bash
# Test basic recording
python -m recorder.play --fps 60

# Test god mode
python -m recorder.play --god_mode

# Test multi-episode recording
python -m recorder.play --continue_after_done
```

### Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Metadata Schema

The `metadata.json` file includes:

```json
{
  "schema_version": "1.1.0",
  "session_id": "traj_20260317_143022",
  "created_at": "2026-03-17T14:30:22",
  "saved_at": "2026-03-17T14:35:18",
  "env_name": "Craftax-Symbolic-v1",
  "env_params_digest": "sha1_hash_of_env_params",
  "repo_commit": "abc1234",
  "num_timesteps": 1523,
  "obs_dim": 512,
  "paths": {
    "obs_vectors": "obs_vectors.npy",
    "text_obs": "text_obs.jsonl",
    "trajectory_arrays": "trajectory.npz",
    "trajectory_pairs": "trajectory_pairs.pbz2",
    "env_states_dir": "env_states"
  }
}
```

## Troubleshooting

### JAX/CUDA Issues

If you encounter CUDA errors:

```bash
# Force CPU mode
JAX_PLATFORM_NAME=cpu python -m recorder.play
```

### Pygame Window Not Opening

Ensure you have a display available:

```bash
# On headless servers, use virtual display
xvfb-run python -m recorder.play
```

### Large File Sizes

Environment states can be large (~1-5 MB each). To reduce file size:

- Use `--no_zip` and compress manually
- Record shorter sessions
- Modify `recorder/play.py` to disable `save_env_states`

## Related Projects

- [Craftax](https://github.com/MichaelTMatthews/Craftax) - The main Craftax environment
- [Craftax Paper](https://arxiv.org/abs/2402.16801) - Original research paper

## License

MIT License - see [LICENSE](LICENSE) file for details

## Citation

If you use this tool in your research, please cite the Craftax paper:

```bibtex
@article{matthews2024craftax,
  title={Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning},
  author={Matthews, Michael and Samvelyan, Mikayel and Parker-Holder, Jack and Grefenstette, Edward and Rockt{\"a}schel, Tim},
  journal={arXiv preprint arXiv:2402.16801},
  year={2024}
}
```

## Acknowledgments

Built for the [Craftax Baselines](https://github.com/3N3G/Craftax_Baselines) research project.
