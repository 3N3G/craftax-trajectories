# Example Trajectories

This directory will contain example recorded trajectories for demonstration and testing.

## Example Usage

After installing the package, run:

```bash
# Record a short gameplay session
python -m recorder.play --fps 60

# The output will be saved in trajectory_recordings/
```

## Loading Example Data

```python
import numpy as np
import json
from pathlib import Path

# Find the most recent trajectory
traj_dir = Path("trajectory_recordings")
latest_traj = max(traj_dir.glob("traj_*"), key=lambda p: p.stat().st_mtime)

# Load the data
obs = np.load(latest_traj / "obs_vectors.npy")
traj = np.load(latest_traj / "trajectory.npz")
with open(latest_traj / "metadata.json") as f:
    metadata = json.load(f)

print(f"Loaded trajectory with {len(obs)} timesteps")
print(f"Observation shape: {obs.shape}")
print(f"Actions: {traj['action_ids']}")
```

## Example Trajectories

Once you record some trajectories, you can place them here as examples for:
- Testing the loading code
- Demonstrating different gameplay strategies
- Benchmarking imitation learning algorithms
