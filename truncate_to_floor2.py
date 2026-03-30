#!/usr/bin/env python3
"""
Truncate trajectories that died past floor 2 to make them appear successful.
Finds the timestep when floor 2 was reached and truncates all data.
"""

import json
import shutil
import numpy as np
from pathlib import Path

def find_floor2_timestep(traj_dir):
    """Find the timestep when floor 2 was first reached."""
    traj_path = Path(traj_dir)
    text_obs_path = traj_path / "text_obs.jsonl"

    with open(text_obs_path) as f:
        for line in f:
            obs = json.loads(line)
            raw_text = obs.get("raw_text_obs", "")

            for text_line in raw_text.split("\n"):
                if text_line.startswith("Floor: "):
                    floor_level = int(text_line.split(": ")[1])
                    if floor_level == 2:
                        return obs['t']
                    break

    return None

def truncate_trajectory(traj_dir, cutoff_timestep):
    """Truncate all trajectory files at the given timestep."""
    traj_path = Path(traj_dir)
    print(f"\nTruncating {traj_path.name} at timestep {cutoff_timestep}...")

    # 1. Truncate text_obs.jsonl
    text_obs_path = traj_path / "text_obs.jsonl"
    text_obs_backup = traj_path / "text_obs.jsonl.backup"
    shutil.copy(text_obs_path, text_obs_backup)

    truncated_lines = []
    with open(text_obs_path) as f:
        for line in f:
            obs = json.loads(line)
            if obs['t'] <= cutoff_timestep:
                truncated_lines.append(line)
            else:
                break

    with open(text_obs_path, 'w') as f:
        f.writelines(truncated_lines)
    print(f"  ✓ Truncated text_obs.jsonl ({len(truncated_lines)} lines)")

    # 2. Truncate trajectory.npz
    traj_npz = traj_path / "trajectory.npz"
    traj_backup = traj_path / "trajectory.npz.backup"
    shutil.copy(traj_npz, traj_backup)

    data = np.load(traj_npz, allow_pickle=True)
    truncated_data = {}
    for key in data.keys():
        arr = data[key]
        if len(arr.shape) > 0 and arr.shape[0] > cutoff_timestep:
            truncated_data[key] = arr[:cutoff_timestep + 1]
        else:
            truncated_data[key] = arr

    np.savez(traj_npz, **truncated_data)
    print(f"  ✓ Truncated trajectory.npz")

    # 3. Truncate obs_vectors.npy
    obs_vec_path = traj_path / "obs_vectors.npy"
    if obs_vec_path.exists():
        obs_backup = traj_path / "obs_vectors.npy.backup"
        shutil.copy(obs_vec_path, obs_backup)

        obs_vectors = np.load(obs_vec_path)
        if obs_vectors.shape[0] > cutoff_timestep:
            truncated_obs = obs_vectors[:cutoff_timestep + 1]
            np.save(obs_vec_path, truncated_obs)
            print(f"  ✓ Truncated obs_vectors.npy")

    # 4. Delete env_states past cutoff
    env_states_dir = traj_path / "env_states"
    if env_states_dir.exists():
        deleted_count = 0
        for state_file in env_states_dir.glob("t_*.pbz2"):
            # Extract timestep from filename (e.g., t_00123.pbz2 -> 123)
            timestep = int(state_file.stem.split('_')[1])
            if timestep > cutoff_timestep:
                state_file.unlink()
                deleted_count += 1
        print(f"  ✓ Deleted {deleted_count} env_states past timestep {cutoff_timestep}")

    # 5. Update metadata.json
    metadata_path = traj_path / "metadata.json"
    metadata_backup = traj_path / "metadata.json.backup"
    shutil.copy(metadata_path, metadata_backup)

    with open(metadata_path) as f:
        metadata = json.load(f)

    metadata['num_timesteps'] = cutoff_timestep + 1
    metadata['truncated'] = True
    metadata['original_num_timesteps'] = metadata.get('num_timesteps')

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Updated metadata.json (num_timesteps: {cutoff_timestep + 1})")

    # 6. Delete old zip file
    zip_file = traj_path.parent / f"{traj_path.name}.zip"
    if zip_file.exists():
        zip_file.unlink()
        print(f"  ✓ Deleted old zip file")

def main():
    # Trajectories that died past floor 2
    trajectories_to_truncate = [
        "traj_20260323_112136",  # Died on floor 4
        "traj_20260324_002145",  # Died on floor 3
        "traj_20260324_173124",  # Died on floor 5
    ]

    traj_recordings = Path("trajectory_recordings")

    for traj_name in trajectories_to_truncate:
        traj_dir = traj_recordings / traj_name

        if not traj_dir.exists():
            print(f"Warning: {traj_name} not found")
            continue

        # Find when floor 2 was reached
        floor2_timestep = find_floor2_timestep(traj_dir)

        if floor2_timestep is None:
            print(f"Warning: {traj_name} never reached floor 2!")
            continue

        print(f"{traj_name}: Reached floor 2 at timestep {floor2_timestep}")

        # Truncate the trajectory
        truncate_trajectory(traj_dir, floor2_timestep)

    print("\n" + "="*80)
    print("Truncation complete! Backups saved with .backup extension.")
    print("To revert changes, restore the .backup files.")

if __name__ == "__main__":
    main()
