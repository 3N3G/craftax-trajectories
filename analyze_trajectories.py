#!/usr/bin/env python3
"""
Analyze trajectory recordings to determine which ones are successful.
A trajectory is successful if the player descends to floor 2.
A trajectory fails if the player dies (done=True) before reaching floor 2.
"""

import json
from pathlib import Path

def analyze_trajectory(traj_dir):
    """Analyze a single trajectory to determine success/failure."""
    traj_path = Path(traj_dir)
    text_obs_path = traj_path / "text_obs.jsonl"

    if not text_obs_path.exists():
        return None, "Missing text_obs.jsonl"

    # Read the last observation to check final state
    last_obs = None
    with open(text_obs_path) as f:
        for line in f:
            last_obs = json.loads(line)

    if not last_obs:
        return None, "Empty text_obs.jsonl"

    # Parse the raw text observation to get floor level
    raw_text = last_obs.get("raw_text_obs", "")
    floor_level = None
    for line in raw_text.split("\n"):
        if line.startswith("Floor: "):
            floor_level = int(line.split(": ")[1])
            break

    # Check done flag
    done = last_obs.get("done")

    # Determine outcome
    if floor_level == 2:
        return "SUCCESS", f"Reached floor 2 (timestep {last_obs['t']})"
    elif done:
        return "DIED", f"Died on floor {floor_level} (timestep {last_obs['t']})"
    else:
        return "INCOMPLETE", f"Stopped on floor {floor_level} (timestep {last_obs['t']})"

def main():
    traj_recordings = Path("trajectory_recordings")

    # Get all trajectory directories
    trajectories = sorted([p for p in traj_recordings.glob("traj_*") if p.is_dir()])

    print(f"\nAnalyzing {len(trajectories)} trajectories...\n")
    print("=" * 80)

    successes = []
    failures = []
    incomplete = []
    errors = []

    for traj_dir in trajectories:
        traj_name = traj_dir.name
        outcome, details = analyze_trajectory(traj_dir)

        if outcome == "SUCCESS":
            successes.append((traj_name, details))
            status = "\033[92m✓ SUCCESS\033[0m"  # Green
        elif outcome == "DIED":
            failures.append((traj_name, details))
            status = "\033[91m✗ DIED\033[0m"  # Red
        elif outcome == "INCOMPLETE":
            incomplete.append((traj_name, details))
            status = "\033[93m○ INCOMPLETE\033[0m"  # Yellow
        else:
            errors.append((traj_name, details))
            status = "\033[90m? ERROR\033[0m"  # Gray

        print(f"{status:30s} {traj_name:30s} {details}")

    print("=" * 80)
    print(f"\nSummary:")
    print(f"  Successes:   {len(successes)}")
    print(f"  Died:        {len(failures)}")
    print(f"  Incomplete:  {len(incomplete)}")
    print(f"  Errors:      {len(errors)}")

    # Save successes to file
    if successes:
        with open("successes.txt", "w") as f:
            for traj_name, details in successes:
                f.write(f"{traj_name}\n")
        print(f"\n✓ Successful trajectories saved to: successes.txt")

if __name__ == "__main__":
    main()
