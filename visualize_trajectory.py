#!/usr/bin/env python3
"""
Generate an interactive HTML visualization of a Craftax trajectory recording.
"""

import os
import sys
import json
import bz2
import pickle
import base64
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


def load_env_state(path):
    """Load a compressed environment state."""
    with bz2.open(path, 'rb') as f:
        return pickle.load(f)


def render_state_to_png(state):
    """Render a Craftax state to PNG image."""
    from craftax.craftax.renderer import render_craftax_pixels
    import jax.numpy as jnp

    # Render to JAX array
    pixels = render_craftax_pixels(state, block_pixel_size=16)

    # Convert JAX array to numpy, then to PIL Image
    pixels_np = np.array(pixels).astype(np.uint8)
    img = Image.fromarray(pixels_np)
    return img


def image_to_base64(img):
    """Convert PIL Image to base64 data URL."""
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def generate_html_visualization(traj_dir):
    """Generate interactive HTML visualization for a trajectory."""
    traj_path = Path(traj_dir)

    print(f"Loading trajectory from: {traj_path}")

    # Load metadata
    with open(traj_path / "metadata.json") as f:
        metadata = json.load(f)

    num_timesteps = metadata['num_timesteps']
    print(f"Number of timesteps: {num_timesteps}")

    # Load text observations
    print("Loading text observations...")
    text_obs = []
    with open(traj_path / "text_obs.jsonl") as f:
        for line in f:
            text_obs.append(json.loads(line))

    # Load trajectory data
    print("Loading trajectory data...")
    traj_data = np.load(traj_path / "trajectory.npz")
    actions = traj_data['action_ids']
    rewards = traj_data['rewards']
    dones = traj_data['dones']

    # Initialize Craftax environment for rendering
    print("Initializing Craftax environment...")
    from craftax.craftax_env import make_craftax_env_from_name
    env = make_craftax_env_from_name(metadata['env_name'], auto_reset=False)
    params = env.default_params

    # Render all states
    print("Rendering environment states to images...")
    images_b64 = []
    env_states_dir = traj_path / "env_states"

    for t in range(num_timesteps):
        if t % 10 == 0:
            print(f"  Rendering timestep {t}/{num_timesteps}...")

        state_path = env_states_dir / f"t_{t:05d}.pbz2"
        if not state_path.exists():
            print(f"Warning: Missing state file {state_path}")
            images_b64.append("")
            continue

        state = load_env_state(state_path)
        img = render_state_to_png(state)
        img_b64 = image_to_base64(img)
        images_b64.append(img_b64)

    print("Generating HTML...")

    # Create JavaScript data structure
    js_data = {
        "metadata": metadata,
        "timesteps": []
    }

    for t in range(num_timesteps):
        timestep_data = {
            "t": t,
            "image": images_b64[t],
            "action": text_obs[t]["action_name"] if t < len(text_obs) else "N/A",
            "reward": float(rewards[t]) if not np.isnan(rewards[t]) else 0,
            "done": bool(dones[t]),
            "text_obs": text_obs[t]["raw_text_obs"] if t < len(text_obs) else ""
        }
        js_data["timesteps"].append(timestep_data)

    # Generate HTML
    html = generate_html_template(js_data)

    # Save HTML file
    output_path = traj_path / "visualization.html"
    with open(output_path, 'w') as f:
        f.write(html)

    print(f"\n✓ Visualization saved to: {output_path}")
    print(f"\nOpen in your browser:")
    print(f"  open {output_path}")

    return output_path


def generate_html_template(js_data):
    """Generate the HTML template with embedded data."""

    # Convert data to JSON
    data_json = json.dumps(js_data, indent=2)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Craftax Trajectory Visualization</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1a1a1a;
            color: #e0e0e0;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        h1 {{
            text-align: center;
            margin-bottom: 10px;
            color: #4CAF50;
        }}

        .metadata {{
            text-align: center;
            margin-bottom: 20px;
            color: #999;
            font-size: 14px;
        }}

        .main-content {{
            display: grid;
            grid-template-columns: 1fr 400px;
            gap: 20px;
            margin-bottom: 20px;
        }}

        .left-panel {{
            background: #2a2a2a;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}

        .right-panel {{
            background: #2a2a2a;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            overflow-y: auto;
            max-height: 700px;
        }}

        .image-container {{
            text-align: center;
            margin-bottom: 20px;
        }}

        #stateImage {{
            max-width: 100%;
            height: auto;
            border: 2px solid #4CAF50;
            border-radius: 4px;
            image-rendering: pixelated;
            image-rendering: crisp-edges;
        }}

        .controls {{
            background: #333;
            padding: 15px;
            border-radius: 8px;
        }}

        .slider-container {{
            margin-bottom: 15px;
        }}

        #timestepSlider {{
            width: 100%;
            height: 8px;
            border-radius: 4px;
            background: #555;
            outline: none;
            -webkit-appearance: none;
        }}

        #timestepSlider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #4CAF50;
            cursor: pointer;
        }}

        #timestepSlider::-moz-range-thumb {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #4CAF50;
            cursor: pointer;
            border: none;
        }}

        .timestep-info {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 15px;
            font-size: 14px;
        }}

        .button-group {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
        }}

        button {{
            background: #4CAF50;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.3s;
        }}

        button:hover {{
            background: #45a049;
        }}

        button:active {{
            background: #3d8b40;
        }}

        .info-section {{
            margin-bottom: 20px;
        }}

        .info-section h3 {{
            color: #4CAF50;
            margin-bottom: 10px;
            font-size: 16px;
            border-bottom: 1px solid #444;
            padding-bottom: 5px;
        }}

        .info-row {{
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            font-size: 14px;
        }}

        .info-label {{
            color: #999;
        }}

        .info-value {{
            color: #e0e0e0;
            font-weight: bold;
        }}

        .text-obs {{
            background: #1a1a1a;
            padding: 10px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            white-space: pre-wrap;
            max-height: 400px;
            overflow-y: auto;
        }}

        .reward-positive {{
            color: #4CAF50;
        }}

        .reward-negative {{
            color: #f44336;
        }}

        .done-true {{
            color: #ff9800;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Craftax Trajectory Visualization</h1>
        <div class="metadata">
            <span id="sessionId"></span> |
            <span id="envName"></span> |
            <span id="totalTimesteps"></span>
        </div>

        <div class="main-content">
            <div class="left-panel">
                <div class="image-container">
                    <img id="stateImage" src="" alt="Game State">
                </div>

                <div class="controls">
                    <div class="timestep-info">
                        <span>Timestep: <strong id="currentTimestep">0</strong></span>
                        <span>Total: <strong id="maxTimestep">0</strong></span>
                    </div>

                    <div class="slider-container">
                        <input type="range" id="timestepSlider" min="0" max="0" value="0">
                    </div>

                    <div class="button-group">
                        <button onclick="firstTimestep()">⏮ First</button>
                        <button onclick="prevTimestep()">◀ Prev</button>
                        <button onclick="nextTimestep()">Next ▶</button>
                        <button onclick="playPause()" id="playButton">▶ Play</button>
                        <button onclick="step10()">+10 ⏩</button>
                        <button onclick="lastTimestep()">Last ⏭</button>
                    </div>
                </div>
            </div>

            <div class="right-panel">
                <div class="info-section">
                    <h3>Current Timestep Info</h3>
                    <div class="info-row">
                        <span class="info-label">Action:</span>
                        <span class="info-value" id="action">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Reward:</span>
                        <span class="info-value" id="reward">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Done:</span>
                        <span class="info-value" id="done">-</span>
                    </div>
                </div>

                <div class="info-section">
                    <h3>State Observation</h3>
                    <div class="text-obs" id="textObs"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Embedded trajectory data
        const DATA = {data_json};

        let currentTimestep = 0;
        let isPlaying = false;
        let playInterval = null;

        // Initialize
        function init() {{
            const metadata = DATA.metadata;
            document.getElementById('sessionId').textContent = metadata.session_id;
            document.getElementById('envName').textContent = metadata.env_name;
            document.getElementById('totalTimesteps').textContent = `${{metadata.num_timesteps}} timesteps`;

            const slider = document.getElementById('timestepSlider');
            slider.max = DATA.timesteps.length - 1;
            document.getElementById('maxTimestep').textContent = DATA.timesteps.length - 1;

            updateDisplay(0);
        }}

        function updateDisplay(t) {{
            currentTimestep = Math.max(0, Math.min(t, DATA.timesteps.length - 1));
            const ts = DATA.timesteps[currentTimestep];

            // Update image
            document.getElementById('stateImage').src = ts.image;

            // Update slider
            document.getElementById('timestepSlider').value = currentTimestep;
            document.getElementById('currentTimestep').textContent = currentTimestep;

            // Update info
            document.getElementById('action').textContent = ts.action;

            const rewardEl = document.getElementById('reward');
            rewardEl.textContent = ts.reward.toFixed(2);
            rewardEl.className = 'info-value ' + (ts.reward > 0 ? 'reward-positive' : ts.reward < 0 ? 'reward-negative' : '');

            const doneEl = document.getElementById('done');
            doneEl.textContent = ts.done ? 'True' : 'False';
            doneEl.className = 'info-value ' + (ts.done ? 'done-true' : '');

            // Update text observation
            document.getElementById('textObs').textContent = ts.text_obs;
        }}

        // Slider event
        document.getElementById('timestepSlider').addEventListener('input', function(e) {{
            updateDisplay(parseInt(e.target.value));
        }});

        // Navigation functions
        function firstTimestep() {{
            updateDisplay(0);
        }}

        function lastTimestep() {{
            updateDisplay(DATA.timesteps.length - 1);
        }}

        function prevTimestep() {{
            updateDisplay(currentTimestep - 1);
        }}

        function nextTimestep() {{
            updateDisplay(currentTimestep + 1);
        }}

        function step10() {{
            updateDisplay(currentTimestep + 10);
        }}

        function playPause() {{
            if (isPlaying) {{
                stopPlay();
            }} else {{
                startPlay();
            }}
        }}

        function startPlay() {{
            isPlaying = true;
            document.getElementById('playButton').textContent = '⏸ Pause';
            playInterval = setInterval(() => {{
                if (currentTimestep >= DATA.timesteps.length - 1) {{
                    stopPlay();
                    return;
                }}
                nextTimestep();
            }}, 200);  // 5 FPS
        }}

        function stopPlay() {{
            isPlaying = false;
            document.getElementById('playButton').textContent = '▶ Play';
            if (playInterval) {{
                clearInterval(playInterval);
                playInterval = null;
            }}
        }}

        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {{
            switch(e.key) {{
                case 'ArrowLeft':
                    prevTimestep();
                    break;
                case 'ArrowRight':
                    nextTimestep();
                    break;
                case ' ':
                    e.preventDefault();
                    playPause();
                    break;
                case 'Home':
                    firstTimestep();
                    break;
                case 'End':
                    lastTimestep();
                    break;
            }}
        }});

        // Initialize on load
        init();
    </script>
</body>
</html>
"""

    return html


if __name__ == "__main__":
    if len(sys.argv) > 1:
        traj_dir = sys.argv[1]
    else:
        # Find most recent trajectory directory
        traj_recordings = Path("trajectory_recordings")
        if traj_recordings.exists():
            trajectories = sorted([p for p in traj_recordings.glob("traj_*") if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
            if trajectories:
                traj_dir = trajectories[0]
            else:
                print("No trajectories found in trajectory_recordings/")
                sys.exit(1)
        else:
            print("No trajectory_recordings directory found")
            sys.exit(1)

    generate_html_visualization(traj_dir)
