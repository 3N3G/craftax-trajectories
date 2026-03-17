"""Craftax Trajectory Recorder - Record human gameplay for training RL agents."""

__version__ = "0.1.0"

from .play import main, TrajectoryRecorder

__all__ = ["main", "TrajectoryRecorder"]
