"""Utility functions for trajectory recording."""

import zipfile
from pathlib import Path


def create_zip_archive(trajectory_dir: Path) -> Path:
    """Create a zip archive of the trajectory directory.

    Args:
        trajectory_dir: Path to the trajectory directory

    Returns:
        Path to the created zip file
    """
    trajectory_dir = Path(trajectory_dir)
    zip_path = trajectory_dir.parent / f"{trajectory_dir.name}.zip"

    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in trajectory_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(trajectory_dir.parent)
                    zipf.write(file_path, arcname)

        file_size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"Created zip archive: {zip_path.name} ({file_size_mb:.2f} MB)")
        return zip_path

    except Exception as e:
        print(f"Warning: Failed to create zip archive: {e}")
        return None
