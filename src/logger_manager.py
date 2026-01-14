import json
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import wandb
from matplotlib.figure import Figure

from config import OUTPUTS_DIR


class Logger:
    def __init__(
        self, base_dir: Path = OUTPUTS_DIR, custom_dir: Path | None = None
    ) -> None:
        """
        Initialize logging directory and any integrations.

        Args:
            base_dir: Base directory for outputs (default: OUTPUTS_DIR)
            custom_dir: If provided, use this exact directory instead of creating
                       a timestamped subdirectory. This is useful for batch operations.
        """
        self._base_dir = base_dir
        self._log_dir = None  # Will be created lazily
        self._custom_dir = custom_dir
        self._lock = threading.Lock()

    @property
    def log_dir(self) -> Path:
        """
        Get the log directory, creating it lazily on first access.
        """
        if self._log_dir is None:
            with self._lock:
                # Double-check after acquiring lock
                if self._log_dir is None:
                    if self._custom_dir is not None:
                        self._log_dir = Path(self._custom_dir)
                    else:
                        # Original timestamping logic
                        now = datetime.now()
                        self._log_dir = (
                            self._base_dir
                            / f"{now.year}"
                            / f"{now.month:02}"
                            / f"{now.day:02}"
                            / f"{now.hour:02}:{now.minute:02}"
                        )
                    os.makedirs(self._log_dir, exist_ok=True)
        return self._log_dir

    def set_log_dir(self, log_dir: Path) -> None:
        """
        Override the current log directory with a custom path.
        Creates the directory if it doesn't exist.

        This allows external code to redirect all logging to a specific location
        without recreating the Logger instance.

        Args:
            log_dir: The new directory path for logging
        """
        with self._lock:
            self._log_dir = Path(log_dir)
            os.makedirs(self._log_dir, exist_ok=True)

    def log_record(self, record: dict | list, file_name: str) -> None:
        """
        Log the evolution record to a JSON or JSONL file inside log_dir.
        """
        path = self.log_dir / file_name
        with self._lock:
            match path.suffix:
                case ".jsonl":
                    with open(path, "a", encoding="utf-8") as f:
                        json.dump(record, f)
                        f.write("\n")
                case ".json":
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(record, f, indent=2)
                case _:
                    raise ValueError(f"Unsupported file type: {path.suffix}")

    def write_to_txt(self, content: str, filename: str) -> None:
        """
        Write a string into a .txt file inside log_dir.
        Overwrites if the file does not exist.
        """
        path = self.log_dir / filename
        with self._lock:
            if not path.exists():
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)

    def append_to_txt(self, content: str, filename: str) -> None:
        """
        Append a string into a .txt file inside log_dir.
        """
        path = self.log_dir / filename
        with self._lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(content)

    def retag(self, suffix: str) -> None:
        """Rename the current log directory to include run-specific metadata."""
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", suffix).strip("-")
        if not slug:
            return
        with self._lock:
            parent = self.log_dir.parent
            base_name = self.log_dir.name
            new_path = parent / f"{base_name}_{slug}"
            counter = 1
            while new_path.exists():
                counter += 1
                new_path = parent / f"{base_name}_{slug}-{counter}"
            os.rename(self._log_dir, new_path)
            self._log_dir = new_path


LOGGER = Logger()


class WandBLogger:
    """Minimal wrapper so your plotting code never touches wandb directly."""

    def __init__(self, project: str, config: dict[str, Any] | None = None):
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run = wandb.init(
            project=project,
            name=run_name,
            config=config or {},
            save_code=True,
            reinit=True,
        )

    def log_figure(self, fig: Figure, name: str):
        """Log a Matplotlib figure as an image artefact."""
        self.run.log({name: wandb.Image(fig)})
