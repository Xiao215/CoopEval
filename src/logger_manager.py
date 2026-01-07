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
    def __init__(self, base_dir: Path = OUTPUTS_DIR) -> None:
        """
        Initialize logging directory and any integrations.
        Creates a timestamped subdirectory under `base_dir`.
        """
        now = datetime.now()
        self.log_dir = (
            base_dir
            / f"{now.year}"
            / f"{now.month:02}"
            / f"{now.day:02}"
            / f"{now.hour:02}:{now.minute:02}"
        )
        os.makedirs(self.log_dir, exist_ok=True)
        self._lock = threading.Lock()

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
            os.rename(self.log_dir, new_path)
            self.log_dir = new_path


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
