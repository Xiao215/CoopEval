from datetime import datetime
from typing import Any
import os
from matplotlib.figure import Figure

import wandb
from config import OUTPUTS_DIR
from pathlib import Path
import json


now = datetime.now()
log_dir = (
    OUTPUTS_DIR
    / f"{now.year}"
    / f"{now.month:02}"
    / f"{now.day:02}"
    / f"{now.hour:02}:{now.minute:02}"
)
os.makedirs(log_dir, exist_ok=True)


def log_record(record: dict | list, file_name: str) -> None:
    """Log the evolution record to a JSON file."""
    path = log_dir / file_name
    if path.suffix == ".jsonl":
        with open(path, "a", encoding="utf-8") as f:
            json.dump(record, f)
            f.write("\n")
    elif path.suffix == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")


def write_to_txt(content: str, filename: str) -> None:
    """
    Write a string into a .txt file.
    """
    path = log_dir / filename
    if not path.exists():
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)


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
