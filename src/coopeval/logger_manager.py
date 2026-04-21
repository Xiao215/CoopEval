"""Utility helpers for logging experiment outputs locally."""

import json
import os
import threading
from datetime import datetime
from pathlib import Path

from coopeval.config import OUTPUTS_DIR


class Logger:
    """Create per-run output directories and provide basic JSON/TXT append helpers."""

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
        self._log_dir = None
        self._custom_dir = custom_dir
        self._lock = threading.Lock()

    @property
    def log_dir(self) -> Path:
        """
        Get the log directory, creating it lazily on first access.
        """
        if self._log_dir is None:
            with self._lock:
                if self._log_dir is None:
                    if self._custom_dir is not None:
                        self._log_dir = Path(self._custom_dir)
                    else:
                        now = datetime.now()
                        self._log_dir = (
                            self._base_dir
                            / f"{now.year}"
                            / f"{now.month:02}"
                            / f"{now.day:02}"
                            / f"{now.hour:02}:{now.minute:02}:{now.second:02}"
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
                        json.dump(record, f, default=lambda x: x.serialize())
                        f.write("\n")
                case ".json":
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(
                            record, f, indent=2, default=lambda x: x.serialize()
                        )
                case _:
                    raise ValueError(f"Unsupported file type: {path.suffix}")

    def append_to_txt(self, content: str, filename: str) -> None:
        """
        Append a string into a .txt file inside log_dir.
        """
        path = self.log_dir / filename
        with self._lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(content)


LOGGER = Logger()
