"""Modular configuration loading system."""

from pathlib import Path
from typing import Any

import yaml

from config import CONFIG_DIR


class ConfigLoader:
    """Handles loading and merging modular configuration files."""

    def __init__(self):
        self.config_dir = Path(CONFIG_DIR)

    def load_main_config(self, main_config_path: str) -> dict:
        """
        Load a main config and resolve all component references.

        Args:
            main_config_path: Path to main config (relative to CONFIG_DIR)

        Returns:
            Fully resolved configuration dictionary with keys:
            - 'game': Game configuration dict
            - 'mechanism': Mechanism configuration dict
            - 'agents': List of agent configurations
            - 'evaluation': Evaluation configuration dict (OPTIONAL - may be absent)
            - 'concurrency': Concurrency settings (if specified in main config)
            - 'name': Experiment name (if specified in main config)

        Note:
            The 'evaluation' key will NOT be present if evaluation_config is omitted
            from the main config. Downstream code should use config.get('evaluation')
            rather than config['evaluation'].
        """
        main_path = self._resolve_config_path(main_config_path)
        main = self._load_yaml(main_path)

        # Check if this is a modular config or legacy monolithic
        if self._is_modular_config(main):
            return self._load_modular_config(main)
        else:
            # Legacy monolithic config - return as-is
            return main

    def _is_modular_config(self, config: dict) -> bool:
        """
        Detect if config is modular (has component references).

        Requires ALL three core gameplay components to be present
        to avoid false positives.
        """
        modular_keys = {'game_config', 'mechanism_config', 'agents_config'}
        return all(key in config for key in modular_keys)

    def _load_modular_config(self, main: dict) -> dict:
        """Load and merge all component configs referenced in main."""
        resolved = {}

        # Load game component
        if 'game_config' in main:
            game_path = self._resolve_config_path(main['game_config'])
            resolved['game'] = self._load_yaml(game_path)

        # Load mechanism component
        if 'mechanism_config' in main:
            mech_path = self._resolve_config_path(main['mechanism_config'])
            resolved['mechanism'] = self._load_yaml(mech_path)

        # Load agents component
        if 'agents_config' in main:
            agents_path = self._resolve_config_path(main['agents_config'])
            # Agent configs are lists directly (no wrapper key)
            resolved['agents'] = self._load_yaml(agents_path)

        # Load evaluation component (OPTIONAL - may be omitted for reputation mechanisms)
        if 'evaluation_config' in main:
            evaluation_path = self._resolve_config_path(main['evaluation_config'])
            resolved['evaluation'] = self._load_yaml(evaluation_path)
        # Note: If evaluation_config is omitted, 'evaluation' key will NOT be present in returned dict

        # Add experiment-level settings from main
        if 'concurrency' in main:
            resolved['concurrency'] = main['concurrency']

        if 'name' in main:
            resolved['name'] = main['name']

        return resolved

    def load_component(self, component_path: str, component_type: str):
        """
        Load a single component config.

        Args:
            component_path: Path to component (relative to CONFIG_DIR)
            component_type: Type of component (game, mechanism, agents, evaluation)

        Returns:
            Component configuration (dict for game/mechanism/evaluation, list for agents)
        """
        path = self._resolve_config_path(component_path)
        return self._load_yaml(path)

    def _resolve_config_path(self, config_path: str) -> Path:
        """
        Resolve a config path relative to CONFIG_DIR.

        Handles both:
        - Relative paths: 'games/pd_standard.yaml'
        - Absolute paths: '/full/path/to/config.yaml'
        """
        path = Path(config_path)

        if path.is_absolute():
            return path

        # Try relative to config_dir first
        resolved = self.config_dir / path
        if resolved.exists():
            return resolved

        # Try adding .yaml extension if missing
        if not config_path.endswith('.yaml'):
            resolved_with_ext = self.config_dir / f"{config_path}.yaml"
            if resolved_with_ext.exists():
                return resolved_with_ext

        # If still not found, raise error
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Searched: {resolved}"
        )

    def _load_yaml(self, path: Path) -> dict:
        """Load and parse a YAML file."""
        if not path.exists():
            raise FileNotFoundError(f"Config file {path} not found.")

        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
