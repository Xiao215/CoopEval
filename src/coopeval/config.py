"""Global configuration defaults and directories used across CoopEval utilities."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = PROJECT_ROOT / "figures"
CONFIG_DIR = PROJECT_ROOT / "configs"
MODEL_WEIGHTS_DIR = PROJECT_ROOT / "model-weights"
CACHE_DIR = PROJECT_ROOT / "caches"
DATA_DIR = PROJECT_ROOT / "data"
JUDGE_OUTPUT_DIR = OUTPUTS_DIR / "judge"
LATEX_DIR = PROJECT_ROOT / "latex"

class Settings(BaseSettings):
    """Load API credentials from `.env` so downstream scripts can reuse them.

    Keys default to `None` so that local development can proceed without
    configuring every provider; downstream callers should validate before use.
    """

    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env")
    OPENAI_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    OPENROUTER_API_KEY: str | None = None


settings = Settings()
