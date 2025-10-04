from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
FIGURE_DIR = PROJECT_DIR / "figures"
CONFIG_DIR = PROJECT_DIR / "configs"
MODEL_WEIGHTS_DIR = Path("/model-weights")
CACHE_DIR = PROJECT_DIR / "caches"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_DIR / ".env")
    OPENAI_API_KEY: str
    GEMINI_API_KEY: str
    OPENROUTER_API_KEY: str


settings = Settings()  # type: ignore
