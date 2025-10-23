import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=str(ENV_PATH))

DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
DASHBOARDS_DIR = BASE_DIR / "dashboards"
MEMORY_DIR = BASE_DIR / "memory"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
