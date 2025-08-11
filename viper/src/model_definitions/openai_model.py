import os
import warnings
from .base_model import BaseModel
import threading
from openai import OpenAI


class OpenAIModel(BaseModel):
    _api_key: str | None = None
    __client: OpenAI | None = None
    _lock = threading.Lock()

    def __init__(self, gpu_number=0):
        super().__init__(gpu_number=gpu_number)

    # --- helpers ------------------------------------------------------------
    @classmethod
    def _load_key_from_env(cls) -> str | None:
        """
        Try OPENAI_API_KEY first; if absent, try OPENAI_API_KEY_PATH (file with the key).
        """
        key = os.getenv("OPENAI_API_KEY")
        if key:
            return key.strip()

        path = os.getenv("OPENAI_API_KEY_PATH")
        if path:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except OSError as e:
                warnings.warn(f"Couldn't read OPENAI_API_KEY_PATH={path}: {e}")
        return None

    @classmethod
    def set_api_key(cls, api_key: str | None):
        """
        Explicitly set the key; passing None clears it so _client() will try env/file.
        """
        cls._api_key = api_key
        cls.__client = OpenAI(api_key=cls._api_key) if cls._api_key else None

    @classmethod
    def _client(cls) -> OpenAI:
        with cls._lock:
            if cls.__client is None:
                # If no explicit key, try to load from env/file just-in-time.
                if not cls._api_key:
                    env_key = cls._load_key_from_env()
                    if env_key:
                        cls._api_key = env_key
                if not cls._api_key:
                    raise RuntimeError(
                        "OpenAI API key is not set. "
                        "Set OPENAI_API_KEY, OPENAI_API_KEY_PATH, or call OpenAIModel.set_api_key()."
                    )
                cls.__client = OpenAI(api_key=cls._api_key)
            return cls.__client

    @property
    def client(self) -> OpenAI:
        return self.__class__._client()
