import os


class Settings:
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    retry_max_attempts: int = int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))
    retry_min_wait_seconds: float = float(os.getenv("RETRY_MIN_WAIT_SECONDS", "1.0"))
    retry_max_wait_seconds: float = float(os.getenv("RETRY_MAX_WAIT_SECONDS", "30.0"))
    retry_jitter: float = float(os.getenv("RETRY_JITTER", "0.5"))


settings = Settings()

if __name__ == "__main__":
    settings = Settings()
    print(settings.openai_base_url)
