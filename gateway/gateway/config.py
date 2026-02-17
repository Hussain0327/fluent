from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Twilio
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""

    # LLM API keys
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # Database
    database_url: str = "postgresql://gateway:gateway@localhost:5432/personaplex"

    # PersonaPlex
    personaplex_ws_url: str = "ws://personaplex:8998/api/chat"
    default_voice_prompt: str = "NATF0.pt"

    # LLM defaults
    default_llm_provider: str = "claude"  # "claude" or "openai"
    default_llm_model: str = "claude-sonnet-4-20250514"
    embedding_model: str = "text-embedding-3-small"

    # Server
    gateway_host: str = "0.0.0.0"
    gateway_port: int = 8080

    # Memory
    memory_top_k: int = 10
    conversation_idle_timeout_minutes: int = 30

    model_config = {"env_prefix": "", "case_sensitive": False}


settings = Settings()
