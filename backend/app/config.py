from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://fencing:fencing@postgres:5432/fencing_analyzer"
    redis_url: str = "redis://redis:6379/0"

    s3_bucket: str = "fencing-videos"
    s3_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    s3_endpoint_url: str = ""

    anthropic_api_key: str = ""
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "llama3.1:8b"

    secret_key: str = "change-me-in-production"
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]


settings = Settings()
