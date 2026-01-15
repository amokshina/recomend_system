from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    db_host: str = "postgres"
    db_port: int = 5432
    db_name: str = "recsys"
    db_user: str = "recsys"
    db_password: str = "recsys"

    artifacts_dir: str = "/app/artifacts"

    @property
    def db_url(self) -> str:
        return f"postgresql+psycopg2://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

settings = Settings()
