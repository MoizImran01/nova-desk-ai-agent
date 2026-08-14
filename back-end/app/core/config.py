from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Having a default string is okay for non-sensitive things like the project name
    PROJECT_NAME: str = "Nova Desk Backend"
    
    # By NOT giving these a default value, we force Pydantic to look for them in the .env file.
    # If they are missing from the .env file, the app will refuse to start.
    DATABASE_URL: str
    GROQ_API_KEY: str
    GOOGLE_GEMINI_API_KEY: str
    PINECONE_API_KEY: str

    # This tells Pydantic to read variables from the .env file
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

# Create a global instance to import anywhere in your app
settings = Settings()