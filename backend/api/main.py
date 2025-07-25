import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from fastapi import FastAPI
from api.router import router as chat_router
from scripts.log_config import setup_logging, get_logger

setup_logging(task_type="api")


logger = get_logger("API")

app = FastAPI(title="Terradata Assignment API", version="1.0.0", description="FastAPI backend for SavvyThreads")
logger.info("Starting Terradata Assignment API...")

@app.get("/", tags=["Root"], response_model=dict)
async def root() -> dict:
    logger.info("Root endpoint accessed.")
    
    return {"message": "Welcome to Terrdata Assignment API"}

logger.info("Including chat router...")
app.include_router(chat_router)
logger.info("Chat router included.")
