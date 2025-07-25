"""
Chat router for the SavvyThreads API.
"""

import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
import json
from backend.api.models import UserRequest, ChatResponse
from scripts.log_config import get_logger
from backend.core.orchestrator import process_message

router = APIRouter(
    prefix="/api/v1/chat",
    tags=["Chat"]
)
logger = get_logger(__name__)

@router.post("/")
async def chat(request: UserRequest):
    logger.info(f"Received chat request: {request}")
    chat_id = f"{request.user_id}_default"
    async def event_stream():
        try:
            # Import the orchestrator here to avoid circular import issues
            from backend.core.orchestrator import process_message_stream
            async for chunk in process_message_stream(request.user_id, chat_id, request.message):
                yield json.dumps(chunk) + "\n"
        except Exception as e:
            logger.error(f"Chat error: {e}")
            yield json.dumps({"type": "error", "error": str(e)}) + "\n"
    return StreamingResponse(event_stream(), media_type="application/json")

