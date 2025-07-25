from pydantic import BaseModel, Field

# User request
class UserRequest(BaseModel):
    user_id: str = Field(..., description="The user ID")
    message: str = Field(..., description="The message to send to the API")

# Chat Response
class ChatResponse(BaseModel):
    response: str = Field(..., description="The response from the API")