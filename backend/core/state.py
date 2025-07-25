from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

# --- Basic Message Structures ---

class MessageRole(str, Enum):
    """Message roles for chat history"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class Message(BaseModel):
    """Individual message in chat history"""
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

# --- Core Chat and History Structures ---

class Chat(BaseModel):
    """Represents a single conversation thread."""
    chat_id: str
    user_id: str
    messages: List[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)

    def add_message(self, role: MessageRole, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a new message to the chat."""
        message = Message(role=role, content=content, metadata=metadata or {})
        self.messages.append(message)
        self.last_updated = datetime.now()

    def get_context_window(self, max_tokens: int = 4000) -> List[Message]:
        """Get messages from the chat within a token limit for context."""
        # Simple token estimation (4 chars per token)
        total_chars = 0
        context_messages = []
        
        for message in reversed(self.messages):
            message_chars = len(message.content)
            if total_chars + message_chars > max_tokens * 4:
                break
            context_messages.insert(0, message)
            total_chars += message_chars
        
        return context_messages
    
    def needs_summarization(self, token_threshold: int = 2000) -> bool:
        """Check if the chat exceeds the token threshold for summarization."""
        total_chars = sum(len(message.content) for message in self.messages)
        return total_chars > token_threshold * 4

class LongTermHistory(BaseModel):
    """Long-term history with summarized conversations for a single user."""
    user_id: str
    summaries: Dict[str, Dict[str, Any]] = Field(default_factory=dict)  # chat_id -> summary
    key_topics: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)

    def add_summary(self, chat_id: str, summary: Dict[str, Any]):
        """Add a new chat summary."""
        self.summaries[chat_id] = summary
        self.last_updated = datetime.now()

class WebSearchState(BaseModel):
    """State for tracking web search results."""
    web_search_query: Optional[str] = Field(default_factory=str)
    web_search_results: Optional[List[str]] = Field(default_factory=list)

# --- Composable State Snippets for Graph Execution ---

class ProcessingState(BaseModel):
    """State for tracking the current processing status of a request."""
    user_input: Optional[str] = None
    is_processing: bool = False
    current_agent: Optional[str] = None
    route_decision: Optional[str] = None
    confidence_score: Optional[float] = None
    next_agent: Optional[str] = None
    plan: List[Dict[str, Any]] = Field(default_factory=list)
    replan_attempts: int = 0  # Number of times the system has replanned for this input
    last_tool: Optional[str] = None  # Track the last tool used
    web_search: WebSearchState = Field(default_factory=WebSearchState)
    executed_steps: List[str] = Field(default_factory=list)  # Track executed agents/tools

class RetrievalState(BaseModel):
    """State for storing retrievals for each user input in a chat. Structure: {user_input: [retrieved_documents]}"""
    chat_id: Optional[str] = None
    retrieved_documents: Dict[str, list] = Field(default_factory=dict)  # user_input -> list of docs

class ResponseState(BaseModel):
    """State for storing the agent's response."""
    response: Optional[str] = None
    tool_responses: List[Dict[str, Any]] = Field(default_factory=list)
    response_metadata: Dict[str, Any] = Field(default_factory=dict)

class ErrorState(BaseModel):
    """State for tracking errors."""
    error: Optional[str] = None
    error_details: Dict[str, Any] = Field(default_factory=dict)

# --- The Main State Object for the Graph ---

class AgentState(BaseModel):
    """
    The main state passed through the LangGraph workflow.
    It's a composite of various state snippets.
    """
    user_id: str
    chat_id: str
    
    # The full history of the current chat, for context.
    chat_history: List[Message] = Field(default_factory=list)
    
    # Composable state parts
    processing: ProcessingState = Field(default_factory=ProcessingState)
    retrieval: RetrievalState = Field(default_factory=RetrievalState)
    response: ResponseState = Field(default_factory=ResponseState)
    error: ErrorState = Field(default_factory=ErrorState)
    
    # For providing long-term context to agents
    long_term_history_for_context: Optional[LongTermHistory] = None

    class Config:
        arbitrary_types_allowed = True

# --- State Management for the Entire Application ---

class UserState:
    """Manages all data for a single user, including all their chats and history."""
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.chats: Dict[str, Chat] = {}
        self.long_term_history = LongTermHistory(user_id=user_id)
        
    def get_chat(self, chat_id: str) -> Optional[Chat]:
        return self.chats.get(chat_id)
        
    def new_chat(self) -> Chat:
        chat_id = f"chat_{datetime.now().timestamp()}"
        chat = Chat(chat_id=chat_id, user_id=self.user_id)
        self.chats[chat_id] = chat
        return chat

    def summarize_chat_if_needed(self, chat_id: str):
        chat = self.get_chat(chat_id)
        if chat and chat.needs_summarization() and chat_id not in self.long_term_history.summaries:
            # Placeholder for summarization logic
            summary_content = f"Summary of chat {chat_id} with {len(chat.messages)} messages."
            # TODO: Implement summarization logic
            summary = {
                "chat_id": chat.chat_id,
                "message_count": len(chat.messages),
                "timestamp": datetime.now(),
                "summary": summary_content
            }
            self.long_term_history.add_summary(chat_id, summary)

class StateManager:
    """
    Global manager for all user states.
    Maps user_id to their respective UserState.
    """
    def __init__(self):
        self.user_states: Dict[str, UserState] = {}

    def get_user_state(self, user_id: str) -> UserState:
        """Get or create state for a user."""
        if user_id not in self.user_states:
            self.user_states[user_id] = UserState(user_id=user_id)
        return self.user_states[user_id]

    def get_all_user_states(self) -> Dict[str, UserState]:
        """Get all user states."""
        return self.user_states.copy()
