import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from backend.core.state import AgentState, MessageRole
from scripts.log_config import get_logger
import os
from datetime import datetime
from langsmith import traceable
from pydantic import BaseModel, Field

logger = get_logger(__name__)

groq_api_key = os.environ.get('GROQ_API_KEY')

@traceable(name="content_moderation_agent")
async def content_moderation_agent(state: AgentState) -> AgentState:
    logger.info("---Entering content_moderation_agent---")
    try:
        user_input = state.processing.user_input or ""
        agent_name = "content_moderation_agent"
        
        fallback = False
        fallback_solution = None
        error_message = None
        if (
            state.processing.current_agent == "fallback_agent"
            and state.response.response_metadata.get("rerun_agent") == agent_name
            and state.response.response_metadata.get("fallback")
        ):
            fallback_solution = state.response.response_metadata["fallback"]
            error_message = state.error.error or ""
            fallback = True

        llm = ChatGroq(
            temperature=0.1,
            model_name="llama3-8b-8192",
            max_tokens=512,
            streaming=True,
            api_key=groq_api_key
        )

        # Inline the system message directly instead of using {prompt_template} as a variable
        system_message = r"""
        You are content moderation agent. Your primary goal is to be a safety and content moderation assistant providing a helpful message to the user.
        current date and time: {current_time_date_str}
        If the user input indicates harmful, dangerous, or sensitive content (such as self-harm, suicide, violence, etc.), provide a supportive message encouraging the user to seek help.
        If the user input is not harmful, dangerous, or sensitive, provide a clear, concise, and helpful message for the user.
        """
        if fallback:
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_message + "\nYou are being rerun due to a previous error.\nError: {{error}}\nFallback Solution: {{fallback_solution}}\nUse the fallback solution and the original input to answer, and ensure you do not repeat the previous error.\nDo not mention the error or the fallback solution in your response."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "User input: {input}\n\n")
            ])
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "User input: {input}\n\n")
            ])
        
        chat_history = state.chat_history[-10:]
        _chat_history = []
        for msg in chat_history:
            if msg.role == MessageRole.USER:
                _chat_history.append(("user", msg.content))
            elif msg.role == MessageRole.ASSISTANT:
                _chat_history.append(("assistant", msg.content))

        chain = prompt | llm | StrOutputParser()
        chain_input = {
            "input": user_input,
            "chat_history": _chat_history,
            "current_time_date_str": datetime.now().isoformat()
        }
        
        response = ""
        async for chunk in chain.astream(chain_input):
            response += str(chunk)

        response = response.strip()
        logger.info(f"Content moderation agent output: {response}")

        state.response.response = response
        state.processing.current_agent = agent_name
        if not state.response.response_metadata:
            state.response.response_metadata = {}
        state.response.response_metadata.update({
            "agent_type": agent_name,
            "processing_time": datetime.now().isoformat()
        })
        state.processing.executed_steps.append("content_moderation_agent")
        logger.info(f"Returning from content_moderation_agent with type: {type(state)}")
        logger.info("---End of content_moderation_agent---")
        return state
    except Exception as e:
        logger.error(f"Error in content moderation agent: {str(e)}")
        state.error.error = f"Content moderation agent processing error: {str(e)}"
        state.response.response = "I'm sorry, I encountered an error while processing your request. Please try again."
        state.processing.current_agent = "content_moderation_agent"
        logger.info(f"Returning from content_moderation_agent with type: {type(state)} (error path)")
        logger.info("---End of content_moderation_agent---")
        return state
