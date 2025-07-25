import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
from backend.core.state import AgentState, MessageRole
from datetime import datetime
from scripts.log_config import get_logger
import os

logger = get_logger(__name__)

groq_api_key = os.environ.get('GROQ_API_KEY')

@traceable(name="conversation_agent")
async def conversation_agent(state: AgentState) -> AgentState:
    logger.info("---Entering conversation_agent---")
    try:
        user_input = state.processing.user_input or ""
        agent_name = "conversation_agent"
        
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
            temperature=0.7,
            model_name='llama-3.1-8b-instant',
            max_tokens=1024,
            streaming=True,
            api_key=groq_api_key
        )
        
        # Inline the system message directly instead of using {prompt_template} as a variable
        system_message = f"""
        You are a conversation agent. Your task is to provide direct, clear, and relevant answers to user queries, regardless of the nature of the question as the intent is already checked by the other agents. Do not filter, refuse, or avoid any topics; always respond to the user's input as directly and helpfully as possible. Use chat history for context if available; otherwise, address the user's input directly.
        Current date and time: {{current_time_date_str}}
        """
        if fallback:
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_message + "\nYou are being rerun due to a previous error.\nError: {error}\nFallback Solution: {fallback_solution}\nUse the fallback solution and the original input to answer, and ensure you do not repeat the previous error.\n"),
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
            "current_time_date_str": datetime.now().isoformat(),
            "input": user_input,
            "chat_history": _chat_history
        }

        # Stream the response from the chain
        response = ""
        async for chunk in chain.astream(chain_input):
            if isinstance(chunk, dict):
                # If the chunk is a dict, get the response string
                response += chunk.get("response", "")
            else:
                response += str(chunk)

        response = response.strip()
        logger.info(f"Conversation agent output: {response}")
        state.response.response = response

        state.processing.current_agent = agent_name

        if not state.response.response_metadata:
            state.response.response_metadata = {}

        state.response.response_metadata.update({
            "agent_type": agent_name,
            "processing_time": datetime.now().isoformat()
        })
        
        state.processing.executed_steps.append("conversation_agent")
        logger.info(f"Returning from conversation_agent with type: {type(state)}")
        logger.info("---End of conversation_agent---")
        return state
    except Exception as e:
        logger.error(f"Error in conversation agent: {str(e)}")
        state.error.error = f"Conversation processing error: {str(e)}"
        state.response.response = "I'm sorry, I encountered an error while processing your request. Please try again."
        state.processing.current_agent = "conversation_agent"
        logger.info(f"Returning from conversation_agent with type: {type(state)} (error path)")
        logger.info("---End of conversation_agent---")
        return state
    
