import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import logging
from datetime import datetime
from typing import Dict, Any, List
from backend.core.state import AgentState, MessageRole
from scripts.log_config import get_logger
from langsmith import traceable
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from backend.core.utils import executor_tool

logger = get_logger(__name__)

groq_api_key = os.environ.get('GROQ_API_KEY')

@traceable(name="summarization_agent")
async def summarization_agent(state: AgentState) -> AgentState:
    logger.info("---Entering summarization_agent---")
    try:
        agent_name = "summarization_agent"
        plan = getattr(state.processing, "plan", [])
        user_input = state.processing.user_input or ""
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

        # Find the plan step for this agent
        agent_plan = None
        for step in plan:
            if step.get("agent") == agent_name:
                agent_plan = step
                break
        if not agent_plan or not agent_plan.get("tools"):
            logger.warning("No tools specified for summarization_agent in plan. Returning state unchanged.")
            state.processing.current_agent = agent_name
            return state
        # Set the plan to just the tools for this agent
        state.processing.plan = agent_plan["tools"]
        # Call executor_tool to execute the tools
        state = await executor_tool(state)
        # Collect tool responses as context
        tool_responses = []
        if hasattr(state.response, "tool_responses") and state.response.tool_responses:
            for tool_resp in state.response.tool_responses:
                if isinstance(tool_resp, dict) and "response" in tool_resp:
                    tool_responses.append(tool_resp["response"])
                elif isinstance(tool_resp, str):
                    tool_responses.append(tool_resp)
        else:
            # Fallback: use state.response.response if tool_responses not set
            if state.response.response:
                tool_responses.append(state.response.response)
        context = "\n---\n".join(tool_responses)
        
        # LLM to synthesize final summary
        llm = ChatGroq(
            temperature=0.2,
            model_name="meta-llama/llama-4-scout-17b-16e-instruct",
            max_tokens=1024,
            streaming=True,
            api_key=groq_api_key
        )
        if fallback:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """
                You are a helpful assistant. The current date and time is {current_time_date_str}.
                You are being rerun due to a previous error.
                Error: {{error}}
                Fallback Solution: {{fallback_solution}}
                Use the fallback solution and the original input to answer, and ensure you do not repeat the previous error.
                Do not mention the error or the fallback solution in your response.
                Use the provided context and chat history to summarize the user's documents or conversation.
                Context:
                {context}
                Chat History:
                {chat_history}
                Important: Do not mention that you are using tool outputs or document retrieval. Just summarize naturally.
                """),
                ("human", "User request: {input}")
            ])
        else:
            prompt = ChatPromptTemplate([
                ("system", """
                You are a helpful assistant. The current date and time is {current_time_date_str}.
                Use the provided context and chat history to summarize the user's documents or conversation.
                Context:
                {context}
                Chat History:
                {chat_history}
                Important: Do not mention that you are using tool outputs or document retrieval. Just summarize naturally.
                """),
                ("human", "User request: {input}")
            ])

        # Prepare chat history string
        chat_history = state.chat_history[-10:]
        _chat_history = []
        for msg in chat_history:
            if msg.role == MessageRole.USER:
                _chat_history.append(("user", msg.content))
            elif msg.role == MessageRole.ASSISTANT:
                _chat_history.append(("assistant", msg.content))

        chain = prompt | llm | StrOutputParser()
        chain_input = {
            "context": context,
            "chat_history": _chat_history,
            "input": user_input,
            "current_time_date_str": datetime.now().isoformat()
        }
        final_response = ""
        async for chunk in chain.astream(chain_input):
            final_response += str(chunk)
        final_response = final_response.strip()
        state.response.response = final_response
        state.processing.current_agent = agent_name
        if not state.response.response_metadata:
            state.response.response_metadata = {}
        state.response.response_metadata.update({
            "agent_type": agent_name,
            "processing_time": datetime.now().isoformat()
        })
        logger.info(f"Summarization agent executed tools and synthesized final summary.")
        logger.info("---End of summarization_agent---")
        state.processing.executed_steps.append("summarization_agent")
        return state
    except Exception as e:
        logger.error(f"Error in summarization agent: {str(e)}")
        state.error.error = f"Summarization error: {str(e)}"
        state.response.response = "Error in summarization agent."
        state.processing.current_agent = "summarization_agent"
        logger.info(f"Returning from summarization_agent with type: {type(state)} (error path)")
        logger.info("---End of summarization_agent---")
        return state 