import sys
from pathlib import Path
from datetime import datetime

# Add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from langsmith import traceable
from backend.core.state import AgentState, MessageRole
import os
from scripts.log_config import get_logger
import json

logger = get_logger(__name__)

class RouteDecision(BaseModel):
    """Output model for routing decisions"""
    agent: str = Field(description="The agent that should handle this request")
    confidence: float = Field(description="Confidence score for the routing decision (0-1)")
    reasoning: str = Field(description="Reason for choosing this agent")
    requires_context: bool = Field(description="Whether this request requires document context")
    is_greeting: bool = Field(description="Whether this is a greeting or general conversation")

groq_api_key = os.environ.get('GROQ_API_KEY')

@traceable(name="router_agent")
async def router_agent(state: AgentState) -> AgentState:
    logger.info("---Entering router_agent---")
    try:
        user_input = state.processing.user_input or ""
        if not user_input:
            state.processing.route_decision = "fallback_agent"
            state.error.error = "No user input provided"
            state.processing.current_agent = "router_agent"
            return state
        fallback = False
        fallback_solution = None
        error_message = None
        agent_name = "router_agent"
        if (
            state.processing.current_agent == "fallback_agent"
            and state.response.response_metadata.get("rerun_agent") == agent_name
            and state.response.response_metadata.get("fallback")
        ):
            fallback_solution = state.response.response_metadata["fallback"]
            error_message = state.error.error or ""
            fallback = True

        
        # Inline the system message directly instead of using {prompt_template} as a variable
        system_message = r'''
        You are a routing agent for a multi-agent system. The current date and time is {current_time_date_str}.
        You determine which  specialized agent should handle user requests related to Bachelor thesis projects.
        Available agents:
        1. 'conversation_agent' - For greetings, onboarding, general questions about using the system, or non-thesis/non-research queries. Use when the user is starting a conversation, asking for help, or their intent is unclear but conversational.
        2. 'planning_agent' - For requests that require multi-step reasoning, workflow planning, or chaining multiple tools/agents together (e.g., document retrieval, summarization, analysis)
        3. 'content_moderation_agent' - For inappropriate, harmful, or blocked content
        You must respond with ONLY a valid JSON object containing these exact keys:
        - "agent": one of 'conversation_agent', 'planning_agent', or 'content_moderation_agent'
        - "confidence": a float between 0 and 1
        - "reasoning": a short string explaining your choice
        - "requires_context": true or false
        - "is_greeting": true or false
        Do not include any other text, only the JSON object.
        '''
        if fallback:
            routing_prompt = ChatPromptTemplate.from_messages([
                ("system", system_message + "\nYou are being rerun due to a previous error.\nError: {{error}}\nFallback Solution: {{fallback_solution}}\nUse the fallback solution and the original input to answer, and ensure you do not repeat the previous error.\nDo not mention the error or the fallback solution in your response."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "User input: {input}\n\nReturn only the JSON object:")
            ])
        else:
            routing_prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "User input: {input}\n\nReturn only the JSON object:")
            ])

        chat_history = state.chat_history[-10:]
        _chat_history = []
        for msg in chat_history:
            if msg.role == MessageRole.USER:
                _chat_history.append(("user", msg.content))
            elif msg.role == MessageRole.ASSISTANT:
                _chat_history.append(("assistant", msg.content))
        
        logger.info(f"Chat history: {_chat_history}")

        llm = ChatGroq(temperature=0.1,
                       model_name="meta-llama/llama-4-scout-17b-16e-instruct",
                       max_tokens=512,
                       streaming=True,
                       api_key=groq_api_key
            )

        parser = JsonOutputParser(pydantic_object=RouteDecision)
        routing_chain = routing_prompt | llm | parser
        chain_input = {
            "current_time_date_str": datetime.now().isoformat(),
            "input": user_input,
            "chat_history": _chat_history
        }
        route_result = await routing_chain.ainvoke(chain_input)
        logger.info(f"Router agent output: {route_result}")

        state.processing.route_decision = route_result.get("agent", "fallback_agent")
        state.processing.confidence_score = route_result.get("confidence", 0.0)
        state.processing.current_agent = route_result.get("agent", "router_agent")
        if not state.response.response_metadata:
            state.response.response_metadata = {}
        state.response.response_metadata.update({
            "agent_type": state.processing.current_agent,
            "processing_time": datetime.now().isoformat()
        })

        logger.info(f"Routing decision: {route_result.get('agent')} (confidence: {route_result.get('confidence')})")

        state.processing.executed_steps.append("router_agent")
        return state
    except Exception as e:
        logger.error(f"Error in router agent: {str(e)}")
        state.processing.route_decision = "fallback_agent"
        state.error.error = f"Routing error: {str(e)}"
        state.processing.current_agent = "router_agent"
        logger.info("---End of router_agent---")
        return state
