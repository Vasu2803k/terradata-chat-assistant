import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

import os
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from langsmith import traceable
from backend.core.state import AgentState, MessageRole
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
from scripts.log_config import get_logger
import json

logger = get_logger(__name__)

groq_api_key = os.environ.get('GROQ_API_KEY')

class ToolStep(BaseModel):
    tool: str = Field(description="The tool to call (rag_tool or web_search_tool)")
    args: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Arguments for the tool")

class AgentPlan(BaseModel):
    agent: str = Field(description="The agent to use (analysis_agent, summarization_agent)")
    tools: Optional[List[ToolStep]] = Field(description="A list of tools (rag_tool, web_search_tool)")

class PlanOutput(BaseModel):
    plan: List[AgentPlan] = Field(description="A list of agent plans to execute in order")

agent_registry = [
    {
        "name": "analysis_agent",
        "tools": [{"tool": "rag_tool", "args": {"query": "<query>"}}, {"tool": "web_search_tool", "args": {"query": "<query>"}}],
        "description": "For analyzing, comparing, or synthesizing information."
    },
    {
        "name": "summarization_agent",
        "tools": [{"tool": "rag_tool", "args": {"query": "<query>"}}, {"tool": "web_search_tool", "args": {"query": "<query>"}}],
        "description": "For summarizing documents or chat history."
    }
]

@traceable(name="planning_agent")
async def planning_agent(state: AgentState) -> AgentState:
    logger.info("---Entering planning_agent---")
    try:
        user_input = state.processing.user_input or ""
        agent_name = "planning_agent"
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
            model_name="meta-llama/llama-4-scout-17b-16e-instruct",
            max_tokens=512,
            streaming=True,
            api_key=groq_api_key
        )
        agent_registry_str_lines = []
        for agent in agent_registry:
            tools_str = ', '.join([f"{tool['tool']} (args: {tool['args']})" for tool in agent['tools']])
            agent_registry_str_lines.append(f"- {agent['name']}: {agent['description']} (tools: {tools_str})")
        agent_registry_str = "\n".join(agent_registry_str_lines)

        
        # Inline the system message directly instead of using {prompt_template} as a variable
        system_message = r'''
        You are a planning agent for a multi-agent system. Your job is to break down the user's request into a sequence of agent and tool calls (a plan) by identifying the user's intent.
        You have access to the following agents and tools:
        {agent_registry_str}
        For each user request, provide a plan (list of steps/agent plans with tools which ever is needed) tailored to the specific use case. Each step must have: 
        - 'agent': the agent name
        - 'tools': a list of tool calls if needed, otherwise empty list each with: 
            - 'tool': the tool name (can be empty if not needed)
            - 'args': a dictionary of arguments for the tool (can be empty if not needed)
        
        Return ONLY a JSON object with a 'plan' key containing the list of agent steps.

        Important to remember: Internal documents refer to thesis projects and research papers maintained by the institute.
        '''
        if fallback:
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_message + "\nYou are being rerun due to a previous error.\nError: {{error}}\nFallback Solution: {{fallback_solution}}\nUse the fallback solution and the original input to answer, and ensure you do not repeat the previous error.\nDo not mention the error or the fallback solution in your response. \nCurrent date and time: {current_time_date_str}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "User input: {input}\n\nReturn only the JSON object:")
            ])
        else:
            prompt = ChatPromptTemplate.from_messages([
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

        chain_input = {
            "input": user_input,
            "chat_history": _chat_history,
            "current_time_date_str": datetime.now().isoformat(),
            "agent_registry_str": agent_registry_str
        }

        parser = JsonOutputParser(pydantic_object=PlanOutput)
        chain = prompt | llm | parser

        plan_result = await chain.ainvoke(chain_input)
        logger.info(f"Planning agent output: {plan_result}")
        plan = plan_result.get("plan", [])
        state.processing.plan = plan
        state.processing.current_agent = agent_name
        if not state.response.response_metadata:
            state.response.response_metadata = {}
        state.response.response_metadata.update({
            "agent_type": agent_name,
            "processing_time": datetime.now().isoformat()
        })
        logger.info(f"Planning agent generated plan: {state.processing.plan}")
        logger.info("---End of planning_agent---")
        state.processing.executed_steps.append("planning_agent")
        return state
    except Exception as e:
        state.error.error = f"Planning error: {str(e)}"
        state.processing.current_agent = "planning_agent"
        logger.error(f"Planning agent error: {str(e)}")
        logger.info("---End of planning_agent---")
        return state 