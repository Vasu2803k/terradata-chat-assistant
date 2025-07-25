import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from backend.core.state import AgentState, MessageRole
from scripts.log_config import get_logger
import os
from datetime import datetime
from langsmith import traceable

logger = get_logger(__name__)

groq_api_key = os.environ.get('GROQ_API_KEY')

class FallbackOutput(BaseModel):
    rerun_agent: str = Field(description="The agent to rerun, i.e., the previous agent that failed.")
    solution: str = Field(description="A concise answer to ensure no error this time.")

@traceable(name="fallback_agent")
async def fallback_agent(state: AgentState) -> AgentState:
    logger.info("---Entering fallback_agent---")
    try:
        error = state.error.error or "Unknown error"
        rerun_agent = state.processing.current_agent or "unknown_agent"
        fallback = False
        fallback_solution = None
        error_message = None
        if (
            state.processing.current_agent == "fallback_agent"
            and state.response.response_metadata.get("rerun_agent") == "fallback_agent"
            and state.response.response_metadata.get("fallback")
        ):
            fallback_solution = state.response.response_metadata["fallback"]
            error_message = state.error.error or ""
            fallback = True

        llm = ChatGroq(
            temperature=0.1,
            model_name="llama-3.1-8b-instant",
            max_tokens=256,
            streaming=True,
            api_key=groq_api_key
        )
        parser = JsonOutputParser(pydantic_object=FallbackOutput)

        prompt_template = r"""
        You are a fallback agent in a multi-agent system. The current date and time is {current_time_date_str}.
        You are given the error message and the name of the agent that failed (rerun_agent).
        Your job is to provide a JSON object with:
        - 'rerun_agent': the agent to rerun (the previous agent that failed)
        - 'solution': provide a concise, actionable answer or fix to ensure no error this time (e.g., clarify input, suggest a fix, or rephrase the request)
        """
        if fallback:
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"""{{prompt_template}}
                You are being rerun due to a previous error.
                Error: {{error}}
                Fallback Solution: {{fallback_solution}}
                Use the fallback solution and the original input to answer, and ensure you do not repeat the previous error.
                Do not mention the error or the fallback solution in your response.
                - 'solution': Mention the error first and then provide a concise, actionable answer or fix to ensure no error this time (e.g., clarify input, suggest a fix, or rephrase the request)
                """),
                ("human", "Error: {error}\nPrevious agent: {rerun_agent}\nReturn only the JSON object with the following keys: 'rerun_agent' and 'solution':")
            ])
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", prompt_template),
                ("human", "Error: {error}\nPrevious agent: {rerun_agent}\nReturn only the JSON object with the following keys: 'rerun_agent' and 'solution':")
            ])
        chain = prompt | llm | parser
        chain_input = {
            "current_time_date_str": datetime.now().isoformat(),
            "error": error,
            "rerun_agent": rerun_agent,
            "fallback_solution": fallback_solution if fallback else None
        }
        logger.info(f"Fallback agent input: {chain_input}")
        result = await chain.ainvoke(chain_input)
        logger.info(f"Fallback agent result: {result}")
        # we will get a json object which has rerun_agent and solution
        state.response.response = "No response from the previous agent. Please try again."
        state.response.response_metadata["rerun_agent"] = result["rerun_agent"]
        state.response.response_metadata["fallback"] = result["solution"]
        state.processing.current_agent = "fallback_agent"
        state.response.response_metadata.update({
            "agent_type": "fallback_agent",
            "processing_time": datetime.now().isoformat(),
            "fallback": result
        })
        state.processing.executed_steps.append("fallback_agent")
        logger.info("---End of fallback_agent---")
        return state
    except Exception as e:
        logger.error(f"Error in fallback agent: {str(e)}")
        state.error.error = f"Fallback agent error: {str(e)}"
        state.response.response = "I'm sorry, I encountered an error while processing your request. Please try again."
        state.processing.current_agent = "fallback_agent"
        logger.info(f"Returning from fallback_agent with type: {type(state)} (error path)")
        logger.info("---End of fallback_agent---")
        return state 
