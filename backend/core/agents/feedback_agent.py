import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from backend.core.state import AgentState
from scripts.log_config import get_logger
import os
from datetime import datetime
from langsmith import traceable

logger = get_logger(__name__)

groq_api_key = os.environ.get('GROQ_API_KEY')

@traceable(name="feedback_agent")
async def feedback_agent(state: AgentState) -> AgentState:
    logger.info("---Entering feedback_agent---")
    try:
        user_input = state.processing.user_input or ""
        answer = state.response.response or ""
        
        fallback = False
        fallback_solution = None
        error_message = None
        if (
            state.processing.current_agent == "fallback_agent"
            and state.response.response_metadata.get("rerun_agent") == "feedback_agent"
            and state.response.response_metadata.get("fallback")
        ):
            fallback_solution = state.response.response_metadata["fallback"]
            error_message = state.error.error or ""
            fallback = True

        llm = ChatGroq(
            temperature=0.1,
            model_name="meta-llama/llama-4-scout-17b-16e-instruct",
            max_tokens=256,
            streaming=True,
            api_key=groq_api_key
        )

        prompt_template = r"""
        You are a feedback agent. The current date and time is {current_time_date_str}.
        Your job is to decide, based on the user's input and the answer provided by the previous agent, whether the workflow should proceed to the next step or call the planning agent for further tool invocation.
        Return ONLY a valid JSON object with this key:
        - 'proceed': true if the answer is sufficient and the workflow should continue, false if the planning agent should be called to invoke other tools/agents.
        """
        if fallback:
            prompt= ChatPromptTemplate.from_messages([
            ("system", """{{prompt_template}}
            You are being rerun due to a previous error.
            Error: {{error}}
            Fallback Solution: {{fallback_solution}}
            Use the fallback solution and the original input to answer, and ensure you do not repeat the previous error.
            Do not mention the error or the fallback solution in your response.
            """),
            ("human", "User input: {input}\n---\nAnswer: {answer}\n---Return only the JSON object:")
            ])
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", prompt_template),
                ("human", "User input: {input}\n---\nAnswer: {answer}\n---Return only the JSON object:")
                ])

        chain = prompt | llm | JsonOutputParser()
        chain_input = {
            "current_time_date_str": datetime.now().isoformat(),
            "input": user_input,
            "answer": answer
        }

        logger.info(f"Feedback agent input: {chain_input}")
        feedback_result = await chain.ainvoke(chain_input)
        logger.info(f"Feedback agent result: {feedback_result}")
        state.response.response_metadata["feedback"] = feedback_result
        # Increment replan_attempts if replan is triggered
        if isinstance(feedback_result, dict) and feedback_result.get("proceed") is False:
            if hasattr(state.processing, "replan_attempts"):
                state.processing.replan_attempts += 1
        state.processing.current_agent = "feedback_agent"
        state.processing.executed_steps.append("feedback_agent")
        logger.info("---End of feedback_agent---")
        return state
    except Exception as e:
        state.error.error = f"Feedback agent error: {str(e)}"
        state.processing.current_agent = "feedback_agent"
        logger.error(f"Feedback agent error: {str(e)}")
        logger.info("---End of feedback_agent---")
        return state 