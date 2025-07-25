import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Environment variables
from dotenv import load_dotenv
load_dotenv()

import os
import logging
from datetime import datetime
from langgraph.graph import StateGraph, END

# Local imports
from backend.core.state import (
    AgentState, StateManager, UserState, Chat, MessageRole, ProcessingState, RetrievalState, ResponseState, ErrorState
)
from backend.core.agents.router_agent import router_agent
from backend.core.agents.conversation_agent import conversation_agent
from backend.core.agents.content_moderation_agent import content_moderation_agent
from backend.core.agents.planning_agent import planning_agent
from backend.core.agents.feedback_agent import feedback_agent
from backend.core.agents.analysis_agent import analysis_agent
from backend.core.agents.summarization_agent import summarization_agent
from backend.core.agents.fallback_agent import fallback_agent
from scripts.log_config import get_logger
import traceback

logger = get_logger(__name__)

# Check if LangSmith API key is set
LANGSMITH_API_KEY = os.environ.get("LANGCHAIN_API_KEY")
LANGSMITH_ENABLED = LANGSMITH_API_KEY is not None and LANGSMITH_API_KEY != ""

# Initialize LangSmith client if API key is available
if LANGSMITH_ENABLED:
    try:
        from langsmith import Client
        from langsmith.run_helpers import traceable
        langsmith_client = Client(api_key=LANGSMITH_API_KEY)
        logger.info("LangSmith client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize LangSmith client: {str(e)}")
        logger.error(traceback.format_exc())
        LANGSMITH_ENABLED = False
        # Define a no-op traceable decorator when LangSmith is not available
        def traceable(*args, **kwargs):
            def decorator(func):
                return func
            return decorator if len(args) == 0 else decorator(args[0])
else:
    logger.info("LangSmith API key not found. Running without LangSmith tracking.")
    # Define a no-op traceable decorator when LangSmith is not available
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if len(args) == 0 else decorator(args[0])

class TerradataAssignmentOrchestrator:
    """
    Main orchestrator using LangGraph for routing and state management.
    """
    def __init__(self):
        logger.info("---Entering __init__---")
        self.state_manager = StateManager()
        self.graph = self._create_workflow_graph()
        logger.info("TerradataAssignmentOrchestrator initialized successfully.")
        logger.info("---End of __init__---")

    def _create_workflow_graph(self) -> StateGraph:
        logger.info("---Entering _create_workflow_graph---")
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("router_agent", router_agent)
        workflow.add_node("conversation_agent", conversation_agent)
        workflow.add_node("planning_agent", planning_agent)
        workflow.add_node("dispatcher_agent", self._dispatcher_agent)
        workflow.add_node("analysis_agent", analysis_agent)
        workflow.add_node("summarization_agent", summarization_agent)
        workflow.add_node("content_moderation_agent", content_moderation_agent)
        workflow.add_node("feedback_agent", feedback_agent)
        workflow.add_node("fallback_agent", fallback_agent)
        workflow.add_node("final_response", self._final_response)

        # Set entry point
        workflow.set_entry_point("router_agent")
        
        # router -> (conversation_agent, planning_agent, content_moderation_agent)
        workflow.add_conditional_edges(
            "router_agent",
            lambda state: state.processing.route_decision,  # route_to_agent returns a string
            {
                "conversation_agent": "conversation_agent",
                "planning_agent": "planning_agent",
                "content_moderation_agent": "content_moderation_agent",
                "fallback_agent": "fallback_agent"
            }
        )
        # planning_agent -> dispatcher_agent
        workflow.add_edge("planning_agent", "dispatcher_agent")
        # dispatcher_agent -> (analysis_agent or summarization_agent)
        workflow.add_conditional_edges(
            "dispatcher_agent",
            self._dispatcher_decision,
            {
                "analysis_agent": "analysis_agent",
                "summarization_agent": "summarization_agent",
                "final_response": "final_response"  # fallback if no agent in plan
            }
        )
        workflow.add_edge("analysis_agent", "feedback_agent")
        workflow.add_edge("summarization_agent", "feedback_agent")
        # feedback_agent -> (proceed: final_response, replan: planning_agent, fallback: fallback_agent)
        workflow.add_conditional_edges(
            "feedback_agent",
            self._feedback_decision,
            {
                "proceed": "final_response",
                "replan": "planning_agent",
                "fallback": "fallback_agent"
            }
        )
        # conversation_agent and content_moderation_agent -> final_response
        workflow.add_edge("conversation_agent", "final_response")
        workflow.add_edge("content_moderation_agent", "final_response")
        # Add conditional edge: fallback_agent -> rerun_agent (from response_metadata["rerun_agent"]), else final_response
        workflow.add_conditional_edges(
            "fallback_agent",
            lambda state: state.response.response_metadata.get("rerun_agent", "final_response"),
            {
                "conversation_agent": "conversation_agent",
                "planning_agent": "planning_agent",
                "content_moderation_agent": "content_moderation_agent",
                "feedback_agent": "feedback_agent",
                "router_agent": "router_agent",
                "final_response": "final_response"
            }
        )
        logger.info("Workflow graph constructed: router -> (conversation_agent, planning_agent, content_moderation_agent) ...")
        logger.info("---End of _create_workflow_graph---")
        return workflow.compile()

    async def _dispatcher_agent(self, state: AgentState) -> AgentState:
        logger.info("---Entering dispatcher_agent---")
        # This node does nothing but is used for routing
        logger.info(f"Dispatcher received plan: {getattr(state.processing, 'plan', [])}")
        logger.info("---End of dispatcher_agent---")
        return state

    async def _dispatcher_decision(self, state: AgentState) -> str:
        logger.info("---Entering _dispatcher_decision---")
        plan = getattr(state.processing, "plan", [])
        agent_names = [step.get("agent") for step in plan if step.get("agent") in ["analysis_agent", "summarization_agent"]]
        logger.info(f"Dispatcher decision, agents in plan: {agent_names}")
        # Prioritize analysis_agent if present, else summarization_agent, else final_response
        if "analysis_agent" in agent_names:
            logger.info("---End of _dispatcher_decision---")
            return "analysis_agent"
        elif "summarization_agent" in agent_names:
            logger.info("---End of _dispatcher_decision---")
            return "summarization_agent"
        else:
            logger.info("---End of _dispatcher_decision---")
            return "final_response"

    async def _feedback_decision(self, state: AgentState) -> str:
        logger.info("---Entering _feedback_decision---")
        feedback = state.response.response_metadata.get("feedback", {})
        # Max retries logic
        max_retries = 2
        if hasattr(state.processing, "replan_attempts") and state.processing.replan_attempts >= max_retries:
            logger.warning(f"Max replan attempts ({max_retries}) reached. Breaking loop.")
            return "proceed"
        if isinstance(feedback, dict):
            if feedback.get("proceed", False):
                logger.info("---End of _feedback_decision---")
                return "proceed"
            else:
                logger.info("---End of _feedback_decision---")
                return "replan"
        logger.info("---End of _feedback_decision---")
        return "proceed"

    async def process_user_input(self, user_id: str, chat_id: str, user_input: str) -> dict:
        logger.info("---Entering process_user_input---")
        try:
            user_state = self.state_manager.get_user_state(user_id)
            chat = user_state.get_chat(chat_id)
            if not chat:
                chat = user_state.new_chat()
            # Add user message to chat
            chat.add_message(MessageRole.USER, user_input)
            # Prepare AgentState for workflow
            agent_state = AgentState(
                user_id=user_id,
                chat_id=chat.chat_id,
                chat_history=chat.get_context_window(),
                processing=ProcessingState(user_input=user_input, is_processing=True),
                retrieval=RetrievalState(chat_id=chat.chat_id, retrieved_documents={}),
                response=ResponseState(),
                error=ErrorState(),
                long_term_history_for_context=user_state.long_term_history
            )
            config = {"configurable": {"thread_id": user_id}}
            logger.info(f"Starting workflow with state: {type(agent_state)}")
            result = await self.graph.ainvoke(agent_state, config)
            logger.info(f"Workflow result type: {type(result)}")
            # If result is a dict, convert to AgentState
            if isinstance(result, dict):
                result = AgentState(**result)
                logger.info(f"Modified Workflow result type: {type(result)}")
            # Save assistant message to chat if response exists
            if result.response and result.response.response:
                resp = result.response.response
                # If the response is a dict and has an 'answer' key, use that
                if isinstance(resp, dict):
                    if 'answer' in resp and isinstance(resp['answer'], str):
                        chat.add_message(MessageRole.ASSISTANT, resp['answer'])
                    else:
                        chat.add_message(MessageRole.ASSISTANT, str(resp))
                else:
                    chat.add_message(MessageRole.ASSISTANT, resp)
            # Summarize chat if needed
            user_state.summarize_chat_if_needed(chat.chat_id)

            logger.info("---End of process_user_input---")
            return {
                "response": result.response.response,
                "agent_used": result.processing.current_agent,
                "route_decision": result.processing.route_decision,
                "confidence_score": result.processing.confidence_score,
                "metadata": {**result.response.response_metadata, "executed_steps": result.processing.executed_steps},
                "error": result.error.error
            }
        except Exception as e:
            logger.error(f"Error processing user input: {str(e)}")
            logger.info("---End of process_user_input---")
            return {
                "response": "I'm sorry, I encountered an error while processing your request. Please try again.",
                "error": str(e),
                "agent_used": "content_moderation_agent"
            }

    async def _final_response(self, state: AgentState) -> AgentState:
        logger.info("---Entering _final_response---")
        try:
            if state.response and state.response.response_metadata is not None:
                state.response.response_metadata.update({
                    "final_processing_time": datetime.now().isoformat(),
                    "workflow_completed": True
                })
            logger.info("---End of _final_response---")
            return state
        except Exception as e:
            logger.error(f"Error in final response: {str(e)}")
            logger.info("---End of _final_response---")
            return state

    async def get_chat_history(self, user_id: str, chat_id: str) -> list:
        logger.info("---Entering get_chat_history---")
        try:
            user_state = self.state_manager.get_user_state(user_id)
            chat = user_state.get_chat(chat_id)
            if chat:
                logger.info("---End of get_chat_history---")
                return [msg.dict() for msg in chat.messages]
            logger.info("---End of get_chat_history---")
            return []
        except Exception as e:
            logger.error(f"Error getting chat history: {str(e)}")
            logger.info("---End of get_chat_history---")
            return []

    async def clear_chat(self, user_id: str, chat_id: str) -> bool:
        logger.info("---Entering clear_chat---")
        try:
            user_state = self.state_manager.get_user_state(user_id)
            if chat_id in user_state.chats:
                del user_state.chats[chat_id]
            logger.info("---End of clear_chat---")
            return True
        except Exception as e:
            logger.error(f"Error clearing chat: {str(e)}")
            logger.info("---End of clear_chat---")
            return False

    async def get_long_term_history(self, user_id: str) -> dict:
        logger.info("---Entering get_long_term_history---")
        try:
            user_state = self.state_manager.get_user_state(user_id)
            logger.info("---End of get_long_term_history---")
            return user_state.long_term_history.summaries
        except Exception as e:
            logger.error(f"Error getting long term history: {str(e)}")
            logger.info("---End of get_long_term_history---")
            return {}

# Create a singleton instance
orchestrator = TerradataAssignmentOrchestrator()

# Convenience functions for easy access
async def process_message(user_id: str, chat_id: str, message: str) -> dict:
    return await orchestrator.process_user_input(user_id, chat_id, message)

# Streaming version for FastAPI StreamingResponse
async def process_message_stream(user_id: str, chat_id: str, message: str):
    """
    Yields reasoning steps and final response as JSON-serializable dicts for streaming.
    """
    import asyncio
    # Call the original process_user_input to get the final result
    result = await orchestrator.process_user_input(user_id, chat_id, message)
    # Get executed steps from the result's metadata if available
    executed_steps = []
    if result.get("metadata") and "executed_steps" in result["metadata"]:
        executed_steps = result["metadata"]["executed_steps"]
    elif result.get("agent_used"):
        executed_steps = [result["agent_used"]]
    # Get tool/step responses if available
    step_responses = []
    if result.get("metadata") and "tool_responses" in result["metadata"]:
        step_responses = result["metadata"]["tool_responses"]
    # Yield each executed step as a reasoning step, with response if available
    for idx, step in enumerate(executed_steps):
        step_response = None
        # Try to get a response for this step from tool_responses
        if step_responses and idx < len(step_responses):
            step_response = step_responses[idx].get("response")
        # Fallback: for the last step, use the main response
        if step_response is None and idx == len(executed_steps) - 1:
            step_response = result.get("response")
        yield {"type": "reasoning", "step": step, "status": "executed", "response": step_response}
        await asyncio.sleep(0.1)  # Small delay for streaming effect
    # Yield all reasoning fields for the last step (for details)
    yield {"type": "reasoning", "step": result.get("agent_used"), "route_decision": result.get("route_decision"), "agent_used": result.get("agent_used"), "confidence_score": result.get("confidence_score"), "metadata": result.get("metadata"), "error": result.get("error")}
    # Final response
    yield {"type": "final", "response": result.get("response", "(No response)")}

async def get_history(user_id: str, chat_id: str) -> list:
    return await orchestrator.get_chat_history(user_id, chat_id)

async def clear_chat(user_id: str, chat_id: str) -> bool:
    return await orchestrator.clear_chat(user_id, chat_id)

async def get_long_term_history(user_id: str) -> dict:
    return await orchestrator.get_long_term_history(user_id)


    