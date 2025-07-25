import logging
from backend.core.state import AgentState
from scripts.log_config import get_logger
from backend.tools.rag_tool import rag_tool
from backend.tools.web_search_tool import web_search_tool
import asyncio
from langsmith import traceable

logger = get_logger(__name__)

# Registry mapping tool names to their async functions
TOOL_REGISTRY = {
    "rag_tool": rag_tool,
    "web_search_tool": web_search_tool,
}
@traceable(name="executor_tool")
async def executor_tool(state: AgentState) -> AgentState:
    logger.info("---Entering executor_tool---")
    plan = getattr(state.processing, "plan", [])
    tool_responses = []
    for idx, step in enumerate(plan):
        tool_name = step.get("tool")
        args = step.get("args", {})
        tool_func = TOOL_REGISTRY.get(tool_name)
        if not tool_func:
            logger.warning(f"Executor tool: Unknown tool '{tool_name}' in plan. Skipping.")
            continue
        logger.info(f"Executor tool: Executing {tool_name} with args: {args}")
        try:
            # If the tool expects state and args, pass both
            if asyncio.iscoroutinefunction(tool_func):
                response = await tool_func(state, **args) if args else await tool_func(state)
            else:
                response = tool_func(state, **args) if args else tool_func(state)
            # If the tool returns a state, try to extract response
            tool_response = None
            if hasattr(state.response, "response") and state.response.response:
                tool_response = state.response.response
            elif isinstance(response, dict) and "response" in response:
                tool_response = response["response"]
            else:
                tool_response = str(response)
            tool_responses.append({
                "tool": tool_name,
                "args": args,
                "response": tool_response
            })
            logger.info(f"Executor tool: {tool_name} response: {tool_response}")
        except Exception as e:
            logger.error(f"Executor tool: Error executing {tool_name}: {str(e)}")
            tool_responses.append({
                "tool": tool_name,
                "args": args,
                "response": f"Error: {str(e)}"
            })
    state.response.tool_responses = tool_responses
    logger.info(f"Executor tool finished executing plan. Tool responses: {tool_responses}")
    logger.info("---End of executor_tool---")
    return state 