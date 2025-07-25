import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from datetime import datetime
from typing import Dict, Any, List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from langchain.chains.combine_documents import create_stuff_documents_chain
from langsmith import traceable
from backend.core.state import AgentState, MessageRole
from backend.tools.text_embedding_tool import TextEmbedding
from scripts.log_config import get_logger
import os
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dataclasses import dataclass, field
from typing import List
from langchain.schema import Document

from backend.tools.rag_tool import rag_tool

logger = get_logger(__name__)
groq_api_key = os.environ.get('GROQ_API_KEY')


@traceable(name="rag_agent_invoke")
async def rag_agent(state: AgentState) -> AgentState:
    logger.info("---Entering rag_agent_invoke---")
    try:
        user_input = state.processing.user_input or ""
        chat_history = state.chat_history
        if not user_input:
            state.error.error = "No user input provided"
            state.response.response = "No input to process."
            logger.warning("RAG agent: No user input provided.")
            logger.info("---End of rag_agent_invoke---")
            return state
        
        fallback = False
        fallback_solution = None
        error_message = None
        agent_name = "rag_agent"
        if (
            state.processing.current_agent == "fallback_agent"
            and state.response.response_metadata.get("rerun_agent") == agent_name
            and state.response.response_metadata.get("fallback")
        ):
            fallback_solution = state.response.response_metadata["fallback"]
            error_message = state.error.error or ""
            fallback = True

        llm = ChatGroq(
            temperature=0.2,
            model_name="meta-llama/llama-4-scout-17b-16e-instruct",
            max_tokens=1024,
            streaming=True,
            api_key=groq_api_key)

        context_retriever = await rag_tool()

        # Inline the system message directly instead of using {prompt_template} as a variable
        system_message = r"""
        You are a RAG agent. 
        You are provided with the following context documents:
        {context}

        Current date and time: {current_time_date_str}

        Carefully answer the user's question by using the provided documents, being concise, factual, and clear. Organize data in tables if needed.

        If the user's question is not related to the context documents, say \"I'm sorry, I don't have information on that topic.\"
        """
        if fallback:
            stuff_prompt = ChatPromptTemplate.from_messages([
                ("system", system_message + "\nYou are being rerun due to a previous error.\nError: {{error}}\nFallback Solution: {{fallback_solution}}\nUse the fallback solution and the original input to answer, and ensure you do not repeat the previous error.\nDo not mention the error or the fallback solution in your response."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}")
            ])
        else:
            stuff_prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}")
            ])
        
        chat_history = state.chat_history[-10:]
        _chat_history = []
        for msg in chat_history:
            if msg.role == MessageRole.USER:
                _chat_history.append(("user", msg.content))
            elif msg.role == MessageRole.ASSISTANT:
                _chat_history.append(("assistant", msg.content))

        
        stuff_documents_chain = create_stuff_documents_chain(llm, stuff_prompt)

        # Explicitly retrieve top documents
        retrieved_docs = await context_retriever.aget_relevant_documents(user_input)
        top_docs = retrieved_docs[:5]  # Flashrank already returns top 5, but this is explicit

        chain_input = {
            "input": user_input,
            "chat_history": _chat_history,
            "current_time_date_str": datetime.now().isoformat(),
            "context": top_docs
        }

        response = ""

        async for chunk in stuff_documents_chain.astream(chain_input):
            response += chunk

        logger.info(f"Response: {response}")

        # Log the number of retrieved documents
        logger.info(f"Number of retrieved documents: {len(top_docs)}")
        # Save response and docs into state
        state.response.response = response
        # Store retrieved docs in retrieval state using chat_id as key
        state.retrieval.retrieved_documents[state.chat_id] = top_docs

        state.processing.current_agent = "rag_agent"

        if not state.response.response_metadata:
            state.response.response_metadata = {}

        state.response.response_metadata.update({
            "agent_type": "rag_agent",
            "processing_time": datetime.now().isoformat()
        })

        logger.info(f"Returning from rag_agent with type: {type(state)}")
        logger.info("---End of rag_agent_invoke---")
        return state
    
    except Exception as e:
        logger.error(f"Error in RAG agent: {str(e)}")
        state.error.error = f"RAG error: {str(e)}"
        state.response.response = "Error in RAG agent."
        state.processing.current_agent = "rag_agent"
        logger.info(f"Returning from rag_agent with type: {type(state)} (error path)")
        logger.info("---End of rag_agent_invoke---")
        return state 