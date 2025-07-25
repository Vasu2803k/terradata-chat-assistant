import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from datetime import datetime
from typing import Dict, Any, List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langsmith import traceable
from flashrank import Ranker
from langchain_chroma import Chroma
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

from backend.core.vectorstore import vectorstore

logger = get_logger(__name__)
groq_api_key = os.environ.get('GROQ_API_KEY')

async def rag_tool(*args, **kwargs):
    logger.info("---Entering rag_tool---")
    """
    Returns the context retriever and stuff documents chain for explicit doc retrieval and LLM invocation.
    """
    llm = ChatGroq(
        temperature=0.2,
        model_name="meta-llama/llama-4-scout-17b-16e-instruct",
        max_tokens=1024,
        streaming=True,
        api_key=groq_api_key
    )
    multiquery_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        Generate alternative phrasings of the user's question. Break down the question into smaller parts and generate atleast one alternative phrasings for each part. Provide a total of only 5 alternative phrasings without part numbers, labels, explanation or any other text. Do not mention 'alternative phrasings' keyword in the response. 
        Do not add numbering, labels, explanation or any other text.
        """),
        ("human", "Original question: {question}")
    ])
    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
    reranker = FlashrankRerank(client=ranker, top_n=5, model="ms-marco-MiniLM-L-12-v2")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
    )
    context_retriever = MultiQueryRetriever.from_llm(
        retriever=compression_retriever,
        llm=llm,
        prompt=multiquery_prompt
    )

    retrieved_docs = await context_retriever.aget_relevant_documents(kwargs.get('query'))
    top_docs = retrieved_docs[:5]  # Flashrank already returns top 5, but this is explicit
    logger.info(f"RAG tool: Retrieved {len(top_docs)} documents.")
    logger.info("---End of rag_tool---")
    return top_docs


