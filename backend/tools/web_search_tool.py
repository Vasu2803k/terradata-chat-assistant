import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import asyncio
from scripts.log_config import get_logger

logger = get_logger(__name__)

# Import ddgs (new DuckDuckGo search package)
from ddgs import DDGS

async def web_search_tool(*args, **kwargs):
    logger.info("---Entering web_search_tool---")
    """
    Performs a DuckDuckGo web search using ddgs and returns the top N results as a list of strings.
    Input should be a search query string.
    """
    query = kwargs.get('query') if 'query' in kwargs else (args[0] if args else None)
    max_results = kwargs.get('max_results', 10)
    if not isinstance(query, str) or not query.strip():
        logger.error("Query must be a non-empty string.")
        return []
    if not isinstance(max_results, int) or max_results <= 0:
        logger.warning(f"Invalid max_results '{max_results}', defaulting to 10.")
        max_results = 10
    logger.info(f"Performing web search for: '{query}' with top {max_results} results.")
    try:
        # ddgs is blocking, so run in executor
        loop = asyncio.get_running_loop()
        def search():
            with DDGS() as ddgs:
                results = []
                for i, r in enumerate(ddgs.text(query)):
                    if i >= max_results:
                        break
                    # Compose a string with title and url
                    title = r.get('title', '').strip()
                    url = r.get('href', '').strip()
                    snippet = r.get('body', '').strip()
                    result_str = f"{title} - {url}\n{snippet}" if snippet else f"{title} - {url}"
                    results.append(result_str)
                return results
        results = await loop.run_in_executor(None, search)
        logger.info(f"Web search returned {len(results)} results.")
        logger.info("---End of web_search_tool---")
        return results
    except Exception as e:
        logger.error(f"Exception during web search: {e}", exc_info=True)
        return []