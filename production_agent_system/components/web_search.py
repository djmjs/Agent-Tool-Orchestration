import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from ..utils.logger import log_info, log_error

class WebSearch:
    def __init__(self):
        self.tool = None
        self._setup()

    def _setup(self):
        """Initialize the Tavily search tool."""
        try:
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                log_info("TAVILY_API_KEY not found. Web search will be disabled.")
                return

            self.tool = TavilySearchResults(max_results=3)
            log_info("WebSearch initialized successfully.")
        except Exception as e:
            log_error(f"Failed to setup WebSearch: {e}")

    async def search(self, query: str, config=None):
        """
        Search the web and return a list of Document objects.
        """
        if not self.tool:
            log_info("WebSearch tool not available.")
            return []

        log_info(f"Searching web for: {query}")
        try:
            # TavilySearchResults returns [{'url': '...', 'content': '...'}]
            results = await self.tool.ainvoke(query, config=config)
            
            documents = []
            for res in results:
                doc = Document(
                    page_content=res.get("content", ""),
                    metadata={"source": res.get("url", ""), "title": "Web Search Result"}
                )
                documents.append(doc)
            
            return documents
        except Exception as e:
            log_error(f"Web search failed: {e}")
            return []
