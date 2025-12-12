from typing import Dict, Any, List
from langchain_core.output_parsers import StrOutputParser

from ..components.short_term_memory import ShortTermMemory
from ..components.retriever import Retriever
from ..components.reranker import ReRanker
from ..components.prompt_gen import PromptGenerator
from ..components.llm import LLM
from ..components.query_reformulator import QueryReformulator
from ..utils.fallback import FallbackHandler
from ..utils.logger import log_info, log_error

class Orchestrator:
    def __init__(self):
        self.llm = LLM()
        self.stm = ShortTermMemory(llm=self.llm.llm)
        self.retriever = Retriever()
        self.reranker = ReRanker()
        self.prompt_gen = PromptGenerator()
        self.reformulator = QueryReformulator(self.llm)
        self.fallback = FallbackHandler()

    async def process_query(self, query: str) -> Dict[str, Any]:
        try:
            log_info(f"Processing query: {query}")
            
            # 0. Reformulate Query (History Aware)
            history = self.stm.get_chat_history()
            if history:
                reformulated_query = await self.reformulator.reformulate(query, history)
                log_info(f"Reformulated query: {reformulated_query}")
            else:
                reformulated_query = query

            # 1. Retrieve (using reformulated query)
            docs = await self.retriever.retrieve(reformulated_query)
            if not docs:
                return {
                    "answer": self.fallback.handle_no_results(),
                    "sources": [],
                    "user_prompt_history": [],
                    "chat_answers_history": []
                }

            # 2. Re-Rank (using reformulated query)
            ranked_docs = self.reranker.rerank(reformulated_query, docs)

            # 3. Prepare Context
            context = "\n\n".join([d.page_content for d in ranked_docs])
            
            # 4. Generate Prompt
            history = self.stm.get_chat_history()
            messages = self.prompt_gen.generate_prompt(context, query, history)

            # 5. LLM Generation
            response_msg = await self.llm.generate_response(messages)
            answer = response_msg.content

            # 6. Format Response (with sources)
            sources_list = [
                f"{doc.metadata.get('title', 'Unknown')} ({doc.metadata.get('url', '#')})"
                for doc in ranked_docs
            ]
            sources_str = "\nSources:\n" + "\n".join(sources_list) if sources_list else ""
            formatted_response = f"{answer}\n\n{sources_str}"

            # 7. Update Memory
            self.stm.add_interaction(query, answer, formatted_response)

            # 8. Return Full History for UI
            ui_history = self.stm.get_ui_history()
            
            # Also return the latest answer/sources for API compatibility if needed
            sources_objs = [
                {
                    "title": doc.metadata.get('title', 'Unknown'),
                    "url": doc.metadata.get('url', '#'),
                    "content_preview": doc.page_content[:200]
                }
                for doc in ranked_docs
            ]

            return {
                "answer": answer,
                "sources": sources_objs,
                "user_prompt_history": ui_history["user_prompt_history"],
                "chat_answers_history": ui_history["chat_answers_history"]
            }

        except Exception as e:
            log_error(f"Orchestration failed: {e}")
            return {
                "answer": self.fallback.handle_error(e),
                "sources": []
            }
