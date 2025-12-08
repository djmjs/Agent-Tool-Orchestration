from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .llm import LLM
from ..utils.logger import log_info

class Router:
    def __init__(self, llm_service: LLM):
        self.llm = llm_service.llm
        
        # We ask for a single word response to make parsing robust for local models
        self.system_prompt = """You are an intelligent router for an AI assistant.
Your goal is to decide if a user's query requires looking up external information in a vector database (RAG) or if it can be answered with your general knowledge.

The Vector Database contains specialized information about:
- Large Language Models (LLMs)
- Reinforcement Learning (RL)
- Embodied Intelligence
- AI Agents and Academic Papers

Decide based on the following criteria:
- If the query is about the topics above, specific papers, or technical details -> Respond with "DATABASE"
- If the query is about general knowledge (e.g., "Who is Elon Musk?", "Python code", "Greetings") -> Respond with "GENERAL"

Return ONLY the word "DATABASE" or "GENERAL". Do not add punctuation or explanation.
"""
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{question}"),
        ])
        self.output_parser = StrOutputParser()

    async def route(self, question: str) -> str:
        """
        Returns 'vector_db' or 'general_chat'
        """
        if not self.llm:
             return "vector_db" # Default

        try:
            chain = self.prompt | self.llm | self.output_parser
            result = await chain.ainvoke({"question": question})
            cleaned_result = result.strip().upper()
            
            if "DATABASE" in cleaned_result:
                return "vector_db"
            else:
                return "general_chat"
                
        except Exception as e:
            log_info(f"Router error: {e}, defaulting to vector_db")
            return "vector_db"
