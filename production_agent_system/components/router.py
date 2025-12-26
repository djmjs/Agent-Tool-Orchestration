from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .llm import LLM
from ..utils.logger import log_info

class Router:
    def __init__(self, llm_service: LLM):
        self.llm = llm_service.llm
        
        # We ask for a single word response to make parsing robust for local models
        self.system_prompt = """You are an intelligent router for an AI assistant.
Your goal is to classify the user's query into one of four categories:

1. "DATABASE": Use this if the query is about:
   - Large Language Models (LLMs)
   - Reinforcement Learning (RL)
   - Embodied Intelligence
   - AI Agents and Academic Papers
   - Specific technical questions requiring retrieval

2. "WEB": Use this if the query is about:
   - Current events (news, sports, stocks)
   - Weather
   - Specific facts about the question asked, but couldnt answered with the general knowledge or needs up to date information (e.g., "How old is X?", "Who is CEO of Y?")
   - Information not likely to be in the static database
   - "Search for...", "Find online..."

3. "TOOL_Name_related": Use this if the user explicitly asks to:
   - Update their name
   - Change their profile information
   - Set their user ID
   - "My name is X", "Call me Y"

4. "GENERAL": Use this for:
   - General knowledge (e.g., "Who is the president?", "Python code")
   - Greetings (e.g., "Hi", "How are you?")
   - Questions about the conversation history (e.g., "What did I just ask?", "Summarize our chat")
   - Questions not related to the specific topics above

5. "DIRECT_ANSWER": Use this ONLY if:
   - You have enough information in the conversation history or general knowledge to answer the question IMMEDIATELY without any further search or tools.
   - The user is just acknowledging something (e.g., "Okay", "Thanks").
   - The question is trivial.

Return ONLY one word: "DATABASE", "WEB", "TOOL_Name_related", "GENERAL", or "DIRECT_ANSWER". Do not add punctuation or explanation.
"""
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ])
        self.output_parser = StrOutputParser()

    async def route(self, question: str, context: str = "", config=None) -> str:
        """
        Returns 'vector_db', 'web_search', 'TOOL_Name_related_use', 'general_chat', or 'direct_answer'
        """
        if not self.llm:
             return "vector_db" # Default

        try:
            chain = self.prompt | self.llm | self.output_parser
            result = await chain.ainvoke({"question": question, "context": context}, config=config)
            cleaned_result = result.strip().upper()
            
            if "DATABASE" in cleaned_result:
                return "vector_db"
            elif "WEB" in cleaned_result:
                return "web_search"
            elif "TOOL_NAME_RELATED" in cleaned_result:
                return "TOOL_Name_related_use"
            elif "DIRECT_ANSWER" in cleaned_result:
                return "direct_answer"
            else:
                return "general_chat"
                
        except Exception as e:
            log_info(f"Router error: {e}. Defaulting to vector_db.")
            return "vector_db"
