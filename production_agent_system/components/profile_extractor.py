from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from .llm import LLM
from .postgres_storage import UserProfile
from ..utils.logger import log_info, log_error

class ProfileExtractor:
    def __init__(self, llm: Optional[LLM] = None):
        self.llm = llm if llm else LLM()
        self.parser = PydanticOutputParser(pydantic_object=UserProfile)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert data extractor. Your task is to analyze the conversation and extract specific user information into a JSON format.

            Target Information Categories:
            1. preferences: User likes, dislikes, preferred tools/methods (e.g., "I prefer Python", "I hate Java").
            2. projects: Specific projects the user is working on (e.g., "building a RAG agent", "working on a weather app").
            3. constraints: Limitations or requirements (e.g., "must use open source", "no paid APIs", "low latency").
            4. expertise: User's skill level or background (e.g., "I am a beginner", "senior dev", "familiar with Docker").
            5. environment: System info (e.g., "Windows", "using VS Code", "16GB RAM").
            6. personal_info: Personal details about the user (e.g., "My name is John", "I live in New York", "I am 30 years old").

            Instructions:
            - Extract ONLY information explicitly stated or strongly implied in the conversation.
            - If a category has no relevant information, leave it as an empty list [].
            - Do not hallucinate information.
            - Be concise. Extract short phrases or keywords.
            - Return ONLY the JSON object.

            Examples:
            Input: "I'm trying to build a chatbot using LangChain but I'm new to Python. My name is Alice."
            Output: {{
                "preferences": ["LangChain"],
                "projects": ["chatbot"],
                "constraints": [],
                "expertise": ["new to Python"],
                "environment": [],
                "personal_info": ["Name: Alice"]
            }}

            Input: "My computer is a Mac M1 with 16GB RAM. I need to run this locally without internet."
            Output: {{
                "preferences": [],
                "projects": [],
                "constraints": ["run locally", "no internet"],
                "expertise": [],
                "environment": ["Mac M1", "16GB RAM"],
                "personal_info": []
            }}

            {format_instructions}
            """),
            ("human", "{conversation}")
        ])

    async def extract(self, conversation: str) -> UserProfile:
        """
        Extracts user profile information from the conversation text.
        """
        log_info(f"ProfileExtractor analyzing {len(conversation)} chars of conversation.")
        try:
            chain = self.prompt | self.llm.llm | self.parser
            
            result = await chain.ainvoke({
                "conversation": conversation,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            log_info(f"ProfileExtractor result: {result}")
            return result
        except Exception as e:
            log_error(f"Error extracting profile: {e}")
            # Return empty profile on error to avoid breaking the flow
            return UserProfile()
