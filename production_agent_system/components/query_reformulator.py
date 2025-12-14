from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from .llm import LLM

class QueryReformulator:
    def __init__(self, llm_service: LLM):
        self.llm_service = llm_service
        
        # Prompt to rephrase the question
        self.rephrase_system_prompt = (
            "System Role: Query Reformulator\n\n"
            "You convert raw user input into structured, high-quality queries suitable for search, "
            "retrieval-augmented generation (RAG), or tool invocation.\n\n"
            "Instructions:\n"
            "- Use the provided chat history to resolve pronouns and references (e.g., 'it', 'he', 'that').\n"
            "- Retain factual constraints, entities, time ranges, and conditions.\n"
            "- Rewrite the query to be explicit, unambiguous, and self-contained.\n"
            "- Expand shorthand, acronyms, and informal phrasing when beneficial.\n"
            "- Avoid speculative details or inferred intent not present in the input.\n"
            "- Do not answer, summarize, or interpret results.\n\n"
            "Strict Output Rules:\n"
            "- Output only the reformulated query.\n"
            "- No preamble, no explanations, no markdown.\n"
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.rephrase_system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])

    async def reformulate(self, question: str, history: list, config=None) -> str:
        """
        Rewrites the user query to be standalone based on chat history.
        """
        if not history:
            return question
            
        # Convert history tuples to messages
        history_messages = []
        for role, content in history:
            if role == "human":
                history_messages.append(HumanMessage(content=content))
            elif role == "ai":
                history_messages.append(AIMessage(content=content))
                
        # Create chain using the underlying LangChain LLM object
        if not self.llm_service.llm:
            raise ValueError("LLM service not initialized")

        chain = self.prompt | self.llm_service.llm | StrOutputParser()
        
        # Invoke
        reformulated_query = await chain.ainvoke(
            {"history": history_messages, "question": question},
            config=config
        )
        return reformulated_query
