from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from .llm import LLM

class QueryReformulator:
    def __init__(self, llm_service: LLM):
        self.llm_service = llm_service
        
        # Prompt to rephrase the question
        self.rephrase_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.rephrase_system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])

    async def reformulate(self, question: str, history: list) -> str:
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
        reformulated_query = await chain.ainvoke({"history": history_messages, "question": question})
        return reformulated_query
