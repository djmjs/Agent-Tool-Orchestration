from typing import List, Dict, Tuple, Any
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

class ShortTermMemory:
    def __init__(self, llm: BaseChatModel, max_token_limit: int = 2000):
        """
        Initializes the ShortTermMemory with summarization capabilities.
        
        :param llm: The LangChain Chat Model instance to use for summarization.
        :param max_token_limit: The number of tokens to keep before summarizing.
        """
        self.memory = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=max_token_limit,
            return_messages=True
        )
        # We still keep track of UI history separately to show the user the full conversation
        # in the UI, even if the internal memory has summarized it.
        self.user_prompt_history: List[str] = []
        self.chat_answers_history: List[str] = []

    def add_interaction(self, user_query: str, system_response: str, formatted_response: str = None):
        """
        Adds an interaction to the memory.
        :param user_query: The user's question.
        :param system_response: The raw answer from the LLM.
        :param formatted_response: The answer formatted with sources (for UI display).
        """
        if formatted_response is None:
            formatted_response = system_response

        # Save to UI history
        self.user_prompt_history.append(user_query)
        self.chat_answers_history.append(formatted_response)

        # Save to LangChain Memory (which handles summarization automatically)
        self.memory.save_context(
            {"input": user_query}, 
            {"output": system_response}
        )

    def get_chat_history(self) -> List[Tuple[str, str]]:
        """
        Returns the history as a list of tuples (role, content).
        This is used by the QueryReformulator.
        """
        # Load memory variables (this triggers summarization if needed)
        memory_output = self.memory.load_memory_variables({})
        messages = memory_output.get("history", [])
        
        history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append(("human", msg.content))
            elif isinstance(msg, AIMessage):
                history.append(("ai", msg.content))
            elif isinstance(msg, SystemMessage):
                # If there is a summary, it usually comes as a SystemMessage
                history.append(("system", f"Summary of previous conversation: {msg.content}"))
        
        return history

    def get_ui_history(self) -> Dict[str, List[str]]:
        """
        Returns the full history for UI display.
        """
        return {
            "user_prompt_history": self.user_prompt_history,
            "chat_answers_history": self.chat_answers_history
        }

    def clear(self):
        self.memory.clear()
        self.user_prompt_history = []
        self.chat_answers_history = []
