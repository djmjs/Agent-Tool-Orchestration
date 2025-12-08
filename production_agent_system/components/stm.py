from typing import List, Dict, Any, Tuple

class ShortTermMemory:
    def __init__(self, max_history: int = 5):
        self.max_history = max_history
        # Initialize the 3 specific lists requested
        self.chat_answers_history: List[str] = []
        self.user_prompt_history: List[str] = []
        self.chat_history: List[Tuple[str, str]] = []

    def add_interaction(self, user_query: str, system_response: str, formatted_response: str = None):
        """
        Adds an interaction to the memory.
        :param user_query: The user's question.
        :param system_response: The raw answer from the LLM.
        :param formatted_response: The answer formatted with sources (for UI display).
        """
        if formatted_response is None:
            formatted_response = system_response

        self.user_prompt_history.append(user_query)
        self.chat_answers_history.append(formatted_response)
        
        self.chat_history.append(("human", user_query))
        self.chat_history.append(("ai", system_response))
        
        # Trim history if it exceeds max_history
        # Note: chat_history has 2 entries per turn, others have 1
        if len(self.user_prompt_history) > self.max_history:
            self.user_prompt_history = self.user_prompt_history[-self.max_history:]
            self.chat_answers_history = self.chat_answers_history[-self.max_history:]
            self.chat_history = self.chat_history[-(self.max_history * 2):]

    def get_chat_history(self) -> List[Tuple[str, str]]:
        return self.chat_history

    def get_ui_history(self) -> Dict[str, List[str]]:
        return {
            "user_prompt_history": self.user_prompt_history,
            "chat_answers_history": self.chat_answers_history
        }

    def clear(self):
        self.chat_answers_history = []
        self.user_prompt_history = []
        self.chat_history = []
