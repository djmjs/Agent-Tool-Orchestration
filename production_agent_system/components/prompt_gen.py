from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import List, Dict, Tuple

class PromptGenerator:
    def __init__(self):
        pass

    def generate_prompt(self, context: str, question: str, history: List[Tuple[str, str]], user_info: Dict = None):
        messages = []
        
        # 1. System Instruction
        system_text = "You are a helpful AI assistant."
        
        if user_info:
            system_text += f"\n\nKNOWN USER INFORMATION:\n{user_info}\nYou can use this information to personalize your answer if relevant."

        if context:
            # RAG Mode: Strict adherence to context
            system_text += """\n\nUse the provided context to answer the current question. 
Ignore previous conversation history if it conflicts with the current context. 
If the context does not contain the answer to the current question, state that you don't know."""
        else:
            # General Chat Mode: Conversational
            system_text += """\n\nAnswer the user's question using your general knowledge and the conversation history. 
Be helpful, harmless, and honest."""

        messages.append(SystemMessage(content=system_text))

        # 2. Chat History
        # History comes in as [("human", "msg"), ("ai", "msg"), ...]
        for role, content in history:
            if role == "human":
                messages.append(HumanMessage(content=content))
            elif role == "ai":
                messages.append(AIMessage(content=content))
            elif role == "system":
                messages.append(SystemMessage(content=content))

        # 3. Current Question with Context
        if context:
            user_content = f"""Context:
{context}

Question: 
{question}
"""
        else:
            user_content = question

        messages.append(HumanMessage(content=user_content))
        
        return messages
