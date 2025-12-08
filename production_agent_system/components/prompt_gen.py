from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import List, Dict, Tuple

class PromptGenerator:
    def __init__(self):
        pass

    def generate_prompt(self, context: str, question: str, history: List[Tuple[str, str]]):
        messages = []
        
        # 1. System Instruction
        if context:
            # RAG Mode
            system_text = f"""You are a helpful AI assistant. 
Answer the user's question based on the following context. 
If the answer is not in the context, say you don't know.

Context:
{context}
"""
        else:
            # General Chat Mode
            system_text = """You are a helpful AI assistant. 
Answer the user's question using your general knowledge. 
Be helpful, harmless, and honest."""

        messages.append(SystemMessage(content=system_text))

        # 2. Chat History
        # History comes in as [("human", "msg"), ("ai", "msg"), ...]
        for role, content in history:
            if role == "human":
                messages.append(HumanMessage(content=content))
            elif role == "ai":
                messages.append(AIMessage(content=content))

        # 3. Current Question
        messages.append(HumanMessage(content=question))
        
        return messages
