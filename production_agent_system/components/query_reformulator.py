from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from .llm import LLM

class QueryReformulator:
    def __init__(self, llm_service: LLM):
        self.llm_service = llm_service
        
        # --- Prompts for different routes ---
        
        # 1. Vector DB (DATABASE)
        self.db_prompt = (
            "System Role: Database Query Optimizer\n\n"
            "Your task is to reformulate the user's query into a precise, keyword-rich search query for a vector database.\n"
            "The database contains technical documents about AI, LLMs, and Reinforcement Learning.\n\n"
            "Instructions:\n"
            "- Resolve pronouns using history (e.g., 'how does it work?' -> 'how does PPO work?').\n"
            "- Remove conversational filler ('I was wondering', 'Can you tell me').\n"
            "- Focus on technical terms, algorithms, and specific concepts.\n"
            "- Keep it concise and search-oriented.\n"
            "- Output ONLY the reformulated query."
        )

        # 2. Web Search (WEB)
        self.web_prompt = (
            "System Role: Web Search Expert\n\n"
            "Your task is to reformulate the user's query into an effective Google search query.\n\n"
            "Instructions:\n"
            "- Resolve pronouns and context from history.\n"
            "- Include specific entity names, dates, or locations if relevant.\n"
            "- Use search operators if helpful (though keep it natural language mostly).\n"
            "- If the user asks for 'latest news', include the current year/month context if known (assume current date is relevant).\n"
            "- Output ONLY the reformulated query."
        )

        # 3. Tool Use (TOOL_Name_related)
        self.tool_prompt = (
            "System Role: Tool Argument Extractor\n\n"
            "Your task is to clarify the user's intent for a profile update tool.\n"
            "The tool updates the user's name or ID.\n\n"
            "Instructions:\n"
            "- Make sure the new name or ID is clearly stated.\n"
            "- If the user says 'My name is John', rewrite to 'Update user name to John'.\n"
            "- Resolve any ambiguity from history.\n"
            "- Output ONLY the reformulated command/query."
        )

        # 4. General Chat (GENERAL)
        self.general_prompt = (
            "System Role: Conversational Context Resolver\n\n"
            "Your task is to rewrite the last user message to be self-contained, resolving any context from the chat history.\n\n"
            "Instructions:\n"
            "- Resolve pronouns (he, she, it, they) to the entities they refer to.\n"
            "- Maintain the conversational tone.\n"
            "- Do not strip away the user's personality or specific phrasing unless it's ambiguous.\n"
            "- Output ONLY the reformulated message."
        )
        
        # Default fallback
        self.default_prompt = self.general_prompt

    def _get_prompt_template(self, system_prompt: str):
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])

    async def reformulate(self, question: str, history: list, route: str = "general_chat", config=None) -> str:
        """
        Rewrites the user query to be standalone based on chat history and the specific route.
        """
        if not history:
            return question
            
        # Select prompt based on route
        if route == "vector_db":
            sys_prompt = self.db_prompt
        elif route == "web_search":
            sys_prompt = self.web_prompt
        elif route == "TOOL_Name_related_use":
            sys_prompt = self.tool_prompt
        else:
            sys_prompt = self.general_prompt
            
        prompt_template = self._get_prompt_template(sys_prompt)

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

        chain = prompt_template | self.llm_service.llm | StrOutputParser()
        
        # Invoke
        reformulated_query = await chain.ainvoke(
            {"history": history_messages, "question": question},
            config=config
        )
        return reformulated_query
