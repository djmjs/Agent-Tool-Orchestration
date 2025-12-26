from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from ..components.short_term_memory import ShortTermMemory
from ..components.retriever import Retriever
from ..components.reranker import ReRanker
from ..components.prompt_gen import PromptGenerator
from ..components.llm import LLM
from ..components.query_reformulator import QueryReformulator
from ..components.router import Router
from ..components.web_search import WebSearch
from ..components.relevance_grader import RelevanceGrader
from ..components.profile_extractor import ProfileExtractor
from ..components.postgres_storage import postgres_storage
from ..utils.logger import log_info, log_error
from langfuse.langchain import CallbackHandler

# Define the State
class AgentState(TypedDict):
    messages: List[BaseMessage]
    query: str
    documents: List[Any]
    user_info: Dict[str, Any]
    final_answer: str
    route: str
    relevance: str
    is_topic_switch: bool

class GraphOrchestrator:
    def __init__(self):
        self.llm = LLM()
        self.stm = ShortTermMemory(llm=self.llm.llm)
        self.retriever = Retriever()
        self.reranker = ReRanker()
        self.prompt_gen = PromptGenerator()
        self.reformulator = QueryReformulator(self.llm)
        self.router = Router(self.llm)
        self.web_search = WebSearch()
        self.grader = RelevanceGrader(self.llm)
        self.profile_extractor = ProfileExtractor(self.llm)
        
        # Define tools
        self.tools = [self.update_user_info]
        self.llm_with_tools = self.llm.llm.bind_tools(self.tools)
        
        # Initialize the graph
        self.app = self._build_graph()

    @tool
    def update_user_info(user_name: str, user_id: str = "unknown"):
        """
        Updates the user's profile information. 
        Use this tool when the user explicitly asks to change their name or set their ID.
        """
        return f"User info updated: Name={user_name}, ID={user_id}"

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # Define Nodes
        workflow.add_node("manage_context", self.manage_context)
        workflow.add_node("reformulate", self.reformulate_query)
        workflow.add_node("dispatch", self.dispatch_node) # Dummy node for routing
        workflow.add_node("router", self.route_query)
        workflow.add_node("retrieve", self.retrieve_documents)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("web_search", self.perform_web_search)
        workflow.add_node("generate", self.generate_response)
        workflow.add_node("tools", self.execute_tools)

        # Define Edges
        workflow.set_entry_point("manage_context")
        workflow.add_edge("manage_context", "router")
        
        # Conditional routing
        workflow.add_conditional_edges(
            "router",
            lambda state: state["route"],
            {
                "vector_db": "reformulate",
                "web_search": "reformulate",
                "TOOL_Name_related_use": "reformulate",
                "general_chat": "reformulate",
                "direct_answer": "generate"
            }
        )
        
        workflow.add_edge("reformulate", "dispatch")
        
        # Conditional dispatching after reformulation
        workflow.add_conditional_edges(
            "dispatch",
            lambda state: state["route"],
            {
                "vector_db": "retrieve",
                "web_search": "web_search",
                "TOOL_Name_related_use": "generate", # Tools are handled in generate -> tools loop
                "general_chat": "generate"
            }
        )
        
        workflow.add_edge("retrieve", "grade_documents")
        
        # Conditional edge from grader
        workflow.add_conditional_edges(
            "grade_documents",
            self.check_relevance,
            {
                "relevant": "generate",
                "irrelevant": "web_search"
            }
        )

        workflow.add_edge("web_search", "generate")
        
        # Conditional edge from generate
        workflow.add_conditional_edges(
            "generate",
            self.should_continue,
            {
                "tools": "tools",
                END: END
            }
        )
        workflow.add_edge("tools", "generate")

        return workflow.compile()

    async def grade_documents(self, state: AgentState, config: RunnableConfig):
        query = state["query"]
        docs = state.get("documents", [])
        
        if not docs:
            return {"documents": []} # Will trigger irrelevant in check_relevance if we handle it right, or we can just pass through.
            
        context = "\n\n".join([d.page_content for d in docs])
        grade = await self.grader.grade(query, context, config=config)
        state["relevance"] = grade
        
        return {"relevance": grade}

    def check_relevance(self, state: AgentState):
        # If we came from web_search, we don't grade. But this edge is only from grade_documents.
        relevance = state.get("relevance", "yes")
        docs = state.get("documents", [])
        
        if not docs:
            return "irrelevant"
            
        if "no" in relevance:
            log_info("Documents graded as irrelevant. Switching to Web Search.")
            return "irrelevant"
            
        return "relevant"

    def should_continue(self, state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END

    def dispatch_node(self, state: AgentState):
        # Dummy node to facilitate routing after reformulation
        return {}

    async def extract_user_profile_helper(self, query: str, answer: str):
        """
        Extracts user profile information from the conversation and saves it to Postgres.
        """
        conversation_text = f"User: {query}\nAgent: {answer}"
        
        log_info("Extracting user profile information from recent context...")
        profile = await self.profile_extractor.extract(conversation_text)
        
        # Check if any field has data
        has_data = any([
            profile.preferences,
            profile.projects,
            profile.constraints,
            profile.expertise,
            profile.environment,
            profile.personal_info
        ])
        
        if has_data:
            # Use a default user_id
            # In a real app, user_id would come from auth context
            user_id = "default_user"
            
            log_info(f"Saving extracted profile for user {user_id}: {profile}")
            await postgres_storage.upsert_profile(user_id, profile)
        else:
            log_info("No relevant profile information found to save.")

    async def manage_context(self, state: AgentState, config: RunnableConfig):
        query = state["query"]
        history = self.stm.get_chat_history()
        updates = {"is_topic_switch": False} # Default to False

        # --- Long-Term Memory (LTM) Check ---
        try:
            user_id = "default_user" # TODO: Get from auth context
            profile = await postgres_storage.get_profile(user_id)
            
            if profile:
                # Convert profile to string for grading
                profile_data = profile.dict()
                profile_str = str(profile_data)
                
                # Grade relevance of LTM to current query
                ltm_grade = await self.grader.grade_profile(query, profile_str, config=config)
                
                if "yes" in ltm_grade:
                    log_info("LTM Profile found relevant. Injecting into context.")
                    updates["user_info"] = profile_data
                else:
                    log_info("LTM Profile found irrelevant. Skipping.")
        except Exception as e:
            log_error(f"Error checking LTM: {e}")
        # ------------------------------------
        
        if history:
            # Check Topic Relevance using FULL history (Summary + Recent Messages)
            # Format history into a string
            history_str = "\n".join([f"{role.upper()}: {content}" for role, content in history])
            
            grade = await self.grader.grade_history(query, history_str, config=config)
            if "no" in grade:
                log_info("Topic switch detected. Marking context as irrelevant.")
                updates["is_topic_switch"] = True

        return updates

    async def reformulate_query(self, state: AgentState, config: RunnableConfig):
        query = state["query"]
        route = state.get("route", "general_chat")
        history = self.stm.get_chat_history()
        is_topic_switch = state.get("is_topic_switch", False)
        
        use_history = True
        if is_topic_switch:
            use_history = False

        if use_history and history:
            new_query = await self.reformulator.reformulate(query, history, route=route, config=config)
            log_info(f"Reformulated query ({route}): {new_query}")
            return {"query": new_query}
        
        return {"query": query}

    async def route_query(self, state: AgentState, config: RunnableConfig):
        query = state["query"]
        is_topic_switch = state.get("is_topic_switch", False)
        
        # Prepare context string for the router
        context_parts = []
        
        # Add User Info
        user_info = state.get("user_info", {})
        if user_info:
            context_parts.append(f"User Profile: {user_info}")
            
        # Add Chat History (Summary)
        history = self.stm.get_chat_history()
        if history and not is_topic_switch:
            # Format full history (Summary + Recent)
            history_str = "\n".join([f"{role}: {content}" for role, content in history])
            context_parts.append(f"Chat History:\n{history_str}")
            
        context_str = "\n\n".join(context_parts)
        
        route = await self.router.route(query, context=context_str, config=config)
        log_info(f"Routing query to: {route}")
        return {"route": route}

    async def retrieve_documents(self, state: AgentState, config: RunnableConfig):
        query = state["query"]
        docs = await self.retriever.retrieve(query)
        reranked_docs = self.reranker.rerank(query, docs)
        return {"documents": reranked_docs}

    async def perform_web_search(self, state: AgentState, config: RunnableConfig):
        query = state["query"]
        docs = await self.web_search.search(query, config=config)
        return {"documents": docs}

    async def generate_response(self, state: AgentState, config: RunnableConfig):
        query = state["query"]
        docs = state.get("documents", [])
        messages = state["messages"]
        route = state.get("route", "general_chat")
        
        # Get history
        history = self.stm.get_chat_history()
        is_topic_switch = state.get("is_topic_switch", False)
        if is_topic_switch:
            history = []
        
        # Get User Info (LTM)
        user_info = state.get("user_info", None)
        
        # If this is the first pass (no tool calls yet), prepare the prompt
        if len(messages) == 1:
             context_text = ""
             if (route == "vector_db" or route == "web_search") and docs:
                 # Format context from documents
                 context_text = "\n\n".join([d.page_content for d in docs])
             
             # Use the existing generate_prompt method
             messages = self.prompt_gen.generate_prompt(context_text, query, history, user_info=user_info)
        
        # Select LLM based on route
        if route == "TOOL_Name_related_use":
            log_info("Using LLM with tools (Router decision)")
            response = await self.llm_with_tools.ainvoke(messages, config=config)
        else:
            log_info(f"Using plain LLM (Router decision: {route})")
            response = await self.llm.llm.ainvoke(messages, config=config)
        
        return {
            "messages": messages + [response], 
            "final_answer": response.content
        }

    async def execute_tools(self, state: AgentState, config: RunnableConfig):
        messages = state["messages"]
        last_message = messages[-1]
        
        current_info = state.get("user_info", {})
        tool_results = []
        
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            if tool_name == "update_user_info":
                result = self.update_user_info.invoke(tool_args, config=config)
                
                # Update local state user_info as well
                current_info.update(tool_args)
                
                tool_results.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"]
                ))
            else:
                # Handle hallucinated tools
                tool_results.append(ToolMessage(
                    content=f"Error: Tool '{tool_name}' does not exist. Please answer the user's question directly without using tools.",
                    tool_call_id=tool_call["id"]
                ))
        
        return {
            "messages": messages + tool_results,
            "user_info": current_info
        }

    async def update_memory_tool(self, state: AgentState):
        # Legacy placeholder
        pass


    async def process_query(self, query: str) -> Dict[str, Any]:
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "documents": [],
            "user_info": {},
            "final_answer": ""
        }
        
        # Initialize Langfuse Callback Handler
        # It will automatically pick up LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST from env
        langfuse_handler = CallbackHandler()
        
        result = await self.app.ainvoke(initial_state, config={"callbacks": [langfuse_handler]})
        
        # Save to memory (Legacy STM)
        self.stm.add_interaction(query, result["final_answer"])
        
        # Extract and save user profile (Postgres)
        # We run this asynchronously but await it to ensure it completes before returning
        # In a high-throughput system, you might want to run this as a background task
        try:
            await self.extract_user_profile_helper(query, result["final_answer"])
        except Exception as e:
            log_error(f"Failed to extract/save user profile: {e}")

        return {
            "answer": result["final_answer"],
            "sources": [
                {"title": d.metadata.get("title", "Unknown"), "url": d.metadata.get("source", "#"), "content_preview": d.page_content[:200]} 
                for d in result["documents"]
            ],
            "user_prompt_history": self.stm.user_prompt_history,
            "chat_answers_history": self.stm.chat_answers_history
        }
