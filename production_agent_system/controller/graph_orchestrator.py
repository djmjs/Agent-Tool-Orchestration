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
        workflow.add_node("reformulate", self.reformulate_query)
        workflow.add_node("router", self.route_query)
        workflow.add_node("retrieve", self.retrieve_documents)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("web_search", self.perform_web_search)
        workflow.add_node("generate", self.generate_response)
        workflow.add_node("tools", self.execute_tools)

        # Define Edges
        workflow.set_entry_point("reformulate")
        workflow.add_edge("reformulate", "router")
        
        # Conditional routing
        workflow.add_conditional_edges(
            "router",
            lambda state: state["route"],
            {
                "vector_db": "retrieve",
                "web_search": "web_search",
                "TOOL_Name_related_use": "generate",
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

    async def reformulate_query(self, state: AgentState, config: RunnableConfig):
        query = state["query"]
        history = self.stm.get_chat_history()
        
        if history:
            # Check Topic Relevance
            # Find the last human message
            last_human_msg = None
            for role, content in reversed(history):
                if role == "human":
                    last_human_msg = content
                    break
            
            if last_human_msg:
                grade = await self.grader.grade_history(query, last_human_msg, config=config)
                if "no" in grade:
                    log_info("Topic switch detected. Clearing short-term memory context for reformulation.")
                    history = [] # Don't use history for reformulation
            
            if history:
                new_query = await self.reformulator.reformulate(query, history, config=config)
                log_info(f"Reformulated query: {new_query}")
                # Update the state with the NEW query so downstream nodes use it
                return {"query": new_query}
        
        return {"query": query}

    async def route_query(self, state: AgentState, config: RunnableConfig):
        # Use the potentially reformulated query from the state
        query = state["query"]
        route = await self.router.route(query, config=config)
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
        
        # If this is the first pass (no tool calls yet), prepare the prompt
        if len(messages) == 1:
             context_text = ""
             if (route == "vector_db" or route == "web_search") and docs:
                 # Format context from documents
                 context_text = "\n\n".join([d.page_content for d in docs])
             
             # Use the existing generate_prompt method
             messages = self.prompt_gen.generate_prompt(context_text, query, history)
        
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
        
        return {
            "answer": result["final_answer"],
            "sources": [
                {"title": d.metadata.get("title", "Unknown"), "url": d.metadata.get("source", "#"), "content_preview": d.page_content[:200]} 
                for d in result["documents"]
            ],
            "user_prompt_history": self.stm.user_prompt_history,
            "chat_answers_history": self.stm.chat_answers_history
        }
