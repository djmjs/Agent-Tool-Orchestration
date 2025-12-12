from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.types import Command
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
from ..utils.logger import log_info, log_error

# Define the State
class AgentState(TypedDict):
    messages: List[BaseMessage]
    query: str
    documents: List[Any]
    user_info: Dict[str, Any]
    final_answer: str
    route: str

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
        
        # Define tools
        self.tools = [self.update_user_info]
        self.llm_with_tools = self.llm.llm.bind_tools(self.tools)
        
        # Initialize the graph
        self.app = self._build_graph()

    @tool
    def update_user_info(user_name: str, user_id: str = "unknown"):
        """
        Call this tool ONLY when the user explicitly asks to update their name or ID.
        Do NOT call this tool for general questions.
        """
        # In a real system, this would save to a database
        return f"Successfully updated user info: Name={user_name}, ID={user_id}"

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # Define Nodes
        workflow.add_node("reformulate", self.reformulate_query)
        workflow.add_node("router", self.route_query)
        workflow.add_node("retrieve", self.retrieve_documents)
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
        
        workflow.add_edge("retrieve", "generate")
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

    def should_continue(self, state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END

    async def reformulate_query(self, state: AgentState):
        query = state["query"]
        history = self.stm.get_chat_history()
        
        if history:
            new_query = await self.reformulator.reformulate(query, history)
            log_info(f"Reformulated query: {new_query}")
            return {"query": new_query}
        return {"query": query}

    async def route_query(self, state: AgentState):
        query = state["query"]
        route = await self.router.route(query)
        log_info(f"Routing query to: {route}")
        return {"route": route}

    async def retrieve_documents(self, state: AgentState):
        query = state["query"]
        docs = await self.retriever.retrieve(query)
        reranked_docs = self.reranker.rerank(query, docs)
        return {"documents": reranked_docs}

    async def perform_web_search(self, state: AgentState):
        query = state["query"]
        docs = await self.web_search.search(query)
        return {"documents": docs}

    async def generate_response(self, state: AgentState):
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
            response = await self.llm_with_tools.ainvoke(messages)
        else:
            log_info(f"Using plain LLM (Router decision: {route})")
            response = await self.llm.llm.ainvoke(messages)
        
        return {
            "messages": messages + [response], 
            "final_answer": response.content
        }

    async def execute_tools(self, state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        
        current_info = state.get("user_info", {})
        tool_results = []
        
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            if tool_name == "update_user_info":
                result = self.update_user_info.invoke(tool_args)
                
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
        
        result = await self.app.ainvoke(initial_state)
        
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
