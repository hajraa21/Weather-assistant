# main.py - Backend LangGraph weather assistant
import asyncio
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient

# Load env variables
load_dotenv()

async def build_graph(model_name: str = "llama-3.3-70b-versatile", system_prompt: str = None):
    """
    Build and compile the LangGraph weather assistant.
    
    Parameters:
    -----------
    model_name : str
        Groq model to use (default: "llama-3.3-70b-versatile")
    system_prompt : str
        Custom system prompt
    
    Returns:
    --------
    compiled graph : StateGraph
    """
    if system_prompt is None:
        system_prompt = (
            "You are a helpful assistant with access to weather tools. "
            "Always use the tools to provide accurate real-time weather data "
            "in both Celsius and Fahrenheit. Do not make assumptions; provide "
            "only genuine info to the user. Also provide useful lifestyle tips "
            "based on the weather such as whether it is a good day to go out, "
            "what kind of clothes to wear, what food will be good in this weather, etc."
        )
    
    # ── 1. Connect to the MCP weather server ──────────────────────────────────
    client = MultiServerMCPClient(
        {
            "weather": {
                "transport": "streamable_http",
                "url": "https://weather-d26bfd98e65f.fastmcp.app/mcp",
                "headers": {
                    "Authorization": "Bearer fmcp_hCVUlCHnEp4fokPkVlLO9CcqTKzdi7U9K3th4sgBATo"
                },
            }
        }
    )
    tools = await client.get_tools()

    # ── 2. Initialise the LLM and bind weather tools ──────────────────────────
    llm = ChatGroq(model=model_name)
    llm_with_tools = llm.bind_tools(tools)

    # ── 3. Define the graph state schema ─────────────────────────────────────
    class State(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    # ── 4. Chat node: injects system prompt and calls the LLM ─────────────────
    async def chat_node(state: State):
        messages = state["messages"]
        sys_msg = SystemMessage(content=system_prompt)

        # Prepend system message if it isn't already there
        if not isinstance(messages[0], SystemMessage):
            messages = [sys_msg] + messages

        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    # ── 5. Wire up the graph ──────────────────────────────────────────────────
    g = StateGraph(State)
    g.add_node("chat_node", chat_node)
    g.add_node("tools", ToolNode(tools=tools))

    g.add_edge(START, "chat_node")
    g.add_conditional_edges("chat_node", tools_condition)
    g.add_edge("tools", "chat_node")

    return g.compile()

# ════════════════════════════════════════════════════════════════════════════════
#  ORIGINAL TEST FUNCTION (unchanged)
# ════════════════════════════════════════════════════════════════════════════════
async def main():
    """
    Original test function - run with: python main.py
    """
    graph = await build_graph()  # Uses the new build_graph function
    
    initial_input = {'messages': [HumanMessage(content='what is the weather in chennai?')]}
    output = await graph.ainvoke(initial_input)
    
    print(output['messages'][-1].content)

if __name__ == "__main__":
    asyncio.run(main())