# Define Imports
import asyncio
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

#load env variables
load_dotenv()


async def main():
    
    client= MultiServerMCPClient(
     {
        "weather": {
            "transport": "streamable_http",
            "url": "https://weather-d26bfd98e65f.fastmcp.app/mcp",
            "headers": {
                # Replace 'YOUR_API_KEY' with the actual key for that service
                "Authorization": "Bearer fmcp_hCVUlCHnEp4fokPkVlLO9CcqTKzdi7U9K3th4sgBATo" 
            }
        } 
    }
)
    tools= await client.get_tools()
    llm= ChatGroq(model= "llama-3.3-70b-versatile")
    llm_with_tools = llm.bind_tools(tools)

    class State(TypedDict):

        messages: Annotated[list[BaseMessage], add_messages]


    async def chat_node(state: State):
        messages= state['messages']
        sys_msg = SystemMessage(content="You are a helpful assistant with access to weather tools. Always use the tools to provide accurate real-time weather data in both celcius and farheneit. Do not make assumptions and provide only genuine info to the user. Also provide information based on the weather data such as whether it is a good day to go out, what kind of clothes to wear, what food will be good in this weather, etc.")
    
    # 2. Prepend it to the message list if it's not already there
    # This ensures the LLM sees the instructions first
        if not isinstance(messages[0], SystemMessage):
            messages = [sys_msg] + messages

        response= await llm_with_tools.ainvoke(messages)
        return{'messages': [response]}

    def weather(state: State):
        
        tool_node= ToolNode(tools= tools)

    g= StateGraph(State)

    g.add_node('chat_node', chat_node)
    g.add_node('tools', ToolNode(tools= tools))

    g.add_edge(START, 'chat_node')
    g.add_conditional_edges('chat_node', tools_condition)
    g.add_edge('tools','chat_node')

    graph= g.compile()


    #print(graph.get_graph().draw_mermaid())

    initial_input = {'messages': [HumanMessage(content='what is the weather in chennai?')]}
    output = await graph.ainvoke(initial_input)
    
    print(output['messages'][-1].content)

if __name__ == "__main__":
    asyncio.run(main())

