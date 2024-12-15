import streamlit as st
from typing import Annotated
from typing_extensions import TypedDict
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode, tools_condition

# Define your state structure using TypedDict
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Tools Setup for Arxiv and Wikipedia
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

# Add tools to the list
tools = [wiki_tool]

# Get Groq API Key and Setup ChatGroq
groq_api_key = "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"
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")
llm_with_tools = llm.bind_tools(tools=tools)

# Function to interact with the chatbot
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build Langgraph StateGraph
graph_builder = StateGraph(State)

# Add nodes to the graph
graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# Define conditional edges for the state graph
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# Connect the nodes
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# Compile the graph
graph = graph_builder.compile()

# Streamlit app layout
st.title("Chatbot with LangGraph and Groq")

# Display instructions
st.write("Type your message below and interact with the chatbot.")

# User input field
user_input = st.text_input("You:", "")

# # Display chatbot response
# if user_input:
#     events = graph.stream({"messages": [("user", user_input)]}, stream_mode="values")
#     st.write("Event Debug:", events)
#     # for event in events:
#     #     st.write(f"Chatbot: {event['messages'][-1]['bot_reply']}")

if user_input:
    # Stream the graph events
    events = graph.stream({"messages": [("user", user_input)]}, stream_mode="values")
    
    for event in events:
        # st.write("Event Debug:", event)  # Debugging output

        try:
            # Process each message in the event's `messages` list
            for msg in event["messages"]:
                if isinstance(msg, HumanMessage):
                    st.write(f"User: {msg.content}")  # Display user input
                elif isinstance(msg, AIMessage):
                    if msg.content:  # Only show if there's a response
                        st.write(f"Chatbot: {msg.content}")
                elif isinstance(msg, ToolMessage):
                    st.write(f"Tool Response ({msg.name}): {msg.content}")  # Display tool output
                else:
                    st.error("Unknown message type encountered.")
        except Exception as e:
            st.error(f"Error processing messages: {e}")

