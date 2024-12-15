import streamlit as st
from typing import Annotated
from typing_extensions import TypedDict
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, HumanMessage, AIMessage, ToolMessage
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

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input field
user_input = st.text_input("You:", "")

# Process user input
if user_input:
    # Add the user input as a HumanMessage
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Process the message through the graph
    events = graph.stream({"messages": st.session_state.chat_history}, stream_mode="values")
    for event in events:
        st.write("Event Debug:", event)  # Debugging

        try:
            # Get the last message in the 'messages' list
            last_message = event['messages'][-1]

            # Handle different message types
            if isinstance(last_message, HumanMessage):
                st.session_state.chat_history.append(last_message)
            elif isinstance(last_message, AIMessage):
                st.session_state.chat_history.append(last_message)
            elif isinstance(last_message, ToolMessage):
                st.session_state.chat_history.append(last_message)
            else:
                st.error("Unknown message type received.")
        except Exception as e:
            st.error(f"Error retrieving chatbot reply: {e}")

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        st.write(f"You: {message.content}")
    elif isinstance(message, AIMessage):
        st.write(f"Chatbot: {message.content}")
    elif isinstance(message, ToolMessage):
        st.write(f"Tool ({message.name}): {message.content}")
