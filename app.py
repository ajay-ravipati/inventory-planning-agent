import streamlit as st
from agent5 import agent_app
import os

# Page configuration
st.set_page_config(page_title="Inventory Agent", page_icon="📦", layout="centered")

# Create two tabs
tab_chat, tab_readme = st.tabs(["Chat with the Agent", "ReadMe"])

# --- TAB 1: THE README ---
with tab_readme:
    if os.path.exists("README.md"):
        with open("README.md", "r", encoding="utf-8") as f:
            content = f.read()
            # Strip out the Hugging Face YAML header at the top
            if content.startswith("---"):
                clean_content = content.split("---", 2)[-1]
            else:
                clean_content = content
            st.markdown(clean_content)
    else:
        st.warning("README.md not found.")

# --- TAB 2: THE CHAT APP ---
with tab_chat:
    st.title("📦 Inventory Planning Agent")
    st.caption("Ask questions about stock levels, aggregate counts, or supplier policies.")
    
    # ... (Keep all your existing chat UI and session_state logic here) ...
with st.sidebar:
    st.markdown("### Try These Examples(For clearer understanding use ReadMe Tab)")
    st.markdown("**1. Forecasting (SQL)**")
    st.code("How many red sarees do we need to order for next week?", language="text")
    st.markdown("**2. Analytics (SQL)**")
    st.code("What is the average unit cost of all our black pants?", language="text")
    st.markdown("**3. Policies (Vector RAG)**")
    st.code("What is the lead time for SUPP_A?", language="text")
    st.markdown("**4. Scope Guardrail Test**")
    st.code("Write a Python script for me.", language="text")

# 1. UI Memory (Clean text for rendering)
if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = []

# 2. Graph Memory (Complex LangChain objects for the agent's brain)
if "graph_messages" not in st.session_state:
    st.session_state.graph_messages = []

# Display previous chat messages on page refresh
for message in st.session_state.ui_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("E.g., How many products are currently below the safety buffer?"):
    
    # Display user message in UI and save to UI memory
    st.chat_message("user").markdown(prompt)
    st.session_state.ui_messages.append({"role": "user", "content": prompt})

    # Save to Graph memory
    st.session_state.graph_messages.append(("user", prompt))

    # Generate response with a loading spinner
    with st.spinner("Agent is searching databases..."):
        try:
            # Invoke LangGraph with the ENTIRE conversation history
            response = agent_app.invoke({
                "messages": st.session_state.graph_messages,
                "tool_error_retries": 0
            })
            
            # Update Graph memory with the new history (including hidden tool calls)
            st.session_state.graph_messages = response["messages"]
            
            # Extract clean text (handling newer Gemini model list formats)
            final_msg = response["messages"][-1]
            if isinstance(final_msg.content, list):
                final_text = "".join(
                    block["text"] for block in final_msg.content 
                    if isinstance(block, dict) and "text" in block
                )
            else:
                final_text = final_msg.content
                
        except Exception as e:
            final_text = f"An error occurred: {str(e)}"

    # Display assistant response in UI and save to UI memory
    with st.chat_message("assistant"):
        st.markdown(final_text)
    st.session_state.ui_messages.append({"role": "assistant", "content": final_text})