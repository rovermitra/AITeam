import streamlit as st

def render_chat_interface(chat_history):
    """
    Display chat messages in Streamlit.
    """
    for msg in chat_history:
        role = "👤 You" if msg["role"] == "user" else "🤖 Buddy"
        st.markdown(f"**{role}:** {msg['content']}")
