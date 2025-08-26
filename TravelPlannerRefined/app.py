import streamlit as st
import json
import os
from dotenv import load_dotenv
from agents.itinerary_planner import find_best_flights
from agents.chat_agent import chat_with_ai
from components.itinerary_display import display_itinerary
from components.chat_ui import render_chat_interface

# --- Load OpenAI Key ---
load_dotenv()
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Load Data ---
with open("data/flight_data.json") as f:
    flight_data = json.load(f)

with open("data/user_data.json") as f:
    user_data = json.load(f)

# --- Session State Setup ---
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "chat_mode" not in st.session_state:
    st.session_state["chat_mode"] = False

# --- Users ---
main_user = user_data[0]
buddy_user = user_data[1] if len(user_data) > 1 else None

# --- Streamlit Page ---
st.set_page_config(page_title="Travel Buddy AI", page_icon="✈️", layout="wide")
st.title("🌍 Travel Buddy AI")
st.subheader(f"Welcome, {main_user['name']}!")

# --- Sidebar ---
with st.sidebar:
    st.header("🧳 Your Profile")
    st.json(main_user)
    if buddy_user:
        st.header("🤝 Matched Buddy")
        st.json(buddy_user)

st.markdown("---")

# --- Action Buttons ---
col1, col2 = st.columns(2)
with col1:
    if st.button("✈️ Plan Flights Now"):
        itinerary = find_best_flights(main_user, flight_data)
        st.success("Here’s your best itinerary based on preferences!")
        display_itinerary(itinerary)

with col2:
    if st.button("💬 Chat With Buddy"):
        st.session_state["chat_mode"] = True

# --- Chat Mode ---
if st.session_state["chat_mode"]:
    st.markdown("### 💬 Chat with your buddy")

    # Scrollable container for chat messages
    chat_container = st.container()
    with chat_container:
        render_chat_interface(st.session_state["chat_history"])


    # --- Chat input form ---
    # --- Chat input form ---
    with st.form(key="chat_form", clear_on_submit=True): # Use clear_on_submit=True
        user_input = st.text_area("Type your message here:", key="chat_input", height=70)
        send_button = st.form_submit_button("Send")

        if send_button and user_input.strip() != "":
            # Append user message
            st.session_state["chat_history"].append({"role": "user", "content": user_input.strip()})

            # --- THIS IS THE LINE TO CHANGE ---
            # Original call:
            # ai_reply = chat_with_ai(user_input.strip(), st.session_state["chat_history"])
            
            # New call with user profiles for context:
            ai_reply = chat_with_ai(
                user_input.strip(), 
                st.session_state["chat_history"], 
                main_user, 
                buddy_user
            )
            # ------------------------------------

            st.session_state["chat_history"].append({"role": "assistant", "content": ai_reply})

            # Refresh chat display
            chat_container.empty()
            with chat_container:
                render_chat_interface(st.session_state["chat_history"])

    # --- Generate itinerary from chat ---
    if st.button("✨ Generate Itinerary from Chat"):
        itinerary = find_best_flights(main_user, flight_data)
        st.success("Here’s your personalized itinerary based on the conversation!")
        display_itinerary(itinerary)
