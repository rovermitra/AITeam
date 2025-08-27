
import streamlit as st
st.set_page_config(layout="wide")  # MUST be first Streamlit command


import streamlit as st
import json
import os
from dotenv import load_dotenv
import openai
import google.generativeai as genai

# Import all the custom modules from your project structure
from agents.chat_agent import chat_with_ai, get_ai_suggested_destination
from agents.matcher import match_users
from agents.itinerary_planner import find_joint_itinerary
from components.chat_ui import render_chat_interface
from components.itinerary_display import display_itinerary

# --- 1. SETUP AND PAGE CONFIGURATION ---

import json
import os
from dotenv import load_dotenv
import openai
import google.generativeai as genai

# Load API keys from the .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Optional debug info
st.write("OpenAI key loaded:", openai.api_key is not None)
st.write("Google key loaded:", os.getenv("GOOGLE_API_KEY") is not None)
# CORRECT PLACEMENT: set_page_config() must be the first Streamlit command.

# Use Streamlit's caching to load data only once
@st.cache_data
def load_data():
    """Loads user and flight data from the JSON files."""
    try:
        with open('data/user_data.json', 'r') as f:
            all_users = json.load(f)
        with open('data/flight_data.json', 'r') as f:
            flight_data = json.load(f)
        return all_users, flight_data
    except FileNotFoundError:
        st.error("Error: Make sure `user_data.json` and `flight_data.json` are in the 'data' directory.")
        return [], []

all_users, flight_data = load_data()

# --- 2. SESSION STATE INITIALIZATION ---
# Initialize variables in the session state to preserve them across reruns

# For the chat
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# For the trip plan
if 'itinerary' not in st.session_state:
    st.session_state.itinerary = None
if 'suggested_destination' not in st.session_state:
    st.session_state.suggested_destination = None

# For user management
if 'main_user' not in st.session_state:
    st.session_state.main_user = None
if 'buddy_user' not in st.session_state:
    st.session_state.buddy_user = None
if 'matches' not in st.session_state:
    st.session_state.matches = []


# --- 3. UI RENDERING ---

st.title("✈️ Travel Buddy AI")
st.markdown("Find a travel partner and plan your next adventure together with the help of AI.")

# --- Sidebar for User Selection and Matching ---
with st.sidebar:
    st.header("Your Profile")
    if not all_users:
        st.warning("No user data found. Please check your `user_data.json` file.")
    else:
        user_names = [user['name'] for user in all_users]
        selected_user_name = st.selectbox("Select Your Profile", user_names, index=0)

        st.session_state.main_user = next((user for user in all_users if user['name'] == selected_user_name), None)

        if st.session_state.main_user:
            st.success(f"Welcome, **{st.session_state.main_user['name']}**!")

            st.header("Find a Buddy")
            if st.button("Match Me!", use_container_width=True):
                with st.spinner("Finding potential travel buddies..."):
                    st.session_state.matches = match_users(st.session_state.main_user, all_users)
                    st.session_state.buddy_user = None 

            if st.session_state.matches:
                match_names = [match['name'] for match in st.session_state.matches]
                selected_buddy_name = st.selectbox("Select a Buddy to Plan With", match_names)
                st.session_state.buddy_user = next((user for user in all_users if user['name'] == selected_buddy_name), None)
            else:
                st.info("Click 'Match Me!' to find travel partners.")
    
    # --- NEW: AI Model Selection ---
    st.markdown("---")
    st.header("Settings")
    model_display_name = st.radio(
        "Choose your Chat AI:",
        ('GPT-4o Mini (Fast)', 'Gemini Flash (Capable)'),
        index=0,
        help="Select the AI model that will act as your travel buddy in the chat."
    )
    # Convert the user-friendly name to the actual model ID
    st.session_state.model_choice = "gpt-4o-mini" if "GPT" in model_display_name else "gemini-1.5-flash"


# --- Main Content Area ---
if st.session_state.main_user and st.session_state.buddy_user:
    main_user = st.session_state.main_user
    buddy_user = st.session_state.buddy_user

    st.header(f"Planning a trip with {buddy_user['name']}")
    
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Trip Planner")
        
        if st.button("✨ Suggest Destination & Plan Trip!", use_container_width=True):
            with st.spinner("Our AI is finding the perfect spot for you..."):
                destination = get_ai_suggested_destination(main_user, buddy_user)
                st.session_state.suggested_destination = destination
                
                itinerary = find_joint_itinerary(main_user, buddy_user, destination, flight_data)
                st.session_state.itinerary = itinerary

        if st.session_state.suggested_destination:
            st.success(f"**AI Suggestion:** You and {buddy_user['name']} would love **{st.session_state.suggested_destination}**!")

        if st.session_state.itinerary:
            st.markdown("---")
            st.subheader("Your Joint Itinerary")
            
            if st.session_state.itinerary.get("main_user_flight"):
                st.markdown(f"**{main_user['name']}'s Flight:**")
                display_itinerary([st.session_state.itinerary["main_user_flight"]])
            else:
                st.warning(f"Could not find a suitable flight for {main_user['name']} to {st.session_state.suggested_destination}.")

            if st.session_state.itinerary.get("buddy_user_flight"):
                st.markdown(f"**{buddy_user['name']}'s Flight:**")
                display_itinerary([st.session_state.itinerary["buddy_user_flight"]])
            else:
                st.warning(f"Could not find a suitable flight for {buddy_user['name']} to {st.session_state.suggested_destination}.")
        else:
            st.info("Click the button above to generate a travel plan.")

    with col2:
        st.subheader(f"Chat with {buddy_user['name']}")
        
        render_chat_interface(st.session_state.chat_history)

        if user_input := st.chat_input("Say something to start planning..."):
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            with st.spinner(f"{buddy_user['name']} is typing..."):
                # --- UPDATED: Use the selected model from the sidebar ---
                ai_response = chat_with_ai(
                    user_input, 
                    st.session_state.chat_history, 
                    main_user, 
                    buddy_user, 
                    model_choice=st.session_state.model_choice
                )
                st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
            
            st.rerun()

else:
    st.info("⬅️ Please select your profile and a travel buddy from the sidebar to get started.")

