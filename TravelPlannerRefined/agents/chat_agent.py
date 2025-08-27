# agents/chat_agent.py

import openai
import google.generativeai as genai
import os
from dotenv import load_dotenv
import google.generativeai as genai


# Load API keys from the .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Your existing generate_response function...
def generate_response(chat_history: list, system_prompt: str, model_name: str = "gpt-4o-mini"):
    """
    Generates an AI response from a chosen model.
    """
    # ... (the code for this function remains the same)
    if model_name.startswith("gemini"):
        # Gemini uses a different format for roles ("user" and "model")
        gemini_history = []
        for msg in chat_history:
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [{"text": msg["content"]}]})
        
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(gemini_history, generation_config={"temperature": 0.7})
            return response.text
        except Exception as e:
            print(f"An error occurred with Gemini: {e}")
            return "Sorry, I'm having a little trouble connecting right now."

    else: # Default to OpenAI
        # ... (OpenAI logic remains the same)
        pass # Placeholder for your existing OpenAI code

# Your existing chat_with_ai function...
def chat_with_ai(user_input: str, chat_history: list, main_user: dict, buddy_user: dict, model_choice: str = "gpt-4o-mini"):
    """
    Generates an AI response based on conversation history and a defined persona.
    Supports multiple models via `model_choice`.
    """

    # --- Create the Persona (System Prompt) ---
    system_prompt = f"""
    You are {buddy_user['name']}, a travel enthusiast. Your personality is friendly, curious, and adventurous.
    You are currently chatting with {main_user['name']} to plan a trip together.

    Your profile details are:
    - Your Interests: {', '.join(buddy_user['preferences']['activities'])} 
    - Your Travel Preferences: Airline: {buddy_user['preferences']['airline']}, Class: {buddy_user['preferences']['class']}
    - Your Budget: {buddy_user['budget']}
    - Your Origin: {buddy_user['origin']}

    Your goal is to have a natural, two-way conversation with {main_user['name']}.
    - DO NOT act like a generic AI assistant. Act as {buddy_user['name']}.
    - Share your own opinions and suggestions based on YOUR profile.
    - Ask questions, but also contribute your own ideas.
    - Make the conversation feel like a real collaboration between two friends.
    """

    # --- Construct the messages payload for the API ---
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)

    try:
        if model_choice.startswith("gemini"):
            # Gemini-specific formatting
            gemini_history = []
            for msg in messages:
                role = "user" if msg["role"] == "user" else "model"
                gemini_history.append({"role": role, "parts": [{"text": msg["content"]}]})

            model = genai.GenerativeModel(model_choice)
            response = model.generate_content(
                gemini_history,
                generation_config={"temperature": 0.7}
            )
            return response.text

        else:
            # OpenAI (default)
            response = openai.chat.completions.create(
                model=model_choice,
                messages=messages,
                temperature=0.7
            )
            return response.choices[0].message.content

    except Exception as e:
        print(f"An error occurred: {e}")
        return "Sorry, I'm having a little trouble connecting right now. Let's try again in a moment."


# -----------------------------------------------------------------
# ** NEW FUNCTION TO ADD **
# -----------------------------------------------------------------
def get_ai_suggested_destination(user1: dict, user2: dict):
    """
    Asks Gemini to analyze two user profiles and suggest the best travel destination.
    """
    model = genai.GenerativeModel('gemini-1.5-flash') # Using a fast and capable model

    prompt = f"""
    You are an expert travel agent. Analyze the two user profiles below.
    Based on their shared interests, preferences, and desired destinations, suggest the single best city for them to visit together.

    User A:
    - Name: {user1['name']}
    - Interests: {', '.join(user1['preferences']['activities'])}
    - Desired Destinations: {', '.join(user1['destinations'])}
    - Budget: ${user1['budget']}

    User B:
    - Name: {user2['name']}
    - Interests: {', '.join(user2['preferences']['activities'])}
    - Desired Destinations: {', '.join(user2['destinations'])}
    - Budget: ${user2['budget']}

    Suggest one city that would be a great fit for both of them.
    Respond with ONLY the name of the city and nothing else. For example: Rome
    """

    try:
        response = model.generate_content(
            prompt,
            # Use a low temperature for more deterministic, less "creative" answers
            generation_config={"temperature": 0.2} 
        )
        # Clean up the response to ensure it's just the city name
        suggested_city = response.text.strip()
        return suggested_city
    except Exception as e:
        print(f"An error occurred during Gemini suggestion: {e}")
        # Fallback to a common destination if AI fails
        common = set(user1['destinations']) & set(user2['destinations'])
        return list(common)[0] if common else "Paris"