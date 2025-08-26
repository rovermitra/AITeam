# In agents/chat_agent.py

import openai

def chat_with_ai(user_input: str, chat_history: list, main_user: dict, buddy_user: dict):
    """
    Generates an AI response based on conversation history and a defined persona.
    """
    
    # --- Create the Persona (System Prompt) ---
    # This is the key instruction that gives the AI its personality and goals.
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
    - Ask questions, but also contribute your own ideas. For example, if they suggest a place, you can say what you'd like to do there based on your interests.
    - Make the conversation feel like a real collaboration between two friends.
    """

    # --- Construct the messages payload for the API ---
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Sorry, I'm having a little trouble connecting right now. Let's try again in a moment."