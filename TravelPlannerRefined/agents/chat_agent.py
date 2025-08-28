import openai
import google.generativeai as genai
import os
import json

# Configure clients at the start
openai.api_key = os.getenv("OPENAI_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def generate_response(chat_history: list, system_prompt: str, model_name: str = "gpt-4o-mini"):
    """
    Generates an AI response from a chosen model with improved error handling.
    """
    try:
        if model_name.startswith("gemini"):
            gemini_history = []
            for msg in chat_history:
                role = "user" if msg["role"] == "user" else "model"
                gemini_history.append({"role": role, "parts": [{"text": msg["content"]}]})
            
            model = genai.GenerativeModel(model_name, system_instruction=system_prompt)
            response = model.generate_content(gemini_history, generation_config={"temperature": 0.75})

            if response.text:
                return response.text
            else:
                print("Gemini response was empty or blocked. Feedback:", response.prompt_feedback)
                return "I'm not sure how to respond to that. Could we talk about something else?"

        else:  # Default to OpenAI
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(chat_history)
            
            # Use the modern, correct API call
            response = openai.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.75
            )

            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content
            else:
                print("OpenAI response was empty.")
                return "I seem to be at a loss for words. Let's try a different topic."

    except Exception as e:
        print(f"An unexpected error occurred with {model_name}: {e}")
        return "Sorry, I'm having a little trouble connecting right now. Let's try again in a moment."


def chat_with_ai(user_input: str, chat_history: list, main_user: dict, buddy_user: dict, flight_data: list, model_choice: str = "gpt-4o-mini"):
    """
    Generates a context-aware AI response using chat history, user profiles, and flight data.
    """
    # Filter flight data to only what's relevant to the users' origins
    relevant_flights = [
        f for f in flight_data
        if f["from"] in [main_user["origin"], buddy_user["origin"]]
    ]

    # Build the system prompt with data and a strong persona example
    system_prompt = f"""
    You are {buddy_user['name']}, a travel enthusiast. Your personality is friendly, curious, and adventurous. You are chatting with {main_user['name']} to plan a trip.

    YOUR PROFILE:
    - Interests: {', '.join(buddy_user['preferences']['activities'])}
    - Budget: ${buddy_user['budget']}
    - Origin: {buddy_user['origin']}

    {main_user['name'].upper()}'S PROFILE:
    - Interests: {', '.join(main_user['preferences']['activities'])}
    - Budget: ${main_user['budget']}
    - Origin: {main_user['origin']}

    AVAILABLE FLIGHTS (A simplified list for context):
    {json.dumps(relevant_flights, indent=2)}

    YOUR TASK:
    - Have a natural, two-way conversation. DO NOT act like a generic AI.
    - Use the user profiles and flight data to make smart suggestions.
    - If the user asks about flights, use the data above to answer.
    - Share your own opinions and ask questions based on YOUR profile.

    ---
    HERE IS AN EXAMPLE OF HOW YOU SHOULD TALK:

    {main_user['name']}: Hey! I was thinking we could go to Italy. I've always wanted to see the Colosseum.
    You: Italy sounds amazing! I'm totally in. Besides the Colosseum, I'd love to try some authentic pasta making, since I'm really into cooking. What do you think?
    ---
    
    Now, continue your conversation with {main_user['name']}.
    """

    # Create a temporary history for the API call to include the latest user message
    current_chat_history = chat_history + [{"role": "user", "content": user_input}]
    
    # Generate AI response
    response_text = generate_response(current_chat_history, system_prompt, model_name=model_choice)

    return response_text


def get_ai_suggested_destination(user1: dict, user2: dict):
    """
    Suggests the best travel destination for two users using Gemini.
    """
    model = genai.GenerativeModel('gemini-1.5-flash')

    prompt = f"""
    You are an expert travel agent. Analyze the two user profiles below.
    Suggest the single best city for them to visit together based on shared interests and destinations.

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

    Respond with ONLY the name of the city. For example: Rome
    """

    try:
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.2}
        )
        return response.text.strip()
    except Exception as e:
        print(f"An error occurred during Gemini suggestion: {e}")
        # Fallback to a common destination
        common = set(user1['destinations']) & set(user2['destinations'])
        return list(common)[0] if common else "Paris"
