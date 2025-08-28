🌍 Travel Buddy AI
Travel Buddy AI is a smart Streamlit application designed to match users with compatible travel partners. It facilitates trip planning through an interactive chat interface powered by an AI persona and generates personalized flight itineraries based on user preferences and conversations.

```
streamlit run app.py
```


✨ Key Features
AI-Powered Chat: Engage in a natural, two-way conversation with an AI travel buddy that has its own unique personality, interests, and travel preferences.

User Matching: The application loads profiles for a main user and a matched travel buddy.

Instant Itinerary Generation: Instantly generate a flight itinerary based on your pre-defined travel preferences.

Conversational Planning: Discuss and refine your travel plans with your AI buddy, then generate a flight itinerary based on the insights from your chat.

Clean & Interactive UI: A simple and intuitive web interface built with Streamlit, making it easy to navigate and use.

📂 Project Structure
The project is organized into distinct modules for agents, UI components, and data, ensuring a clean and scalable architecture.

```

travel_buddy_ai/
│
├── app.py                  # Main Streamlit application
├── data/
│   ├── flight_data.json    # Sample flight information
│   └── user_data.json      # User profiles for matching
├── agents/
│   ├── __init__.py
│   ├── itinerary_planner.py# Logic for finding the best flights
│   └── chat_agent.py       # AI chat logic with persona generation
├── components/
│   ├── chat_ui.py          # Functions to render the chat interface
│   └── itinerary_display.py# Functions to display the flight itinerary
├── .env                    # For storing the OpenAI API key
└── requirements.txt        # Required Python packages

```

🚀 Getting Started
Follow these steps to set up and run the project on your local machine.

1. Set up your environment variables:

Create a file named .env in the root directory of the project.

Add your OpenAI API key to this file:

```
OPENAI_API_KEY="your_openai_api_key_here"
```

3. Running the Application
Once the setup is complete, you can run the Streamlit application with the following command:

```
streamlit run app.py
```
Your web browser should automatically open to the application's URL (usually http://localhost:8501).

💡 How to Use
View Profiles: The sidebar on the left displays your profile and the profile of your matched travel buddy.

Plan Flights Directly: Click the "✈️ Plan Flights Now" button to get an immediate flight itinerary based on your profile's preferences.

Chat with Your Buddy:

Click the "💬 Chat With Buddy" button to open the chat interface.

Type your message and have a conversation. The AI will respond as your buddy, sharing its own opinions and ideas.

After chatting, click the "✨ Generate Itinerary from Chat" button to create a travel plan that incorporates the discussion.