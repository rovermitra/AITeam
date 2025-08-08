# Matchmaker - Ali Recommendation Feature

This branch introduces a new AI-powered travel matchmaking feature, allowing users to create detailed travel profiles and find compatible travel buddies based on shared interests, location, language, and personality traits.

---

## Features

- **User Profile Creation**: Input name, age, gender, location, interests, languages, and personality traits (on a scale of 1-10).
- **AI-Powered Matching**: Uses OpenAI’s GPT-4o-mini model to find the best matches from existing profiles.
- **Robust JSON parsing and validation** for smooth LLM response handling.
- **Simple & appealing web UI** built with Flask for interactive profile creation and match display.
- **Secure handling of OpenAI API key** via `.env` (ensure `.env` is in `.gitignore`).

---

## Getting Started

### Prerequisites

- Python 3.9+
- OpenAI API key
- Required Python packages (see `requirements.txt`)

### Setup

1. Clone the repo and checkout this branch:

    ```bash
    git clone https://github.com/rovermitra/Matchmaker.git
    cd Matchmaker
    git checkout ali_recommendation
    ```

2. Create and activate a Python virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    venv\Scripts\activate     # Windows
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Create a `.env` file in the root directory and add your OpenAI API key:

    ```env
    OPENAI_API_KEY=your_openai_api_key_here
    ```

### Run the app

```bash
python app.py
