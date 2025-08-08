# Travel Matchmaker

A command-line tool to create travel user profiles and find compatible travel companions using OpenAI's GPT-4o-mini model.

---

## Features

- Create and save travel profiles with personality traits, interests, and demographics.
- Prefilter candidate profiles based on shared interests and languages.
- Query the OpenAI API using function-calling for reliable structured JSON responses.
- Parse and display top matches with explanations and compatibility scores.
- Includes unit tests for parsing logic.

---

## Setup

1. Clone the repo:
   ```bash
   git clone <repo-url>
   cd Matchmaker
   
   
2. Install Dependecies
	```
	pip install -r requirements.txt

	```
	

3. Create a .env file in the project root with your OpenAI API key:
	```
	OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxx
	```
	
	
Usage
Run the main script:

	```
	python main.py

	```
	
Tests
Run tests with:
	```
	pytest tests/
	```
	
	
File structure
main.py: Main interactive matchmaking script.

utils.py: Utility functions (JSON parsing, file operations).

data/: JSON profile databases.

tests/: Unit tests for utilities.

.env: Environment variables (API key).

requirements.txt: Python dependencies.

# Dependencies

* Python 3.8+
* openai

* python-dotenv

* pytest (for tests)