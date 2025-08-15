import os
import json
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash
from dotenv import load_dotenv
import openai
from utils import safe_write_json, llm_find_matches

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
app.secret_key = os.urandom(24)

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE_PATH = os.path.join(BASE_DIR, "data", "travel_ready_user_profiles.json")
CHAT_FILE_PATH = os.path.join(BASE_DIR, "data", "chat_history.json")

# --- Helper functions ---
def read_float(value, default=5.0):
    try:
        f = float(value)
        if 1 <= f <= 10:
            return f
    except:
        pass
    return default

def create_user_from_form(form):
    user = {"personality": {}, "id": str(uuid.uuid4())}
    user['name'] = (form.get('name') or '').strip()
    try:
        user['age'] = int((form.get('age') or '0').strip())
    except ValueError:
        user['age'] = 0
    user['gender'] = (form.get('gender') or '').strip()
    user['location'] = (form.get('location') or '').strip()
    user['country'] = (form.get('country') or '').strip()
    user['occupation'] = (form.get('occupation') or '').strip()
    user['interests'] = [i.strip() for i in (form.get('interests') or '').split(',') if i.strip()]
    user['languages'] = [l.strip() for l in (form.get('languages') or '').split(',') if l.strip()]
    user['bio'] = (form.get('bio') or '').strip()
    user['personality']['openness'] = read_float(form.get('openness'), 5.0) / 10.0
    user['personality']['extraversion'] = read_float(form.get('extraversion'), 5.0) / 10.0
    user['personality']['agreeableness'] = read_float(form.get('agreeableness'), 5.0) / 10.0
    user['personality']['conscientiousness'] = read_float(form.get('conscientiousness'), 5.0) / 10.0
    return user

def load_db():
    try:
        with open(DB_FILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def load_chats():
    try:
        with open(CHAT_FILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_chats(chats):
    safe_write_json(chats, CHAT_FILE_PATH)

# --- Routes ---
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user = create_user_from_form(request.form)

        # Save user to DB
        db_profiles = load_db()
        db_profiles.append(user)
        safe_write_json(db_profiles, DB_FILE_PATH)

        # Redirect to matches page
        return redirect(url_for("show_matches", user_id=user['id']))

    return render_template("index.html")

@app.route("/matches/<user_id>")
def show_matches(user_id):
    db_profiles = load_db()

    # Ensure all users have an 'id' (fix old data)
    for u in db_profiles:
        if 'id' not in u:
            u['id'] = str(uuid.uuid4())
    safe_write_json(db_profiles, DB_FILE_PATH)  # persist fixes

    user = next((u for u in db_profiles if u['id'] == user_id), None)
    if not user:
        flash("User not found.", "danger")
        return redirect(url_for("index"))

    candidates = [p for p in db_profiles if p['id'] != user['id']]
    matches = []
    if candidates:
        try:
            matches = llm_find_matches(user, candidates)

            # Ensure each match keeps its ID for connect link
            for i, match in enumerate(matches):
                if 'id' not in match and i < len(candidates):
                    match['id'] = candidates[i]['id']

        except Exception as e:
            flash(f"Error while finding matches: {e}", "danger")

    return render_template("matches.html", user=user, matches=matches)


@app.route("/connect/<user_id>/<match_id>")
def connect(user_id, match_id):
    # Create a new chat
    chat_id = str(uuid.uuid4())
    chats = load_chats()
    chats[chat_id] = {"users": [user_id, match_id], "messages": []}
    save_chats(chats)

    return redirect(url_for("chat", chat_id=chat_id))

@app.route("/chat/<chat_id>", methods=["GET", "POST"])
def chat(chat_id):
    chats = load_chats()
    if chat_id not in chats:
        flash("Chat not found.", "danger")
        return redirect(url_for("index"))

    chat_data = chats[chat_id]

    if request.method == "POST":
        sender = request.form.get("sender")
        message = request.form.get("message")
        if sender and message:
            chat_data["messages"].append({"sender": sender, "message": message})
            save_chats(chats)
        return redirect(url_for("chat", chat_id=chat_id))

    return render_template("chat.html", chat_id=chat_id, chat=chat_data)

# --- Run app ---
if __name__ == "__main__":
    os.makedirs(os.path.dirname(DB_FILE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(CHAT_FILE_PATH), exist_ok=True)
    app.run(debug=True)
