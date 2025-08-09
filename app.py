import os
import json
import uuid
from flask import Flask, render_template, request, flash
from dotenv import load_dotenv
import openai
from utils import safe_write_json, llm_find_matches

load_dotenv()
openai.api_key = os.getenv("API")

app = Flask(__name__)
app.secret_key = os.urandom(24)

# --- Make the DB path absolute so it ALWAYS points to your file ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE_PATH = os.path.join(BASE_DIR, "data", "travel_ready_user_profiles.json")

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
    # be safe with age
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

@app.route("/", methods=["GET", "POST"])
def index():
    matches = None
    if request.method == "POST":
        user = create_user_from_form(request.form)

        # 1) Load
        db_profiles = load_db()
        before_count = len(db_profiles)

        # 2) Append + write
        db_profiles.append(user)
        safe_write_json(db_profiles, DB_FILE_PATH)

        # 3) Verify it actually persisted
        after = load_db()
        after_count = len(after)
        if after_count != before_count + 1:
            flash("⚠️ Failed to persist the new profile. Check file permissions/path.", "danger")
            return render_template("index.html", user=user)

        # Send ALL others to LLM (no prefilter)
        candidates = [p for p in after if p.get('id') != user['id']]
        if not candidates:
            flash("No candidates available in the database.", "warning")
            return render_template("index.html", user=user)

        try:
            matches = llm_find_matches(user, candidates)
            if not matches:
                flash("AI returned no matches ≥ 75% right now.", "info")
        except Exception as e:
            flash(f"Error while finding matches: {e}", "danger")

        return render_template("index.html", user=user, matches=matches)

    return render_template("index.html")

if __name__ == "__main__":
    os.makedirs(os.path.dirname(DB_FILE_PATH), exist_ok=True)
    app.run(debug=True)
