import os
import json
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash
from dotenv import load_dotenv
import openai
from utils import parse_llm_json, safe_write_json, prefilter_candidates, build_llm_prompt, llm_find_matches

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
app.secret_key = os.urandom(24)  # needed for flashing messages

DB_FILE_PATH = "data/travel_ready_user_profiles.json"


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
    user['name'] = form.get('name', '').strip()
    user['age'] = int(form.get('age', 0))
    user['gender'] = form.get('gender', '').strip()
    user['location'] = form.get('location', '').strip()
    user['country'] = form.get('country', '').strip()
    user['occupation'] = form.get('occupation', '').strip()
    user['interests'] = [i.strip() for i in form.get('interests', '').split(',') if i.strip()]
    user['languages'] = [l.strip() for l in form.get('languages', '').split(',') if l.strip()]
    user['bio'] = form.get('bio', '').strip()
    user['personality']['openness'] = read_float(form.get('openness'), 5.0) / 10.0
    user['personality']['extraversion'] = read_float(form.get('extraversion'), 5.0) / 10.0
    user['personality']['agreeableness'] = read_float(form.get('agreeableness'), 5.0) / 10.0
    user['personality']['conscientiousness'] = read_float(form.get('conscientiousness'), 5.0) / 10.0
    return user


@app.route("/", methods=["GET", "POST"])
def index():
    matches = None
    if request.method == "POST":
        user = create_user_from_form(request.form)
        # Add user to DB
        try:
            with open(DB_FILE_PATH, "r", encoding="utf-8") as f:
                db_profiles = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            db_profiles = []

        db_profiles.append(user)
        safe_write_json(db_profiles, DB_FILE_PATH)

        # Prepare candidates (excluding current user)
        candidates = [p for p in db_profiles if p.get('id') != user['id']]
        filtered_candidates = prefilter_candidates(user, candidates, max_results=5)

        if not filtered_candidates:
            flash("No suitable candidates found in the database for matching.", "warning")
            return render_template("index.html", user=user)

        try:
            matches = llm_find_matches(user, filtered_candidates)
        except Exception as e:
            flash(f"Error while finding matches: {e}", "danger")

        return render_template("index.html", user=user, matches=matches)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
