# RoverMitra Travel Planner — Dataset Guide

This README explains the **dataset structure** we use to feed the RoverMitra AI Travel Planner, how each field is used by the model, and how to test the whole flow with example **prompts** and **outputs**.

---

## Why this dataset?

Our planner needs enough context to:

1. **Choose a meeting plan** for people starting in different cities (e.g., meet at midpoint vs. at destination).
2. Build **door-to-door routes** under constraints (max transfers, time windows).
3. Propose **hotels and activities** that match pace, interests, budget, diet/access needs.
4. Publish a clear, bookable **day-by-day itinerary** into the **RoverMitra chat**.

We store that information as **one JSON object per trip request**:

* `trip_context`: Shared trip-level preferences and constraints.
* `travelers[]`: Per-person data (origins, budgets, preferences, constraints).
* `rovermitra_chat`: Where to post and discuss plans.

---

## File locations

* Generated groups file: `data/travel_group_requests.json` (1,000+ groups).
* Generator script: `generate_groups.py` (creates realistic mixed-quality data).

To (re)generate:

```bash
python generate_groups.py
# writes data/travel_group_requests.json
```

---

## Top-level schema (one group)

```json
{
  "group_id": "uuid",
  "rovermitra_chat": {
    "room_id": "rmr_xxxxxxxx",
    "created_at": "2025-08-10T12:00:00Z"
  },
  "trip_context": { ... },
  "travelers": [ ... ]
}
```

### Why each top-level field matters

* **group\_id** — stable ID for this request; used for updates/threads.
* **rovermitra\_chat** — allows the AI to post plans and revisions inside our product (no external messengers).
* **trip\_context** — shared trip goals & constraints; the “envelope” the itinerary must respect.
* **travelers\[]** — starting points & personal constraints to compute routes, costs, and compatibility.

---

## `trip_context` (shared trip-level fields)

```json
{
  "title": "Group-12 Trip",
  "destinations": ["Zurich", "Lucerne", "Interlaken"],
  "date_window": {
    "earliest_departure": "2025-09-15",
    "latest_return": "2025-09-22",
    "preferred_trip_length_days": 7,
    "blackout_dates": []
  },
  "meeting_strategy_allowed": ["en_route_midpoint", "at_destination", "origin_A", "origin_B"],
  "meeting_priority_objective": "minimize_total_travel_time_and_cost",
  "itinerary_style": "multi-stop",
  "min_time_per_stop_hours": 24,
  "luggage": { "carry_on_only": true, "special_gear": ["camera"] },
  "co2_preference": true,
  "tradeoff_weights": { "cost": 0.35, "time": 0.25, "comfort": 0.20, "scenery": 0.15, "co2": 0.05 },
  "hard_constraints": {
    "earliest_departure_time_local": "08:30",
    "latest_arrival_time_local": "21:30",
    "max_daily_travel_hours": 6,
    "max_transfers": 2,
    "room_setup": "twin"
  },
  "output_preferences": {
    "detail_level": "day-by-day",
    "include_booking_links": true,
    "currency": "EUR",
    "units": "metric",
    "share_to_rovermitra_chat": true
  }
}
```

**Field meanings (short):**

* **title** — human label for the trip.
* **destinations** — target cities/regions. The AI may choose a subset/sequence that fits constraints.
* **date\_window** — acceptable time bounds and preferred length; the AI schedules within this.
* **meeting\_strategy\_allowed** — where travelers are willing to meet (midpoint/en route/destination/one origin).
* **meeting\_priority\_objective** — meeting-point optimization rule (e.g., *minimize total time & cost*).
* **itinerary\_style** — *anchor\_city* (one base) or *multi-stop* (several bases).
* **min\_time\_per\_stop\_hours** — minimum dwell time so we don’t rush.
* **luggage** — carry-on only and special gear (affects transport and hotel choices).
* **co2\_preference** — push greener routes if ties are close.
* **tradeoff\_weights** — what to optimize globally (sum ≈ 1).
* **hard\_constraints** — strict caps (daily travel hours, transfers, time windows, room setup).
* **output\_preferences** — formatting for the chat; currency/units for prices/distances.

---

## `travelers[]` (per-person fields)

```json
{
  "name": "Traveler-12-0",
  "rovermitra_contact": { "channel": "RoverMitra", "user_handle": "rm_ab12cd34" },
  "home_base": {
    "city": "Berlin",
    "nearby_nodes": ["Berlin Hbf", "BER"],
    "willing_radius_km": 60
  },
  "age": 27,
  "gender": "Female",
  "education": "Master",
  "occupation": "Engineer",
  "marital_status": "Single",
  "learning_style": "Visual",
  "humor_style": "Witty",

  "budget": {
    "type": "total",
    "amount": 1200,
    "currency": "EUR",
    "split_rule": "each_own"
  },

  "transport": {
    "allowed_modes": ["train", "plane"],
    "max_transfers": 2,
    "max_leg_hours": 5,
    "night_travel_ok": false
  },

  "accommodation": {
    "types": ["hotel", "apartment"],
    "price_band": "mid-range",
    "room_setup": "twin",
    "must_haves": ["wifi", "near_station"]
  },

  "pace_and_interests": {
    "pace": "balanced",
    "top_interests": ["mountains", "lakes", "scenic trains", "photography"]
  },

  "diet_health": {
    "diet": "vegetarian",
    "allergies": [],
    "accessibility": []
  },

  "comfort": {
    "risk_tolerance": "low",
    "noise_tolerance": "low",
    "cleanliness_preference": "high"
  },

  "work": {
    "hours_online_needed": 1,
    "fixed_meetings": ["2025-09-18T17:00+02:00"],
    "wifi_quality_needed": "good"
  },

  "documents": {
    "passport_valid_until": "2030-03-15",
    "visa_status": "Schengen OK",
    "insurance": true
  },

  "languages": ["en", "de"],
  "dealbreakers": ["no red-eye travel", "no >2 transfers"]
}
```

**Why these matter:**

* **home\_base / nearby\_nodes** — starting points for routing (e.g., Berlin Hbf vs BER).
* **budget** — **per\_day** or **total**, with currency and cost-split rule.
* **transport** — allowed modes + transfer/time caps; the AI avoids itineraries that violate these.
* **accommodation** — types/price band/room setup/must-haves ensure realistic hotel picks.
* **pace\_and\_interests** — “packed” vs “relaxed” and what to do (mountains, museums, etc.).
* **diet\_health / comfort** — filters for restaurants, safety/risk, noise, cleanliness.
* **work** — when they must be online; keeps travel off those slots.
* **documents** — visa/insurance checks; alerts if invalid.
* **dealbreakers** — hard NOs (override softer preferences).

> Note: About **20–25%** of groups are intentionally “lean” (some optional fields missing) to simulate real-world input. The planner must handle missing data gracefully (ask follow-ups or assume defaults).

---

## Example: One real request (Berlin + Frankfurt → Switzerland)

**RoverMitra user message** (free text):

> “I’m in **Berlin** near **Berlin Hbf**. My friend is in **Frankfurt** near **Frankfurt Hbf**. We want a **7-day** Switzerland trip next month, **mid-range budget**, we prefer **trains**, and we’d like to **meet at a point that minimizes total time & cost**, then travel together. Interests: **mountains, lakes, short hikes, scenic trains, photography**. Please plan day-by-day with hotels near stations.”

**What we store** (simplified):

```json
{
  "group_id": "f3c7...4e",
  "rovermitra_chat": { "room_id": "rmr_8b91a2f3d4", "created_at": "2025-08-10T13:11:00Z" },
  "trip_context": {
    "title": "Alpine Week",
    "destinations": ["Zurich", "Lucerne", "Interlaken", "Zermatt"],
    "date_window": {
      "earliest_departure": "2025-09-15",
      "latest_return": "2025-09-22",
      "preferred_trip_length_days": 7,
      "blackout_dates": []
    },
    "meeting_strategy_allowed": ["en_route_midpoint","at_destination","origin_A","origin_B"],
    "meeting_priority_objective": "minimize_total_travel_time_and_cost",
    "itinerary_style": "multi-stop",
    "min_time_per_stop_hours": 24,
    "luggage": { "carry_on_only": true, "special_gear": ["camera"] },
    "co2_preference": true,
    "tradeoff_weights": { "cost": 0.35, "time": 0.25, "comfort": 0.20, "scenery": 0.15, "co2": 0.05 },
    "hard_constraints": {
      "earliest_departure_time_local": "08:30",
      "latest_arrival_time_local": "21:30",
      "max_daily_travel_hours": 6,
      "max_transfers": 2,
      "room_setup": "twin"
    },
    "output_preferences": { "detail_level": "day-by-day", "include_booking_links": true, "currency": "EUR", "units": "metric", "share_to_rovermitra_chat": true }
  },
  "travelers": [
    {
      "name": "Abdul",
      "rovermitra_contact": { "channel": "RoverMitra", "user_handle": "rm_abdul" },
      "home_base": { "city": "Berlin", "nearby_nodes": ["Berlin Hbf","BER"], "willing_radius_km": 60 },
      "budget": { "type": "total", "amount": 1200, "currency": "EUR", "split_rule": "each_own" },
      "transport": { "allowed_modes": ["train"], "max_transfers": 2, "max_leg_hours": 5, "night_travel_ok": false },
      "accommodation": { "types": ["hotel"], "price_band": "mid-range", "room_setup": "twin", "must_haves": ["wifi","near_station"] },
      "pace_and_interests": { "pace": "balanced", "top_interests": ["mountains","lakes","scenic trains","photography"] },
      "languages": ["en","de"],
      "documents": { "passport_valid_until": "2029-11-01", "visa_status": "Schengen OK", "insurance": true },
      "dealbreakers": ["no overnight trains"]
    },
    {
      "name": "Sara",
      "rovermitra_contact": { "channel": "RoverMitra", "user_handle": "rm_sara" },
      "home_base": { "city": "Frankfurt", "nearby_nodes": ["Frankfurt Hbf","FRA"], "willing_radius_km": 40 },
      "budget": { "type": "total", "amount": 1300, "currency": "EUR", "split_rule": "each_own" },
      "transport": { "allowed_modes": ["train"], "max_transfers": 2, "max_leg_hours": 4.5, "night_travel_ok": false },
      "accommodation": { "types": ["hotel"], "price_band": "mid-range", "room_setup": "twin", "must_haves": ["quiet_room","wifi"] },
      "pace_and_interests": { "pace": "balanced", "top_interests": ["lakes","short hikes","old towns","photography"] },
      "languages": ["en","de"],
      "documents": { "passport_valid_until": "2030-03-15", "visa_status": "Schengen OK", "insurance": true }
    }
  ]
}
```

---

## How the AI uses this

1. **Meeting point selection**

   * Evaluate train routes: Berlin→X, Frankfurt→X for candidate X ∈ {Basel, Zurich, Bern…}.
   * Optimize `meeting_priority_objective` with `tradeoff_weights` and `hard_constraints` (max transfers, time windows).
   * Choose **Basel SBB** or **Zurich HB** in many Germany→CH cases.

2. **Routing & timing**

   * Build **door-to-door** segments that respect earliest departure and latest arrival windows; ensure ≤ `max_transfers`, `max_daily_travel_hours`.

3. **Stay pattern**

   * If `itinerary_style=multi-stop`, allocate minimum `min_time_per_stop_hours` per base.
   * If carry-on only, avoid routes with luggage hassles.

4. **Hotels & activities**

   * Filter by `accommodation.price_band`, `room_setup`, `must_haves`, and near-station preference.
   * Choose sights & experiences aligned with `pace_and_interests` and **diet/health**.

5. **CO₂ preference & costs**

   * Prefer rail when differences are acceptable; show cost/time trade-offs.

6. **Output**

   * Post an **itinerary block** in the **RoverMitra room** with bookable links; handle edits iteratively.

---

## Example AI output (what teammates should expect)

**(posted in RoverMitra room `rmr_8b91a2f3d4`)**

```json
{
  "meeting_plan": {
    "meet_at": "Basel SBB",
    "rationale": "Minimizes combined travel time and cost by train within 2 transfers each."
  },
  "routes": [
    {
      "traveler": "Abdul",
      "segments": [
        {"from": "Berlin Hbf", "to": "Basel SBB", "mode": "train", "transfers": 2, "duration_h": 7.8, "depart_after": "08:30"}
      ]
    },
    {
      "traveler": "Sara",
      "segments": [
        {"from": "Frankfurt Hbf", "to": "Basel SBB", "mode": "train", "transfers": 0, "duration_h": 3.0}
      ]
    }
  ],
  "itinerary": [
    {"day": 1, "base": "Interlaken", "plan": "Arrive Basel, coffee buffer, continue together to Interlaken; evening lakeside walk."},
    {"day": 2, "base": "Interlaken", "plan": "Lauterbrunnen–Mürren loop; short hikes; photography spots."},
    {"day": 3, "base": "Interlaken", "plan": "Lake Brienz boat + Harder Kulm; sunset photos."},
    {"day": 4, "base": "Lucerne", "plan": "Scenic Brünig line; Old Town; Chapel Bridge."},
    {"day": 5, "base": "Lucerne", "plan": "Mount Rigi (easy summit) or Pilatus; return to Lucerne."},
    {"day": 6, "base": "Zurich", "plan": "Short transfer to Zurich; Bahnhofstrasse & ETH viewpoint."},
    {"day": 7, "base": "Zurich", "plan": "Depart home by train, within latest-arrival window."}
  ],
  "hotels": [
    {"city": "Interlaken", "near_station": true, "price_band": "mid-range", "room_setup": "twin", "must_haves": ["wifi","quiet_room"]},
    {"city": "Lucerne", "near_station": true, "price_band": "mid-range", "room_setup": "twin"},
    {"city": "Zurich", "near_station": true, "price_band": "mid-range", "room_setup": "twin"}
  ],
  "notes": [
    "All segments respect max 2 transfers and daily travel ≤ 6 hours.",
    "CO₂ preference honored by prioritizing rail.",
    "Booking links attached; adjust dates or budget in this thread to replan."
  ]
}
```

> The visual/chat rendering can be richer in-app (cards, buttons). The JSON above is the **logical payload** the backend can also log.

---

## Prompt template (for our planner agent)

Use this as a **system+user prompt** to the LLM when planning:

```
SYSTEM
You are RoverMitra’s expert travel-planning agent. Produce a realistic, bookable plan that:
- Chooses a meeting point using the group’s meeting_strategy_allowed and meeting_priority_objective,
- Optimizes using tradeoff_weights and respects hard_constraints,
- Builds door-to-door rail/air/bus segments within time windows and transfer caps,
- Selects hotels near transport nodes matching accommodation preferences,
- Proposes day-by-day activities aligned with pace_and_interests and diet/health,
- Prefers greener routes when co2_preference is true,
- Returns valid JSON with keys: meeting_plan, routes, itinerary, hotels, notes.

USER
Here is the trip request as JSON (trip_context + travelers + rovermitra_chat):

<PASTE GROUP JSON HERE>

Return only JSON (no prose).
```

---

## Common edge cases & how to read them

* **Single traveler** (group size = 1): The AI skips “meeting plan” and builds a solo door-to-door plan.
* **Missing optional fields** (lean profiles): The AI assumes defensible defaults (e.g., medium pace, rail for EU) or asks clarifying questions in the RoverMitra chat.
* **Conflicting constraints** (e.g., max\_transfers=0 but train-only to a remote spot): The AI posts alternatives with trade-offs.
* **Budget type**:

  * `per_day` → multiply by trip length for feasibility checks.
  * `total` → treat as cap for all segments + hotels + activities.

---

## Quick glossary

* **Meeting strategy**: where the group unifies (midpoint, at destination, one origin).
* **Anchor city**: single base with day trips.
* **Multi-stop**: multiple bases (move hotels).
* **Tradeoff weights**: scalar preferences for cost/time/comfort/scenery/CO₂.
* **Hard constraints**: strict limits the plan may **not** violate.

---

## TL;DR for coworkers

* The JSON gives the AI **all it needs** to:

  * pick a **meeting plan**,
  * route **door-to-door** within constraints,
  * book **hotels** that actually fit,
  * build a **day-by-day** that matches what people enjoy,
  * and post it neatly in the **RoverMitra chat**.

* When reading a request:

  1. Scan `trip_context` for dates, destinations, constraints.
  2. Check each traveler’s **home\_base**, **transport caps**, and **dealbreakers**.
  3. If something’s missing, the AI can **ask in the room** or assume defaults for a first draft.

That’s it—this is the contract between our product and the planning agent.
