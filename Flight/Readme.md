# RoverMitra — Travel Group Generator & Inventory (v2)

This repo contains a **synthetic but realistic dataset generator** for RoverMitra’s travel-planning AI. It produces group travel requests plus draft inventories (flights, trains, hotels, restaurants, activities) that your LLM/agents can turn into **bookable, day‑by‑day itineraries**.

---

## What this script does

* Creates **N\_GROUPS** (default 1000) travel groups with:

  * **travelers** (size 1–15, realistic distribution)
  * **trip\_context** (destinations, dates, constraints, preferences)
  * **draft\_plan** inventories (flight/train offers, hotel/restaurant/activity options) and a skeleton day-by-day plan
* Ensures **city ↔ country** pairs are valid and uses plausible **airport codes**
* Mixes **rich** profiles (75%) and **lean** profiles (25%) to test agent robustness
* Writes a single JSON file (default: `data/travel_group_requests_with_inventory_v2.json`)

---

## Quick start

1. Open the script and tweak:

   * `N_GROUPS` (e.g., 10\_000 for a big corpus)
   * `RICH_GROUP_RATIO` (0.75 = 75% rich)
   * `OUT_PATH` (output location)
2. Run the script (Python 3.9+):

   ```bash
   python generate_travel_groups_v2.py
   ```
3. Inspect the output:

   ```python
   import json
   data = json.load(open("data/travel_group_requests_with_inventory_v2.json", "r", encoding="utf-8"))
   print(len(data), "groups")
   print(data[0].keys())
   ```

---

## File structure (top level)

Each **group** object contains:

```json
{
  "group_id": "uuid",
  "rovermitra_chat": {"room_id": "rmr_xxx", "created_at": "ISO8601Z"},
  "trip_context": { /* preferences & constraints */ },
  "travelers": [ /* 1..15 traveler objects */ ],
  "draft_plan": { /* offers + skeleton itinerary */ }
}
```

### `rovermitra_chat`

* `room_id`: the internal chat space where the agent will post options & confirmations
* `created_at`: creation timestamp for ordering/history

---

## `trip_context` (what the planner must honor)

Key fields your planning agent should read before generating an itinerary:

* **destinations**: `["Zurich", "Interlaken", ...]` (unique list)
* **date\_window**:

  * `earliest_departure` / `latest_return` (ISO dates)
  * `preferred_trip_length_days` (hint; not hard constraint)
  * `blackout_dates` (hard avoid)
* **meeting\_strategy\_allowed**: e.g., `at_destination`, `en_route_midpoint`, `origin_A` — how people meet
* **meeting\_priority\_objective**: minimize cost / time / both
* **itinerary\_style**: `anchor_city` or `multi-stop`
* **min\_time\_per\_stop\_hours**: do not schedule fly‑by stops
* **luggage**: carry‑on only? special gear (camera, skis, poles…)
* **co2\_preference**: whether eco impact should influence choices
* **tradeoff\_weights**: normalized weights `{cost, time, comfort, scenery, co2}`
* **hard\_constraints**: time windows, `max_daily_travel_hours`, `max_transfers`, `room_setup`
* **output\_preferences**: detail level, currency, metric/imperial, post to chat

> **Tip:** For intercontinental hops, your agent may choose to temporarily relax `max_daily_travel_hours` but must call it out in the plan.

---

## `travelers` (who is going)

Each traveler includes:

* **home\_base**: `{city, nearby_nodes:["<City> Hbf", "IATA"], willing_radius_km}` → helps choose meet points
* **demographics**: `age`, `gender`, `education`, `occupation`, `marital_status`
* **styles**: `learning_style`, `humor_style`, `pace_and_interests.pace` (`relaxed|balanced|packed`)
* **interests**: curated set like `mountains`, `museums`, `street food`, `photography`, …
* **budget**: `{type: per_day|total, amount, currency, split_rule}`
* **transport**: `{allowed_modes, max_transfers, max_leg_hours, night_travel_ok}`
* **accommodation**: `{types, price_band, room_setup, must_haves}`
* **diet\_health**: dietary tags, allergies, accessibility
* **comfort**: `{risk_tolerance, noise_tolerance, cleanliness_preference}`
* **work**: Wi‑Fi needs, online hours
* **documents**: passport validity, visa status, insurance flag
* **languages**: subset of country + English
* **dealbreakers**: e.g., *no party hostels*, *no red‑eye*

> **Lean profiles** randomly omit some fields to test fallback behavior (e.g., infer price band from cohort).

---

## `draft_plan` (offers your agent can reason over)

* **meeting\_plan**: chosen meet city + rationale
* **itinerary**: day-by-day skeleton `{date, base, plan}` → agent expands with booked items
* **intercity\_ground\_offers**: trains between consecutive destinations, with realistic durations & CO₂
* **flight\_offers**: for **each traveler**, candidate outbound & return flights (IATA times, durations, cabin, bag, CO₂, price)
* **chosen\_flights**: the cheapest pair per traveler (baseline the agent can override)
* **hotel\_inventory**: per city, 4–7 options (stars, price, distance to station, amenities)
* **hotel\_reservations**: first-pass holds (status `hold`)
* **restaurant\_inventory**: per city, 6–10 options (cuisine, price, dietary tags)
* **restaurant\_reservations**: 19:30 holds (status `hold`)
* **activities\_inventory**: 3–6 options aligned to group interests (viewpoints, hikes, museums…)
* **hints**: simple seasonal/safety nudges

---

## Realism guarantees

* Cities always match their countries; IATA codes are plausible
* Languages seeded from country + English
* Budgets, star levels, distances, durations, and CO₂ are within human‑plausible ranges
* Group sizes follow a realistic distribution (solo and pairs most common)

---

## Rich vs Lean

* **Rich**: full preferences + constraints → ideal for training the “happy path”
* **Lean**: 20–25% of fields missing or simplified → ideal for testing inference, defaults, and LLM recovery

---

## Using this with LLM/agents

Your pipeline typically looks like:

1. **Grounding**: read a group → summarize constraints & preferences per traveler
2. **Meetpoint search**: if origins differ, consider `en_route_midpoint` vs `at_destination` against tradeoff weights
3. **Routing**: pick long‑haul flights and intercity trains satisfying `hard_constraints`
4. **Lodging**: select hotels by `price_band`, distance to station, and `must_haves`
5. **Dining**: match dietary tags
6. **Activities**: pick options aligned with shared interests & pace
7. **Explain**: produce a short rationale tied to weights & constraints
8. **Post** to `rovermitra_chat.room_id` with proposed plan & book/confirm actions

### Example LLM prompt (trip planner)

> **System**: You are RoverMitra’s trip‑planning agent. Read the group JSON and propose an itinerary that maximizes the given tradeoff weights while respecting hard constraints. If a constraint must be relaxed (e.g., intercontinental flight > max\_daily\_travel\_hours), flag the exception explicitly and justify.
>
> **User**: Here’s our group JSON. Suggest the best meet strategy, flights, hotels, and a day‑by‑day plan. Keep the tone friendly and actionable. Include 2–3 alternatives only where meaningful.

### Example user ask → friendly answer

**User**: “I’m in **Berlin (BER)**, my friend is in **Frankfurt (FRA)**. We want to travel **Apr 8–14** to **Switzerland** with carry‑on only, relaxed pace, and a budget hotel near the station. Please plan and show the best meet point.”

**Agent**: “Great choice! **Meet in Zurich** (shortest combined travel time). Fly **BER→ZRH** for you and **FRA→ZRH** for your friend, both arriving before **16:00**. I’ve held **Zurich Central Hotel (3★, 450m to HB)**. Day‑by‑day: Day 1 Old Town + lake sunset; Day 2 Lucerne day trip (scenic rail); Day 3 Uetliberg viewpoints + fondue; Day 4 Rhine Falls stopover; Day 5 checkout and return flights ≤ 20:00. All within your carry‑on and budget settings—want me to book?”

---

## Data dictionary (selected)

**Traveler**

* `home_base.city` — origin city (valid in catalog)
* `home_base.nearby_nodes` — major station + nearest airport
* `budget.amount` — per‑day or total budget per person (currency inferred by country)
* `transport.allowed_modes` — subset of `[train, plane, bus]`
* `accommodation.must_haves` — features your ranker should respect (e.g., `near_station`, `wifi`)
* `diet_health.diet` — dietary anchor for restaurants
* `dealbreakers` — hard filters

**Trip Context**

* `meeting_strategy_allowed` — how they’re willing to meet
* `tradeoff_weights` — objective function (sum≈1.0)
* `hard_constraints.max_daily_travel_hours` — apply to ground hops; long‑hauls can be exception‑flagged

**Draft Plan**

* `flight_offers` — candidates per traveler
* `chosen_flights` — baseline selection (cheapest)
* `hotel_inventory.options[].distance_to_station_km` — good proxy for convenience
* `restaurant_inventory` & `activities_inventory` — content to personalize day plans

---

## Extending to live providers

Replace generators with API adapters:

* Flights: Skyscanner/Amadeus/Sabre (IATA times, bag, CO₂)
* Rail: RailEurope/Deutsche Bahn APIs
* Hotels: Booking/Amadeus/Lodgify
* Dining: Google Places/Yelp/HappyCow (for diet tags)
* Activities: Viator/GetYourGuide/AYR

Maintain the **same JSON shape** so your LLM prompts remain stable.

---

## Troubleshooting

* **Empty/odd city**: ensure the city exists in `COUNTRIES` and has at least one IATA code
* **Huge outputs**: reduce `N_GROUPS`, or sample groups before feeding to LLM
* **Token limits**: sample candidate inventories (e.g., 3 flights, 3 hotels) before sending to the model
* **Date math** off by one: remember `latest_return` is exclusive vs inclusive depending on your UI semantics

---

## License & attribution

Synthetic data for internal testing at **RoverMitra**. Airport codes, durations, prices and CO₂ are plausible but **not real**. Replace with live data sources for production.

---

## Changelog

* **v2**: Added flight/train/hotel/restaurant/activity inventories; expanded countries; improved realism; rich/lean profiles; day-by-day skeleton.
