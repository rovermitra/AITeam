# RoverMitra Synthetic Data Pipeline – README

A single, end‑to‑end, reproducible pipeline that generates realistic **seed data** for RoverMitra’s Trip Planner stack. All datasets are **linked via stable IDs** and avoid copying user fields across domains (“single source of truth”).

> **Quick start**
>
> ```bash
> # from repo root
> python3 Scripts/user_data_generator.py
> python3 Scripts/matchmaker_data_generator.py
> python3 Scripts/Flight_data_generator.py
> python3 Scripts/restaurant_data_generator.py
> python3 Scripts/activities_events_generator.py
> python3 Scripts/car_rental_data_generator.py
> ```
>
> Each script prints where it wrote files and a short preview of the output.

---

## 1) Repository layout

```
/Users/data/                     # user single-source data (authoritative)
/MatchMaker/data/                # derived match profiles (preferences)
/Flight/data/                    # integrated group trips w/ transport & hotels
/Restaurants/data/               # restaurant catalogs, availability, group holds
/Activities/data/                # attractions, local passes, events & festival holds
/Rentals/data/                   # car suppliers, availability, group rental holds
/Scripts/                        # all generators (run from repo root)
```

---

## 2) ID & Linking model (do not duplicate user fields)

* **user\_id**: canonical ID created in `users_core.json` (e.g., `u_4f2a...`). Used everywhere else to refer to a person.
* **match\_profile\_id**: created in matchmaker; links back with `user_id`.
* **group\_id**: created in Flight generator; represents a travel group/room.
* **Inventory IDs**:

  * Flights: `flt_*` (offer rows), Trains: `rail_*`, Hotels: `hot_*`, Restaurants: `rst_*`, Activities: `act_*`, Rentals: `rent_*`.
* **Reservation/Hold IDs**: `resv_*` for tables, `cr_*` for car rentals, `actres_*` for activities/events.

> **Rule**: Only **users/data/users\_core.json** contains personal fields (name, age, etc). All other files only store IDs + minimal operational data (e.g., handle, budget band is read from `users_core` when needed and *not* duplicated).

---

## 3) Data flow (generation order)

1. **Users** → `Scripts/user_data_generator.py`

   * **Output:** `users/data/users_core.json`
   * Realistic: nationality/language by country, home base (city + Hbf/airport), budgets, interests, diet, comfort, personality, work, travel prefs, and a succinct bio.
2. **Matchmaker** → `Scripts/matchmaker_data_generator.py`

   * **Input:** `users_core.json`
   * **Output:** `MatchMaker/data/matchmaker_profiles.json`
   * Adds match intents, age/gender openness, language policies, availability windows, hard/soft constraints, compatibility weights, visibility & safety controls.
3. **Integrated Groups & Inventories** → `Scripts/Flight_data_generator.py`

   * **Inputs:** `users_core.json`, `matchmaker_profiles.json`
   * **Output:** `Flight/data/travel_groups_integrated_v3.json`
   * Builds **groups** (members by `user_id`), **trip\_context**, **itinerary**, **flight offers** per user, **inter‑city trains**, **hotel inventories**, simple **restaurant/activities inventories** for plan context, and selected cheapest flights.
4. **Restaurants** → `Scripts/restaurant_data_generator.py`

   * **Inputs:** `users_core.json`, `matchmaker_profiles.json` (optional), `travel_groups_integrated_v3.json`
   * **Outputs:**

     * `Restaurants/data/restaurants_catalog.json` (per‑city venues with cuisines, diets, hours, amenities)
     * `Restaurants/data/restaurants_availability.json` (time slots w/ capacity)
     * `Restaurants/data/group_restaurant_reservations.json` (AI-picked group holds by budget/diet/party size)
5. **Activities, Attractions, Events** → `Scripts/activities_events_generator.py`

   * **Inputs:** `users_core.json`, `matchmaker_profiles.json`, `travel_groups_integrated_v3.json`
   * **Outputs:**

     * `Activities/data/activities_catalog.json` (POIs, tours, passes, categories, tags)
     * `Activities/data/events_catalog.json` (seasonal festivals, holidays; API‑friendly fields)
     * `Activities/data/group_activity_reservations.json` (holds aligned to itinerary windows & interests)
6. **Car Rental** → `Scripts/car_rental_data_generator.py`

   * **Inputs:** `users_core.json`, `matchmaker_profiles.json`, `travel_groups_integrated_v3.json`
   * **Outputs:**

     * `Rentals/data/carrental_catalog.json` (per‑city suppliers & fleets)
     * `Rentals/data/carrental_availability.json` (daily stock & price per class)
     * `Rentals/data/group_carrental_reservations.json` (group holds aligned to flight arrivals/returns)

> **Why this order?** Downstream generators rely on `group_id`, itinerary dates/cities, and member lists from the Flight step to customize restaurant/activity/rental choices without duplicating user details.

---

## 4) How each dataset is structured

### 4.1 Users (`users/data/users_core.json`)

Array of user objects:

* `user_id`, `created_at`, `name`, `age`, `gender`
* `contact.rovermitra_handle`, `contact.email`
* `home_base.city`, `home_base.country`, `home_base.nearby_nodes` (e.g., `"Berlin Hbf"`, `"BER"`), `willing_radius_km`
* `languages` (ISO‑like), `interests`, `values`, `personality`, `bio`
* `budget` (amount per day + currency + split rule)
* Diet/health, comfort, social/work prefs, stable travel prefs, privacy flags
* \~20% **lean** profiles (some blocks intentionally missing)

### 4.2 Matchmaker (`MatchMaker/data/matchmaker_profiles.json`)

* `match_profile_id`, `user_id`, `status`, `created_at`/`updated_at`
* `visibility`, `match_intent`, `preferred_companion` (genders, age range, group size, origin distance)
* `communication_preferences` (mode, response SLA)
* `availability_windows` (seasonal)
* `compatibility_weights`, `hard_dealbreakers`, `soft_preferences`
* `language_policy`, `meeting_preference`, compatibility strictness (budget/diet)
* `safety_settings`, `blocklist_user_ids`, and a `match_quality_threshold`

### 4.3 Groups & Transport/Stay (`Flight/data/travel_groups_integrated_v3.json`)

Per group:

* `group_id`, `rovermitra_chat.room_id`
* `members`: array of `{ user_id, rovermitra_handle }`
* `trip_context`: objectives, constraints, luggage, CO₂ preference, output prefs
* `draft_plan`:

  * `meeting_plan` (meet city)
  * `itinerary` (daily `{date, base, plan}`)
  * `intercity_ground_offers` (trains between bases)
  * `flight_offers` per `user_id` + `chosen_flights` (cheapest OB/IB IDs)
  * `hotel_inventory` per city (options with stars, price, distance, amenities)
  * lightweight `restaurant_inventory`, `activities_inventory` for context

### 4.4 Restaurants (`Restaurants/data/...`)

* **Catalog:** per city list of venues with `id`, `name`, `cuisine`, `price_level (€, €€, €€€)`, `dietary_tags`, `rating`, `open_hours`, `amenities`, `seating`, `service_options`, `payment_methods`, `popular_dishes`, `geo`, `reservation_policy`, `currency`, `partner`.
* **Availability:** rows `{ restaurant_id, slot_iso, capacity_left }` for next \~35 days.
* **Group reservations:** rows `{ reservation_id, group_id, city, date, restaurant_id, slot_iso, party_size, status, payment, dietary_considerations, special_requests, matching_rationale }`.

### 4.5 Activities & Events (`Activities/data/...`)

* **Activities catalog:** POIs/tours with categories (museum, hike, rail, market, viewpoint, etc.), tags (family‑friendly, skip‑the‑line, accessible), durations, price, `currency`, `geo` stub, `languages`.
* **Events catalog:** seasonable/events/festivals with `{ id, city, name, start_date, end_date, category, recurrence, venue_hint, ticketing_hint }` and API‑friendly fields; designed to be swapped with real feeds later.
* **Group activity reservations:** `{ reservation_id, group_id, date, city, activity_id | event_id, slot_iso, party_size, status, payment, rationale }`, aligned to itinerary pace/interests and blackout windows.

### 4.6 Car Rentals (`Rentals/data/...`)

* **Catalog:** per city **suppliers** with fleets. Each fleet row includes `vehicle_code` (e.g., `CDMR`), class, seats, large\_bags, transmission, fuel (`ev` included), mileage rules, deposit, included insurance, add‑ons, payment methods, base price, and `pickup_locations` (airport/rail/downtown).
* **Availability:** `{ supplier_id, vehicle_code, date, price_per_day, currency, qty_left }` for \~120 days.
* **Group car reservations:** `{ reservation_id, group_id, city, supplier_id, vehicle_code, pickup.when_iso, dropoff.when_iso, drivers_user_ids, payment, status, matching_rationale }`. Pickup/drop times line up with group flight arrivals/returns when possible.

---

## 5) Running the generators

All scripts live in `/Scripts/` and are **path‑aware** (they compute project root and write to the correct `/data/` folders). Run from repo root:

```bash
python3 Scripts/user_data_generator.py
python3 Scripts/matchmaker_data_generator.py
python3 Scripts/Flight_data_generator.py
python3 Scripts/restaurant_data_generator.py
python3 Scripts/activities_events_generator.py
python3 Scripts/car_rental_data_generator.py
```

Tips:

* Re‑running regenerates fresh data (random but deterministic per script seed). Adjust `random.seed(...)` in each script for variability.
* The Flight step must precede Restaurants/Activities/Rentals so they can read `group_id`, itinerary, and cities.
* Each script prints a small preview; open the generated files to explore full schemas.

---

## 6) Integration principles (what to rely on in apps)

* **Link by IDs** only. Get personal details via a join on `user_id` against `users_core.json` when you render profiles.
* **City currency** flows from country; prices are stored with a `currency` per record.
* **Times are ISO‑8601**; slots and reservations include explicit `...T..:..:..`.
* **Inventories vs. holds**: catalogs/availability are reusable pools; group holds reference those objects by ID and can be promoted to bookings later.
* **CO₂ Preference**: use `trip_context.co2_preference` to prefer trains/E‑cars/near‑center hotels.

---

## 7) Troubleshooting

* **File not found**: Ensure you run scripts from the **repo root** so relative project‑root discovery works.
* **Nothing written to folder**: Check that the parent folder exists; scripts call `mkdir(parents=True, exist_ok=True)` before writing. Inspect console logs for the final **Writing to:** lines.
* **KeyError / schema drift**: If you edited earlier JSONs, re‑run from **Users → Matchmaker → Flight** to re‑establish references.

---

## 8) Extending with live APIs later

The catalogs are purposefully **API‑shaped**:

* Restaurants: add `place_id`/`provider` fields to map to Google/Tripadvisor/Yelp as needed.
* Activities/Events: keep `provider`, `source_url`, `ticketing_provider` for live feeds.
* Car Rentals: add `supplier_code`, `iata` pickup codes when integrating.
* Flights/Hotels: swap synthetic offers with real GDS/OTA content using the existing ID/price/date fields.

---

## 9) Sample joins (pseudocode)

```js
// Show tonight’s dinner place for a group
const group = groups.find(g => g.group_id === GID);
const hold  = restaurantHolds.find(r => r.group_id === GID && r.date === today);
const venue = restaurantCatalog[hold.city].find(v => v.id === hold.restaurant_id);
```

```python
# List drivers with names for a car hold
hold = next(h for h in car_holds if h["group_id"]==gid)
print([ users_by_id[u]["name"] for u in hold["drivers_user_ids"] ])
```

---

## 10) Known simplifications

* Coordinates are **approximate stubs** for clustering/testing only.
* Flight/car availability/pricing are heuristic, not market‑accurate.
* Some countries for events/festivals are illustrative; swap via API later.

---

## 11) Seeds & sizes

* Each script has `random.seed(...)` and knobs like `N_USERS`, `N_GROUPS`, or per‑city catalog ranges.
* Increase counts cautiously — data volume grows fast (availability slots are combinatorial).

---

## 12) License & data handling

* All data are **synthetic** and for development/testing only. Do **not** ship to production or treat as real PII.

Happy shipping! 🚀
