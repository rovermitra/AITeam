def find_best_flights(user, flight_data):
    itinerary = []
    origin = user["origin"]

    for dest in user.get("destinations", []):
        possible_flights = [f for f in flight_data if f["from"] == origin and f["to"] == dest]

        if not possible_flights:
            possible_flights = [f for f in flight_data if f["to"] == dest]

        if possible_flights:
            selected = min(possible_flights, key=lambda x: x["price"])
            itinerary.append(selected)
            origin = selected["to"]

    return itinerary
