# agents/itinerary_planner.py

def find_joint_itinerary(main_user, buddy_user, shared_destination, flight_data):
    """
    Finds the best flight for two users to a shared destination.
    """
    itinerary = {"main_user_flight": None, "buddy_user_flight": None, "total_price": 0}

    # Find best flight for the main user
    main_user_flights = [f for f in flight_data if f["from"] == main_user["origin"] and f["to"] == shared_destination]
    if main_user_flights:
        # Prioritize preferred airline, then price
        main_user_flights.sort(key=lambda x: (x["airline"] != main_user["preferences"]["airline"], x["price"]))
        itinerary["main_user_flight"] = main_user_flights[0]

    # Find best flight for the buddy
    buddy_user_flights = [f for f in flight_data if f["from"] == buddy_user["origin"] and f["to"] == shared_destination]
    if buddy_user_flights:
        buddy_user_flights.sort(key=lambda x: (x["airline"] != buddy_user["preferences"]["airline"], x["price"]))
        itinerary["buddy_user_flight"] = buddy_user_flights[0]

    # Calculate total price if both flights are found
    if itinerary["main_user_flight"] and itinerary["buddy_user_flight"]:
        itinerary["total_price"] = itinerary["main_user_flight"]["price"] + itinerary["buddy_user_flight"]["price"]
        return itinerary
        
    return None # Return None if a full itinerary can't be made