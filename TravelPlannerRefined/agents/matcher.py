# agents/matcher.py

def match_users(user, all_users):
    """
    Simple matching algorithm: find users with similar destinations, dates, and preferences.
    Returns a list of matched users.
    """
    matches = []
    for other in all_users:
        if other["id"] == user["id"]:
            continue

        # Score matches by overlapping destinations
        common_destinations = set(user["destinations"]) & set(other["destinations"])
        common_activities = set(user["preferences"].get("activities", [])) & set(other["preferences"].get("activities", []))
        if common_destinations and common_activities:
            matches.append({
                "user": other,
                "score": len(common_destinations) + len(common_activities)
            })

    # Sort matches by score descending
    matches.sort(key=lambda x: x["score"], reverse=True)
    return [m["user"] for m in matches]
