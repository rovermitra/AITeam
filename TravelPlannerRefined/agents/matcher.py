# agents/matcher.py

def match_users(user, all_users):
    matches = []
    for other in all_users:
        if other["id"] == user["id"]:
            continue

        # Destination and activity score (as before)
        common_destinations = set(user["destinations"]) & set(other["destinations"])
        common_activities = set(user["preferences"].get("activities", [])) & set(other["preferences"].get("activities", []))
        
        # **NEW: Budget Similarity Score**
        # Score of 1 if budgets are within 20% of each other, 0 otherwise.
        budget_score = 0
        if abs(user["budget"] - other["budget"]) <= (user["budget"] * 0.20):
            budget_score = 1

        # Calculate total score
        total_score = len(common_destinations) + len(common_activities) + (budget_score * 2) # Give budget matching more weight

        if total_score > 1: # Only consider matches with some overlap
            matches.append({
                "user": other,
                "score": total_score
            })

    matches.sort(key=lambda x: x["score"], reverse=True)
    return [m["user"] for m in matches]