import streamlit as st

def display_itinerary(itinerary):
    """
    Display flights as cards.
    """
    for flight in itinerary:
        with st.container():
            st.markdown(f"""
            **✈️ Flight {flight['flight_number']}**
            - From: {flight['from']} → To: {flight['to']}
            - Departure: {flight['departure']}
            - Arrival: {flight['arrival']}
            - Airline: {flight['airline']} | Class: {flight['class']}
            - **Price: ${flight['price']}**
            """)
            st.markdown("---")
