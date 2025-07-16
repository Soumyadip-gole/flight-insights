import json
import os
import requests
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
TOKEN = os.getenv("TRAVELPAYOUTS_API_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini AI
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
else:
    model = None

BASE_URL_V1 = "https://api.travelpayouts.com/v1"
BASE_URL_V2 = "https://api.travelpayouts.com/v2"
HEADERS = {"X-Access-Token": TOKEN, "Accept-Encoding": "gzip"}

DATA_DIR = "data"
FLIGHTS_FILE = os.path.join(DATA_DIR, "flights.json")
POPULAR_ROUTES_FILE = os.path.join(DATA_DIR, "popular_routes.json")

def load_flights():
    if not os.path.exists(FLIGHTS_FILE):
        return {}
    try:
        with open(FLIGHTS_FILE, "r") as f:
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
    except Exception as e:
        print(f"Error loading flights.json: {e}")
        return {}

def load_popular_routes():
    """Load popular routes from the new JSON file."""
    if not os.path.exists(POPULAR_ROUTES_FILE):
        return {}
    try:
        with open(POPULAR_ROUTES_FILE, "r") as f:
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
    except Exception as e:
        print(f"Error loading popular_routes.json: {e}")
        return {}

def get_cheapest_summary():
    flights = load_flights()
    summary = []

    for origin, info in flights.items():
        for dest, flight_data in info["flights"].items():
            cheapest = flight_data["cheapest"]
            if cheapest:
                # Pick the lowest price
                price = min([f["value"] for f in cheapest if "value" in f], default=None)
                if price:
                    summary.append({"origin": origin, "destination": dest, "price": price})

    summary = sorted(summary, key=lambda x: x["price"])
    return summary

def get_cheapest_direct_summary():
    """Extracts the cheapest direct flights from the cached data."""
    flights = load_flights()
    summary = []

    for origin, info in flights.items():
        for dest, flight_data in info["flights"].items():
            direct_flight = flight_data.get("direct")
            if direct_flight and "price" in direct_flight:
                summary.append({
                    "origin": origin,
                    "destination": dest,
                    "price": direct_flight["price"]
                })

    summary = sorted(summary, key=lambda x: x["price"])
    return summary

def get_popular_routes_summary():
    """Get popular routes with comprehensive airline data from flights.json."""
    flights = load_flights()
    popular_routes_data = load_popular_routes()
    popular_routes = []

    # Extract airlines from flights.json for each route
    for route_key, route_info in popular_routes_data.items():
        origin = route_info["origin"]
        destination = route_info["destination"]

        # Get airlines from flights.json popular_routes section
        airlines_from_flights = set()
        origin_data = flights.get(origin, {})
        popular_routes_section = origin_data.get("popular_routes", {})

        # Collect all airlines for this destination from the origin
        for dest, route_detail in popular_routes_section.items():
            if dest == destination and isinstance(route_detail, dict):
                airline = route_detail.get("airline")
                if airline:
                    airlines_from_flights.add(airline)

        # If no airlines found in flights.json, use the ones from popular_routes.json
        if not airlines_from_flights:
            airlines_from_flights = set(route_info["airlines"])

        popular_routes.append({
            "origin": origin,
            "destination": destination,
            "frequency": route_info["frequency"],
            "airline_count": len(airlines_from_flights),
            "airlines": ", ".join(sorted(airlines_from_flights))
        })

    return popular_routes[:20]  # Return top 20 most popular routes

def get_route_flights(origin, destination):
    """Get all flight data for a specific route."""
    flights = load_flights()
    route_data = {
        "origin": origin,
        "destination": destination,
        "all_flights": [],
        "direct_flights": []
    }

    # Get data from origin airport
    origin_data = flights.get(origin, {})
    flight_data = origin_data.get("flights", {}).get(destination, {})

    # Process cheapest flights (with connections)
    cheapest = flight_data.get("cheapest", [])
    for flight in cheapest:
        if "value" in flight:
            route_data["all_flights"].append({
                "price": flight["value"],
                "airline": flight.get("airline", "N/A"),
                "flight_number": flight.get("flight_number", "N/A"),
                "departure_at": flight.get("departure_at", "N/A"),
                "return_at": flight.get("return_at", "N/A"),
                "transfers": flight.get("transfers", 0),
                "type": "cheapest"
            })

    # Process direct flights
    direct = flight_data.get("direct", {})
    if direct and "price" in direct:
        route_data["direct_flights"].append({
            "price": direct["price"],
            "airline": direct.get("airline", "N/A"),
            "flight_number": direct.get("flight_number", "N/A"),
            "departure_at": direct.get("departure_at", "N/A"),
            "return_at": direct.get("return_at", "N/A"),
            "transfers": 0,
            "type": "direct"
        })

    # Process nearest places (alternative routes)
    nearest = flight_data.get("nearest", [])
    for flight in nearest:
        if "value" in flight:
            route_data["all_flights"].append({
                "price": flight["value"],
                "airline": flight.get("airline", "N/A"),
                "flight_number": flight.get("flight_number", "N/A"),
                "departure_at": flight.get("departure_at", "N/A"),
                "return_at": flight.get("return_at", "N/A"),
                "transfers": flight.get("transfers", 1),
                "type": "nearest"
            })

    # Sort by price
    route_data["all_flights"] = sorted(route_data["all_flights"], key=lambda x: x["price"])
    route_data["direct_flights"] = sorted(route_data["direct_flights"], key=lambda x: x["price"])

    return route_data

def fetch_live_route_flights(origin, destination):
    """Fetch and combine live flight data for a specific route from multiple API endpoints."""
    route_data = {
        "origin": origin,
        "destination": destination,
        "all_flights": [],
        "direct_flights": []
    }

    try:
        print(f"Fetching live data for {origin} to {destination}...")
        import datetime
        depart_month = datetime.date.today().strftime("%Y-%m")
        return_month = (datetime.date.today() + datetime.timedelta(days=30)).strftime("%Y-%m")

        # Debug: Test a simple API call first
        print(f"Testing API with token: {TOKEN[:10]}...")

        # 1. Try cheapest tickets endpoint
        url_cheap = f"{BASE_URL_V1}/prices/cheap"
        params_cheap = {
            "origin": origin,
            "destination": destination,
            "depart_date": depart_month,
            "return_date": return_month,
            "currency": "aud",
            "token": TOKEN
        }
        print(f"Calling: {url_cheap} with params: {params_cheap}")
        resp_cheap = requests.get(url_cheap, params=params_cheap)
        print(f"Cheap flights response status: {resp_cheap.status_code}")
        print(f"Cheap flights response: {resp_cheap.text[:500]}")

        if resp_cheap.status_code == 200:
            cheap_data = resp_cheap.json().get("data", {})
            if destination in cheap_data:
                for key, flight in cheap_data[destination].items():
                    if isinstance(flight, dict) and "price" in flight:
                        route_data["all_flights"].append({
                            "price": flight["price"],
                            "airline": flight.get("airline", "Unknown"),
                            "flight_number": str(flight.get("flight_number", "")),
                            "departure_at": flight.get("departure_at", "N/A"),
                            "return_at": flight.get("return_at", "N/A"),
                            "transfers": flight.get("transfers", 0),
                            "type": "cheap"
                        })

        # 2. Try latest prices endpoint
        url_latest = f"{BASE_URL_V2}/prices/latest"
        params_latest = {
            "currency": "aud",
            "origin": origin,
            "destination": destination,
            "limit": 30,
            "sorting": "price",
            "token": TOKEN
        }
        print(f"Calling: {url_latest} with params: {params_latest}")
        resp_latest = requests.get(url_latest, params=params_latest)
        print(f"Latest prices response status: {resp_latest.status_code}")
        print(f"Latest prices response: {resp_latest.text[:500]}")

        if resp_latest.status_code == 200:
            latest_data = resp_latest.json().get("data", [])
            for flight in latest_data:
                if "value" in flight:
                    # Use number_of_changes if available, otherwise default to 1
                    transfers = flight.get("number_of_changes", flight.get("transfers", 1))
                    route_data["all_flights"].append({
                        "price": flight["value"],
                        "airline": flight.get("airline", flight.get("gate", "Various")),
                        "flight_number": str(flight.get("flight_number", f"Flight {flight['value']}")),
                        "departure_at": flight.get("depart_date", "N/A"),
                        "return_at": flight.get("return_date", "N/A"),
                        "transfers": transfers,
                        "type": "latest"
                    })

        # 2. Try direct flights endpoint (v1)
        url_direct_v1 = f"{BASE_URL_V1}/prices/direct"
        params_direct = params_cheap.copy()
        print(f"Calling direct v1: {url_direct_v1} with params: {params_direct}")
        resp_direct = requests.get(url_direct_v1, params=params_direct, headers=HEADERS)
        print(f"Direct v1 response status: {resp_direct.status_code}")
        if resp_direct.status_code == 200:
            direct_data_v1 = resp_direct.json().get("data", {})
            if destination in direct_data_v1:
                for key, flight in direct_data_v1[destination].items():
                    if isinstance(flight, dict) and "price" in flight:
                        route_data["direct_flights"].append({
                            "price": flight["price"],
                            "airline": flight.get("airline", "N/A"),
                            "flight_number": str(flight.get("flight_number", "")),
                            "departure_at": flight.get("departure_at", "N/A"),
                            "return_at": flight.get("return_at", "N/A"),
                            "transfers": 0,
                            "type": "direct_v1"
                        })

        # If we still have no data, create some sample data from popular routes
        if not route_data["all_flights"]:
            print("No API data found, using popular routes as fallback...")
            popular_routes_data = load_popular_routes()
            route_key = f"{origin}-{destination}"
            if route_key in popular_routes_data:
                airlines = popular_routes_data[route_key]["airlines"]
                sample_prices = [89, 129, 159, 189, 219, 249]

                for i, airline in enumerate(airlines[:3]):
                    for j, base_price in enumerate(sample_prices[:2]):
                        price = base_price + (i * 15)
                        route_data["all_flights"].append({
                            "price": price,
                            "airline": airline,
                            "flight_number": f"{airline}{300 + j}",
                            "departure_at": "Multiple times daily",
                            "return_at": "Multiple times daily",
                            "transfers": j,  # First is direct, second has transfers
                            "type": "sample"
                        })

        # Separate direct flights
        route_data["direct_flights"] = [f for f in route_data["all_flights"] if f["transfers"] == 0]

        # Sort by price
        route_data["all_flights"] = sorted(route_data["all_flights"], key=lambda x: x["price"])
        route_data["direct_flights"] = sorted(route_data["direct_flights"], key=lambda x: x["price"])

        print(f"Final result for {origin}-{destination}: {len(route_data['all_flights'])} total flights, {len(route_data['direct_flights'])} direct flights")

    except Exception as e:
        print(f"Error in fetch_live_route_flights for {origin}-{destination}: {e}")
        import traceback
        traceback.print_exc()

    return route_data

def get_route_insights(origin, destination):
    """Get comprehensive insights for a specific route from cached data."""
    flights_data = load_flights()
    popular_routes_data = load_popular_routes()

    route_key = f"{origin}-{destination}"
    insights = {
        "origin": origin,
        "destination": destination,
        "route_analysis": {},
        "price_analysis": {},
        "airline_analysis": {},
        "route_popularity": {}
    }

    # --- 1. Airline Analysis ---
    # Get all airlines flying out of the origin airport
    origin_airlines = set()
    if origin in flights_data and "popular_routes" in flights_data[origin]:
        for route in flights_data[origin]["popular_routes"].values():
            if route.get("airline"):
                origin_airlines.add(route["airline"])

    # Get all airlines flying out of the destination airport
    destination_airlines = set()
    if destination in flights_data and "popular_routes" in flights_data[destination]:
        for route in flights_data[destination]["popular_routes"].values():
            if route.get("airline"):
                destination_airlines.add(route["airline"])

    # The airlines serving the route are likely the intersection of both sets
    common_airlines = origin_airlines.intersection(destination_airlines)

    # Add airlines from popular_routes.json as a fallback
    if not common_airlines and route_key in popular_routes_data:
        common_airlines.update(popular_routes_data[route_key].get("airlines", []))

    insights["airline_analysis"] = {
        "total_airlines": len(common_airlines),
        "airlines_list": sorted(list(common_airlines)),
        "market_competition": "High" if len(common_airlines) >= 5 else "Medium" if len(common_airlines) >= 3 else "Low"
    }

    # --- 2. Price Analysis ---
    all_prices = []
    airline_prices = {}
    if origin in flights_data and "popular_routes" in flights_data[origin]:
        route_info = flights_data[origin]["popular_routes"].get(destination)
        if route_info and route_info.get("price"):
            price = route_info["price"]
            all_prices.append(price)
            if route_info.get("airline"):
                airline_prices[route_info["airline"]] = price

    insights["airline_analysis"]["airline_prices"] = airline_prices

    if all_prices:
        insights["price_analysis"] = {
            "min_price": min(all_prices),
            "max_price": max(all_prices),
            "avg_price": round(sum(all_prices) / len(all_prices), 2),
            "price_range": max(all_prices) - min(all_prices)
        }
    else:
        insights["price_analysis"] = {"min_price": 0, "max_price": 0, "avg_price": 0, "price_range": 0}

    # --- 3. Route Popularity ---
    if route_key in popular_routes_data:
        pop_info = popular_routes_data[route_key]
        insights["route_popularity"] = {
            "frequency_score": pop_info.get("frequency", 0),
            "popularity_rank": "High" if pop_info.get("frequency", 0) > 10000000 else "Medium"
        }
    else:
        insights["route_popularity"] = {"frequency_score": 0, "popularity_rank": "Low"}

    # --- 4. Route Analysis ---
    flights_section = flights_data.get(origin, {}).get("flights", {}).get(destination, {})
    insights["route_analysis"] = {
        "has_direct_flights": bool(flights_section.get("direct")),
        "alternative_routes": len(flights_section.get("nearest", [])),
        "route_type": "Domestic" if origin in ["SYD", "MEL", "BNE", "PER", "ADL"] and destination in ["SYD", "MEL", "BNE", "PER", "ADL"] else "International"
    }

    return insights

def get_route_analytics(origin, destination, start_date=None, end_date=None):
    """
    Fetches live data from the /v2/prices/latest endpoint and generates
    analytics and insights for the given route, optionally filtered by a date range.
    """
    print(f"Fetching analytics for {origin} -> {destination} (Range: {start_date} to {end_date})")

    # --- 1. Fetch Live Data ---
    url = "https://api.travelpayouts.com/v2/prices/latest"
    params = {
        "currency": "aud",
        "origin": origin,
        "destination": destination,
        "period_type": "year",
        "one_way": "true",
        "limit": 1000,
        "token": TOKEN
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json().get("data", [])
        print(f"Successfully fetched {len(data)} records.")
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return {"error": "Failed to fetch flight data from API."}

    if not data:
        return {"error": "No flight data available for this route."}

    # --- 2. Process and Clean Data ---
    import pandas as pd
    import numpy as np
    from collections import Counter

    df = pd.DataFrame(data)
    df = df[df['value'] > 0].copy()
    df['depart_date'] = pd.to_datetime(df['depart_date'])

    # Filter by date range if provided
    if start_date and end_date:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        df = df[(df['depart_date'] >= start_date) & (df['depart_date'] <= end_date)]
        print(f"Filtered data to {len(df)} records between {start_date.date()} and {end_date.date()}.")

    if df.empty:
        return {"error": "No flight data available for the selected date range."}

    df['month'] = df['depart_date'].dt.to_period('M')

    # --- 3. Generate Advanced Analytics ---

    # Price Analysis
    price_analysis = {
        "min_price": int(df['value'].min()),
        "max_price": int(df['value'].max()),
        "avg_price": round(df['value'].mean(), 2),
        "price_volatility": round(df['value'].std(), 2) if len(df['value']) > 1 else 0
    }

    # Monthly Price Fluctuation & Trend Line
    monthly_avg = df.groupby('month')['value'].mean().round(2).sort_index()
    trend_line_data = []
    trend_text = "Not enough data for trend"

    # Add some debugging to understand the data
    print(f"Monthly averages: {monthly_avg}")
    print(f"Number of months: {len(monthly_avg)}")

    if len(monthly_avg) > 1:
        x = np.arange(len(monthly_avg))
        y = monthly_avg.values
        # Safely compute trend line only when multiple data points are available
        try:
            coeffs = np.polyfit(x, y, 1)
            slope = coeffs[0]
            trend_line_data = list(np.poly1d(coeffs)(x))
            if slope > 5:
                trend_text = "Prices Trending Up"
            elif slope < -5:
                trend_text = "Prices Trending Down"
            else:
                trend_text = "Prices Stable"
        except Exception:
            trend_text = "Not enough data for trend"
            trend_line_data = []
    elif len(monthly_avg) == 1:
        # For single month data, create a flat line to show the price
        trend_line_data = [monthly_avg.iloc[0], monthly_avg.iloc[0]]
        trend_text = "Limited data - Single month average"
    else:
        # If we have daily data but only one month, try grouping by week instead
        weekly_avg = df.groupby(df['depart_date'].dt.to_period('W'))['value'].mean().round(2).sort_index()
        print(f"Weekly averages: {weekly_avg}")

        if len(weekly_avg) > 1:
            monthly_avg = weekly_avg  # Use weekly data for the chart
            x = np.arange(len(weekly_avg))
            y = weekly_avg.values
            try:
                coeffs = np.polyfit(x, y, 1)
                slope = coeffs[0]
                trend_line_data = list(np.poly1d(coeffs)(x))
                if slope > 2:
                    trend_text = "Prices Trending Up (Weekly)"
                elif slope < -2:
                    trend_text = "Prices Trending Down (Weekly)"
                else:
                    trend_text = "Prices Stable (Weekly)"
            except Exception:
                trend_line_data = [weekly_avg.iloc[0], weekly_avg.iloc[-1]]
                trend_text = "Weekly price variation"

    price_fluctuation = {
        "labels": [str(p) for p in monthly_avg.index],
        "data": list(monthly_avg.values),
        "trend_data": trend_line_data,
        "trend_text": trend_text
    }

    # Best & Worst Time to Fly
    timing_analysis = {}
    if not monthly_avg.empty:
        best_month = monthly_avg.idxmin()
        worst_month = monthly_avg.idxmax()
        timing_analysis = {
            "best_month": str(best_month),
            "best_month_price": monthly_avg.min(),
            "worst_month": str(worst_month),
            "worst_month_price": monthly_avg.max()
        }

    # Compute direct and non-direct average prices
    # Check for different possible column names for transfers/connections
    transfers_col = None
    if 'number_of_changes' in df.columns:
        transfers_col = 'number_of_changes'
    elif 'transfers' in df.columns:
        transfers_col = 'transfers'
    elif 'stops' in df.columns:
        transfers_col = 'stops'

    # Categorize flights properly based on number of changes
    if transfers_col:
        direct_flights = df[df[transfers_col] == 0]
        connecting_flights = df[df[transfers_col] > 0]

        direct_prices = direct_flights['value']
        indirect_prices = connecting_flights['value']

        # Add detailed flight categorization to analytics
        flight_categorization = {
            "total_flights": len(df),
            "direct_flights": len(direct_flights),
            "connecting_flights": len(connecting_flights),
            "direct_percentage": round((len(direct_flights) / len(df)) * 100, 1) if len(df) > 0 else 0,
            "connecting_percentage": round((len(connecting_flights) / len(df)) * 100, 1) if len(df) > 0 else 0
        }

        print(f"Flight breakdown: {len(direct_flights)} direct, {len(connecting_flights)} connecting flights")
    else:
        # If no transfers column exists, assume all flights are mixed
        direct_prices = df['value']  # Use all prices as fallback
        indirect_prices = pd.Series(dtype=float)  # Empty series
        flight_categorization = {
            "total_flights": len(df),
            "direct_flights": len(df),
            "connecting_flights": 0,
            "direct_percentage": 100.0,
            "connecting_percentage": 0.0
        }

    price_analysis['direct_avg_price'] = round(direct_prices.mean(), 2) if not direct_prices.empty else 0
    price_analysis['indirect_avg_price'] = round(indirect_prices.mean(), 2) if not indirect_prices.empty else 0
    price_analysis['flight_breakdown'] = flight_categorization

    # --- 4. Generate AI Insights ---
    print("🤖 Generating AI-powered insights...")

    # Create a temporary insights object for AI analysis
    temp_insights = {
        "origin": origin,
        "destination": destination,
        "price_analysis": price_analysis,
        "price_fluctuation_chart": price_fluctuation,
        "timing_analysis": timing_analysis,
        "total_flights_analyzed": len(df)
    }

    ai_insights = generate_ai_insights(temp_insights)

    # --- 5. Compile Final Insights Object with AI ---
    insights = {
        "origin": origin,
        "destination": destination,
        "price_analysis": price_analysis,
        "price_fluctuation_chart": price_fluctuation,
        "timing_analysis": timing_analysis,
        "total_flights_analyzed": len(df),
        "ai_insights": ai_insights  # Add AI insights to the response
    }

    return insights

def generate_ai_insights(route_data: dict) -> dict:
    """
    Generate AI-powered insights using Gemini for flight route data.
    """
    if not model:
        print("❌ Gemini API not configured")
        return {
            "summary": "AI insights unavailable - Gemini API not configured",
            "price_prediction": "N/A",
            "booking_tip": "N/A",
            "airline_performance": "N/A",
            "travel_recommendations": "N/A"
        }

    # Create a comprehensive prompt for flight insights
    prompt = f"""
You are an expert flight analyst. Analyze this flight route data and provide insights.

Flight Route Data:
- Origin: {route_data.get('origin', 'N/A')}
- Destination: {route_data.get('destination', 'N/A')}
- Average Price: ${route_data.get('price_analysis', {}).get('avg_price', 0)}
- Price Range: ${route_data.get('price_analysis', {}).get('min_price', 0)} - ${route_data.get('price_analysis', {}).get('max_price', 0)}
- Price Volatility: ${route_data.get('price_analysis', {}).get('price_volatility', 0)}
- Direct Flights: {route_data.get('price_analysis', {}).get('flight_breakdown', {}).get('direct_flights', 0)} ({route_data.get('price_analysis', {}).get('flight_breakdown', {}).get('direct_percentage', 0)}%)
- Connecting Flights: {route_data.get('price_analysis', {}).get('flight_breakdown', {}).get('connecting_flights', 0)} ({route_data.get('price_analysis', {}).get('flight_breakdown', {}).get('connecting_percentage', 0)}%)
- Price Trend: {route_data.get('price_fluctuation_chart', {}).get('trend_text', 'N/A')}
- Best Month: {route_data.get('timing_analysis', {}).get('best_month', 'N/A')} (${route_data.get('timing_analysis', {}).get('best_month_price', 0)})
- Worst Month: {route_data.get('timing_analysis', {}).get('worst_month', 'N/A')} (${route_data.get('timing_analysis', {}).get('worst_month_price', 0)})
- Total Flights Analyzed: {route_data.get('total_flights_analyzed', 0)}

Please provide exactly 5 insights in this format:
SUMMARY: [Brief route overview and key findings]
PREDICTION: [Fare trend prediction for next 3 months]
BOOKING: [Best time or strategy to book this route]
PERFORMANCE: [Analysis of airline reliability and performance]
RECOMMENDATIONS: [Additional travel tips and recommendations]
"""

    try:
        print("🤖 Generating AI insights with Gemini...")
        response = model.generate_content(prompt)
        response_text = response.text.strip()

        print(f"Raw AI response: {response_text[:200]}...")

        # Parse the structured response
        insights = {
            "summary": "Route analysis completed",
            "price_prediction": "Price trends analyzed",
            "booking_tip": "Compare prices across platforms",
            "airline_performance": "Service quality varies",
            "travel_recommendations": "Consider flexible dates"
        }

        # Extract insights from the response
        lines = response_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('SUMMARY:'):
                insights["summary"] = line.replace('SUMMARY:', '').strip()
            elif line.startswith('PREDICTION:'):
                insights["price_prediction"] = line.replace('PREDICTION:', '').strip()
            elif line.startswith('BOOKING:'):
                insights["booking_tip"] = line.replace('BOOKING:', '').strip()
            elif line.startswith('PERFORMANCE:'):
                insights["airline_performance"] = line.replace('PERFORMANCE:', '').strip()
            elif line.startswith('RECOMMENDATIONS:'):
                insights["travel_recommendations"] = line.replace('RECOMMENDATIONS:', '').strip()

        print("✅ AI insights generated successfully")
        print(f"Summary: {insights['summary'][:100]}...")
        return insights

    except Exception as e:
        print(f"❌ AI insight generation failed: {e}")
        import traceback
        traceback.print_exc()

        # Return meaningful fallback insights based on the actual data
        price_analysis = route_data.get('price_analysis', {})
        timing_analysis = route_data.get('timing_analysis', {})
        flight_breakdown = price_analysis.get('flight_breakdown', {})

        return {
            "summary": f"Route from {route_data.get('origin', 'N/A')} to {route_data.get('destination', 'N/A')} analyzed with {route_data.get('total_flights_analyzed', 0)} flights. Average price: ${price_analysis.get('avg_price', 0)}, with {flight_breakdown.get('direct_percentage', 0)}% direct flights.",
            "price_prediction": f"Based on the {route_data.get('price_fluctuation_chart', {}).get('trend_text', 'stable')} trend, expect prices to remain within the ${price_analysis.get('min_price', 0)}-${price_analysis.get('max_price', 0)} range.",
            "booking_tip": f"Best time to book: {timing_analysis.get('best_month', 'N/A')} (${timing_analysis.get('best_month_price', 0)}). Avoid {timing_analysis.get('worst_month', 'N/A')} (${timing_analysis.get('worst_month_price', 0)}).",
            "airline_performance": f"With {flight_breakdown.get('direct_percentage', 0)}% direct flights available, this route offers good convenience. Compare airlines for best service.",
            "travel_recommendations": f"Price volatility is ${price_analysis.get('price_volatility', 0)}, so monitor prices closely. Direct flights average ${price_analysis.get('direct_avg_price', 0)}."
        }

