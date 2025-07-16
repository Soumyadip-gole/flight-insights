import requests
import json
import os
import datetime
from dotenv import load_dotenv  # NEW: for loading .env

load_dotenv()  # NEW: Load environment variables from .env
TOKEN = os.getenv("TRAVELPAYOUTS_API_TOKEN")  # NEW: Get token from .env
BASE_URL_V1 = "https://api.travelpayouts.com/v1"
BASE_URL_V2 = "https://api.travelpayouts.com/v2"
HEADERS = {"X-Access-Token": TOKEN, "Accept-Encoding": "gzip"}

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_australian_airports():
    """Fetch Australian airports (cached locally after first run)."""
    cache_file = os.path.join(DATA_DIR, "airports.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                content = f.read().strip()
                if content:
                    return json.loads(content)
        except Exception as e:
            print(f"Error loading cached airports.json: {e}")
            print("Fetching fresh airport data...")

    print("Fetching Australian airports...")
    resp = requests.get("https://api.travelpayouts.com/data/airports.json")
    airports = [a for a in resp.json() if a.get("country_code") == "AU"]
    with open(cache_file, "w") as f:
        json.dump(airports, f, indent=2)
    return airports

def fetch_direct_flights(origin, destination, depart_date):
    """Fetches direct flights for a given month."""
    url = f"{BASE_URL_V1}/prices/direct"
    params = {
        "origin": origin,
        "destination": destination,
        "depart_date": depart_date,
        "currency": "aud",
        "token": TOKEN
    }
    resp = requests.get(url, headers=HEADERS, params=params)
    # The data structure is nested, so we extract it carefully
    return resp.json().get("data", {}).get(destination, {}).get('0', {})

def fetch_monthly_cheapest(origin, destination):
    url = f"{BASE_URL_V2}/prices/month-matrix"
    params = {"currency": "aud", "origin": origin, "destination": destination, "show_to_affiliates": True}
    resp = requests.get(url, headers=HEADERS, params=params)
    return resp.json().get("data", [])

def fetch_nearest_places(origin, destination):
    url = f"{BASE_URL_V2}/prices/nearest-places-matrix"
    params = {"currency": "aud", "origin": origin, "destination": destination, "limit": 5, "show_to_affiliates": True}
    resp = requests.get(url, headers=HEADERS, params=params)
    return resp.json().get("prices", [])

def fetch_latest_cheapest(origin, destination):
    url = f"{BASE_URL_V2}/prices/latest"
    params = {
        "currency": "aud", "origin": origin, "destination": destination,
        "limit": 20, "sorting": "price", "show_to_affiliates": True
    }
    resp = requests.get(url, headers=HEADERS, params=params)
    return resp.json().get("data", [])

def fetch_popular_routes(origin):
    url = f"{BASE_URL_V1}/city-directions"
    params = {"origin": origin, "currency": "aud", "token": TOKEN}
    resp = requests.get(url, headers=HEADERS, params=params)
    return resp.json().get("data", {})

def fetch_airline_directions(airline_code, limit=50):
    """Fetch popular routes for a specific airline."""
    url = f"{BASE_URL_V1}/airline-directions"
    params = {"airline_code": airline_code, "limit": limit, "token": TOKEN}
    resp = requests.get(url, headers=HEADERS, params=params)
    return resp.json().get("data", {})

def fetch_all_popular_routes():
    """Fetch popular routes from major Australian airlines."""
    # Major Australian airlines
    airlines = [
        "QF",  # Qantas
        "VA",  # Virgin Australia
        "JQ",  # Jetstar
        "TT",  # Tiger Airways (now part of Virgin)
        "ZL",  # Regional Express
        "NZ",  # Air New Zealand (operates in Australia)
        "TR"   # Scoot (budget airline serving Australia)
    ]

    all_routes = {}

    for airline in airlines:
        print(f"Fetching popular routes for airline {airline}...")
        try:
            routes_data = fetch_airline_directions(airline)

            for route, frequency in routes_data.items():
                # Parse route (e.g., "SYD-MEL" -> origin: "SYD", destination: "MEL")
                if "-" in route:
                    origin, destination = route.split("-", 1)

                    # Only include routes involving Australian airports
                    australian_airports = {"SYD", "MEL", "BNE", "PER", "ADL", "OOL", "CBR", "CNS", "HBA", "MCY",
                                         "DRW", "TSV", "ROK", "MKY", "ASP", "AVV", "LST", "HTI", "PPP", "BNK"}

                    if origin in australian_airports or destination in australian_airports:
                        route_key = f"{origin}-{destination}"

                        if route_key not in all_routes:
                            all_routes[route_key] = {
                                "origin": origin,
                                "destination": destination,
                                "frequency": 0,
                                "airlines": []
                            }

                        all_routes[route_key]["frequency"] += frequency
                        if airline not in all_routes[route_key]["airlines"]:
                            all_routes[route_key]["airlines"].append(airline)

        except Exception as e:
            print(f"Error fetching data for airline {airline}: {e}")
            continue

    return all_routes

def scrape_all_flights():
    airports = fetch_australian_airports()
    results = {}

    # Use next month for API calls that require a departure date
    next_month = (datetime.date.today().replace(day=1) + datetime.timedelta(days=32)).strftime("%Y-%m")

    for airport in airports:
        # Skip airports without IATA codes or that are not flightable
        if not airport.get("code") or not airport.get("flightable", True):
            continue

        origin = airport["code"]  # Changed from "iata" to "code"
        print(f"Scraping {origin} - {airport.get('name', 'Unknown')}...")

        # For demo, limit to 5 most popular destinations per airport (to save API calls)
        popular_routes = fetch_popular_routes(origin)
        destinations = list(popular_routes.keys())[:5]

        results[origin] = {
            "popular_routes": popular_routes,
            "flights": {}
        }

        for dest in destinations:
            cheapest = fetch_monthly_cheapest(origin, dest)
            if not cheapest:
                cheapest = fetch_latest_cheapest(origin, dest)

            direct = fetch_direct_flights(origin, dest, next_month)
            nearest = fetch_nearest_places(origin, dest)

            results[origin]["flights"][dest] = {
                "cheapest": cheapest,
                "direct": direct,
                "nearest": nearest
            }

    # Save to JSON
    with open(os.path.join(DATA_DIR, "flights.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("✅ Flight data saved to data/flights.json")

def scrape_popular_routes():
    """Scrape and save popular routes data."""
    print("Fetching popular routes from major airlines...")
    popular_routes = fetch_all_popular_routes()

    # Sort by frequency (most popular first)
    sorted_routes = dict(sorted(popular_routes.items(),
                               key=lambda x: x[1]["frequency"],
                               reverse=True))

    # Save to JSON file
    popular_routes_file = os.path.join(DATA_DIR, "popular_routes.json")
    with open(popular_routes_file, "w") as f:
        json.dump(sorted_routes, f, indent=2)

    print(f"✅ Popular routes data saved to {popular_routes_file}")
    print(f"Found {len(sorted_routes)} popular routes")

    return sorted_routes

if __name__ == "__main__":
    # Add option to scrape just popular routes
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "popular":
        scrape_popular_routes()
    else:
        scrape_all_flights()
