from flask import Flask, render_template, jsonify, request
import json
import os
from processor import load_popular_routes, get_route_analytics

app = Flask(__name__)

@app.route('/')
def index():
    """Home page showing popular routes"""
    return render_template('index.html')

@app.route('/route/<origin>/<destination>')
def route_details(origin, destination):
    """Route details page"""
    return render_template('route_details.html', origin=origin, destination=destination)

@app.route('/search')
def route_search():
    """Route search page with query parameters"""
    origin = request.args.get('origin', '')
    destination = request.args.get('destination', '')
    start = request.args.get('start', '')
    end = request.args.get('end', '')
    return render_template('route_search.html', origin=origin, destination=destination, start=start, end=end)

@app.route('/api/route/<origin>/<destination>')
def api_route_data(origin, destination):
    """API endpoint to get route data"""
    try:
        # Get optional date range parameters
        start_date = request.args.get('start')
        end_date = request.args.get('end')

        print(f"Fetching route data for {origin} -> {destination}")

        # Use real analytics instead of sample data
        route_data = get_route_analytics(origin.upper(), destination.upper(), start_date, end_date)

        # Check if we got an error response
        if "error" in route_data:
            print(f"Error from analytics: {route_data['error']}")
            # Return a fallback response with basic structure
            return jsonify({
                "origin": origin.upper(),
                "destination": destination.upper(),
                "error": route_data["error"],
                "price_analysis": {
                    "min_price": 0,
                    "max_price": 0,
                    "avg_price": 0,
                    "price_volatility": 0,
                    "direct_avg_price": 0,
                    "indirect_avg_price": 0,
                    "flight_breakdown": {
                        "total_flights": 0,
                        "direct_flights": 0,
                        "connecting_flights": 0,
                        "direct_percentage": 0,
                        "connecting_percentage": 0
                    }
                },
                "price_fluctuation_chart": {
                    "labels": [],
                    "data": [],
                    "trend_data": [],
                    "trend_text": "No data available"
                },
                "timing_analysis": {
                    "best_month": "N/A",
                    "best_month_price": 0,
                    "worst_month": "N/A",
                    "worst_month_price": 0
                },
                "total_flights_analyzed": 0,
                "ai_insights": {
                    "summary": "No flight data available for this route.",
                    "price_prediction": "Unable to generate prediction without data.",
                    "booking_tip": "Try searching for alternative routes or dates.",
                    "airline_performance": "No performance data available.",
                    "travel_recommendations": "Consider checking back later or contacting airlines directly."
                }
            })

        return jsonify(route_data)
    except Exception as e:
        print(f"Exception in api_route_data: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/popular-routes')
def api_popular_routes():
    """API endpoint to get popular routes"""
    try:
        popular_routes_dict = load_popular_routes()
        # Convert to list and sort by descending frequency
        popular_routes = sorted(
            popular_routes_dict.values(),
            key=lambda r: r.get('frequency', 0),
            reverse=True
        )
        return jsonify(popular_routes)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/search-routes')
def api_search_routes():
    """API endpoint to search for routes"""
    query = request.args.get('q', '').lower()
    try:
        # Load airports data for search
        airports_file = os.path.join('data', 'airports.json')
        if os.path.exists(airports_file):
            with open(airports_file, 'r') as f:
                airports = json.load(f)

            # Filter airports based on query
            filtered_airports = []
            for airport in airports:
                if (query in airport.get('name', '').lower() or
                    query in airport.get('code', '').lower() or
                    query in airport.get('city', '').lower()):
                    filtered_airports.append(airport)

            return jsonify(filtered_airports[:10])  # Limit to 10 results
        else:
            return jsonify([])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/monthly/<origin>/<destination>')
def api_monthly_data(origin, destination):
    """API endpoint to get monthly price data for multi-route chart"""
    try:
        # Use real analytics data instead of sample
        route_data = get_route_analytics(origin.upper(), destination.upper())

        if "error" in route_data:
            return jsonify({'data': {}})

        chart = route_data.get('price_fluctuation_chart', {})
        labels = chart.get('labels', [])
        prices = chart.get('data', [])

        # Build mapping of YYYY-MM to price
        monthly = { labels[i]: {'price': prices[i]} for i in range(min(len(labels), len(prices))) }
        return jsonify({'data': monthly})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ensure data directory exists
    if not os.path.exists('data'):
        os.makedirs('data')

    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
