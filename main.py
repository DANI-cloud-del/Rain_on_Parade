from flask import Flask, render_template, jsonify, request, redirect, url_for
import os
from dotenv import load_dotenv
from groq import Groq
import json
from datetime import datetime
from rainfall_predictor import RainfallPredictor

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Load weather prediction model
weather_predictor = RainfallPredictor()
if os.path.exists('weather_model.pkl'):
    weather_predictor.load_model()
    print("✅ Weather model loaded")

# Weather API Routes
@app.route('/api/weather/predict', methods=['POST'])
def predict_weather():
    """API endpoint for weather prediction"""
    try:
        data = request.json
        city = data.get('city', 'Mumbai')
        date = data.get('date')
        
        if not date:
            return jsonify({'error': 'Date is required'}), 400
        
        result = weather_predictor.predict(city=city, date_str=date)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/event/check-weather', methods=['POST'])
def check_event_weather():
    """Check weather for a specific event"""
    try:
        data = request.json
        event_date = data.get('date')
        event_name = data.get('name', 'Event')
        city = data.get('city', 'Mumbai')
        
        if not event_date:
            return jsonify({'error': 'Date is required'}), 400
        
        result = weather_predictor.predict(city=city, date_str=event_date)
        
        # Add event-specific info
        result['event_name'] = event_name
        result['is_suitable'] = result['rain_probability'] < 40
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Create data directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

EVENTS_FILE = 'data/events.json'

# Initialize events file if it doesn't exist
if not os.path.exists(EVENTS_FILE):
    with open(EVENTS_FILE, 'w') as f:
        json.dump({}, f)


# Event Management Routes
@app.route('/api/events', methods=['GET'])
def get_events():
    try:
        with open(EVENTS_FILE, 'r') as f:
            events = json.load(f)
        return jsonify({'success': True, 'events': events})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/events/<date>', methods=['GET'])
def get_events_by_date(date):
    try:
        with open(EVENTS_FILE, 'r') as f:
            events = json.load(f)
        date_events = events.get(date, [])
        return jsonify({'success': True, 'events': date_events})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/events', methods=['POST'])
def save_event():
    try:
        data = request.json
        date = data.get('date')
        event_text = data.get('text')
        
        if not date or not event_text:
            return jsonify({'success': False, 'error': 'Date and text required'}), 400
        
        # Load existing events
        with open(EVENTS_FILE, 'r') as f:
            events = json.load(f)
        
        # Initialize date if not exists
        if date not in events:
            events[date] = []
        
        # Add new event
        event_id = len(events[date])
        new_event = {
            'id': event_id,
            'text': event_text,
            'created_at': datetime.now().isoformat()
        }
        
        events[date].append(new_event)
        
        # Save back to file
        with open(EVENTS_FILE, 'w') as f:
            json.dump(events, f, indent=2)
        
        return jsonify({'success': True, 'event': new_event})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/events/<date>/<int:event_id>', methods=['DELETE'])
def delete_event(date, event_id):
    try:
        with open(EVENTS_FILE, 'r') as f:
            events = json.load(f)
        
        if date in events and event_id < len(events[date]):
            events[date].pop(event_id)
            
            # Re-index remaining events
            for i, event in enumerate(events[date]):
                event['id'] = i
            
            # Remove date key if no events left
            if not events[date]:
                del events[date]
            
            with open(EVENTS_FILE, 'w') as f:
                json.dump(events, f, indent=2)
            
            return jsonify({'success': True})
        
        return jsonify({'success': False, 'error': 'Event not found'}), 404
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# Main route for home page
@app.route("/")
def home():
    return render_template("HomePage.html")


# Scheduler page route
@app.route("/scheduler")
def scheduler():
    return render_template("scheduler.html")


# Forecaster page route
@app.route("/forecaster")
def forecaster():
    return render_template("forecaster.html")


# Import and register chatbot routes AFTER app is created
from chatbot_api import register_chatbot_routes
register_chatbot_routes(app)

if __name__ == "__main__":
    # For Render deployment - bind to 0.0.0.0
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
