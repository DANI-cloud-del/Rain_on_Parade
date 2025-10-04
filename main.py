from flask import Flask, render_template, jsonify, request, redirect, url_for
import os
from dotenv import load_dotenv
from groq import Groq
import json
from datetime import datetime


# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Create data directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

EVENTS_FILE = 'data/events.json'

# Initialize events file if it doesn't exist
if not os.path.exists(EVENTS_FILE):
    with open(EVENTS_FILE, 'w') as f:
        json.dump({}, f)

# Get all events
@app.route('/api/events', methods=['GET'])
def get_events():
    try:
        with open(EVENTS_FILE, 'r') as f:
            events = json.load(f)
        return jsonify({'success': True, 'events': events})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Get events for a specific date
@app.route('/api/events/<date>', methods=['GET'])
def get_events_by_date(date):
    try:
        with open(EVENTS_FILE, 'r') as f:
            events = json.load(f)
        date_events = events.get(date, [])
        return jsonify({'success': True, 'events': date_events})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Add or update event
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

# Delete event
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

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Store conversation history (in production, use a database)
conversation_history = {}

# Page content descriptions for context-aware responses
PAGE_CONTEXTS = {
    'home': {
        'name': 'Home Page',
        'description': 'This is the main homepage with the Rain Assistant chatbot where users can interact and get help with planning events.',
        'content': 'The home page features an AI chatbot assistant that helps users navigate the app, plan events, and check weather forecasts.'
    },
    'scheduler': {
        'name': 'Event Scheduler',
        'description': 'The Event Scheduler page where users can create, view, and manage their events with weather integration.',
        'content': 'Users can create new events by entering event name, date, time, location, and description. The scheduler shows upcoming events and weather conditions for each event date.'
    },
    'forecaster': {
        'name': 'Weather Forecaster',
        'description': 'The Weather Forecaster page showing detailed weather predictions for different locations.',
        'content': 'Displays detailed weather forecasts including temperature, precipitation chance, humidity, wind speed, and 7-day forecasts for selected locations.'
    }
}

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

# API route for chatbot with context awareness
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    session_id = data.get('session_id', 'default')
    current_page = data.get('current_page', 'home')  # Get current page context
    
    # Initialize conversation history for this session
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    
    # Add user message to history
    conversation_history[session_id].append({
        "role": "user",
        "content": user_message
    })
    
    try:
        # Get current page context
        page_info = PAGE_CONTEXTS.get(current_page, PAGE_CONTEXTS['home'])
        
        # Enhanced system prompt with page context
        system_prompt = f"""You are Rain Assistant, a helpful AI assistant for the 'Rain on Parade' event planning app. 

CURRENT PAGE CONTEXT:
- User is currently on: {page_info['name']}
- Page Description: {page_info['description']}
- Page Content: {page_info['content']}

YOUR CAPABILITIES:
1. Help users plan events and schedules
2. Provide weather forecast information
3. Navigate users to different pages by providing navigation instructions
4. Answer questions about the current page they're viewing
5. Explain features and functionality

NAVIGATION INSTRUCTIONS:
- When users want to go to Scheduler, respond with: "NAVIGATE:scheduler" at the end of your response
- When users want to go to Forecaster/Weather, respond with: "NAVIGATE:forecaster" at the end of your response
- When users want to go to Home, respond with: "NAVIGATE:home" at the end of your response

Be friendly, conversational, and helpful. Keep responses concise (under 100 words) unless detailed information is requested. 
Since you have text-to-speech capabilities, speak naturally as if having a conversation."""

        # Prepare messages
        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ] + conversation_history[session_id]
        
        # Call Groq API
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=500
        )
        
        bot_response = chat_completion.choices[0].message.content
        
        # Check for navigation command
        navigate_to = None
        if "NAVIGATE:" in bot_response:
            parts = bot_response.split("NAVIGATE:")
            bot_response = parts[0].strip()
            navigate_to = parts[1].strip().lower()
        
        # Add bot response to history
        conversation_history[session_id].append({
            "role": "assistant",
            "content": bot_response
        })
        
        # Keep only last 10 messages to avoid token limits
        if len(conversation_history[session_id]) > 10:
            conversation_history[session_id] = conversation_history[session_id][-10:]
        
        return jsonify({
            'success': True,
            'response': bot_response,
            'navigate_to': navigate_to
        })
    
    except Exception as e:
        print(f"Error in chat API: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Sorry, I encountered an error. Please try again.'
        }), 500

# API route to clear chat history
@app.route('/api/clear-chat', methods=['POST'])
def clear_chat():
    data = request.json
    session_id = data.get('session_id', 'default')
    
    if session_id in conversation_history:
        conversation_history[session_id] = []
    
    return jsonify({'success': True})

if __name__ == "__main__":
    app.run(debug=True)
