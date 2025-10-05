from flask import jsonify, request
import os
from groq import Groq
from dotenv import load_dotenv
import re
from datetime import datetime
import traceback

# Load environment variables
load_dotenv()

# Initialize Groq client
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
print(f"🔑 GROQ_API_KEY loaded: {'Yes' if GROQ_API_KEY else 'NO - MISSING!'}")

client = Groq(api_key=GROQ_API_KEY)

# Import the weather predictor
from rainfall_predictor import RainfallPredictor

# Load the trained model
weather_predictor = RainfallPredictor()
if os.path.exists('weather_model.pkl'):
    weather_predictor.load_model()
    print("✅ Weather prediction model loaded in chatbot")


def register_chatbot_routes(app):
    """Register chatbot routes with the Flask app"""
    
    @app.route('/api/chat', methods=['POST'])
    def chat():
        print("\n📨 Received chat request")
        try:
            data = request.json
            user_message = data.get('message', '')
            session_id = data.get('session_id', 'default')
            current_page = data.get('current_page', 'home')
            
            print(f"💬 User message: {user_message}")
            print(f"📄 Current page: {current_page}")
            
            if not user_message:
                return jsonify({
                    'success': False,
                    'error': 'No message provided'
                }), 400
            
            # Check if user is asking about weather prediction
            weather_keywords = ['rain', 'weather', 'forecast', 'predict', 'will it rain', 'precipitation', 'rainfall']
            is_weather_query = any(keyword in user_message.lower() for keyword in weather_keywords)
            
            weather_context = ""
            
            if is_weather_query and weather_predictor.model is not None:
                print("🌧️ Weather query detected")
                # Try to extract date and city from message
                date_str = extract_date_from_message(user_message)
                city = extract_city_from_message(user_message)
                
                print(f"📅 Extracted date: {date_str}")
                print(f"🌆 Extracted city: {city}")
                
                if date_str:
                    # Make weather prediction
                    try:
                        print(f"🔮 Making prediction for {city} on {date_str}")
                        prediction = weather_predictor.predict(city=city, date_str=date_str)
                        
                        if 'error' not in prediction:
                            print(f"✅ Prediction successful: {prediction['rain_probability']}% rain")
                            weather_context = f"""
WEATHER PREDICTION DATA (use this to answer the user):
- Date: {prediction['date']}
- City: {prediction['city']}
- Predicted Rainfall: {prediction['predicted_rainfall_mm']} mm
- Rain Probability: {prediction['rain_probability']}%
- Recommendation: {prediction['recommendation']}

Use this data to give a natural, conversational response about the weather.
"""
                        else:
                            print(f"⚠️ Prediction error: {prediction['error']}")
                    except Exception as e:
                        print(f"❌ Weather prediction error: {e}")
                        traceback.print_exc()
            
            # Define page_info based on current_page
            page_info = {
                'home': {
                    'name': 'Home',
                    'description': 'Welcome page for Rain on Parade event planning app.',
                    'content': 'Overview, navigation, and quick access to features.'
                },
                'scheduler': {
                    'name': 'Scheduler',
                    'description': 'Schedule your events and check for rain predictions.',
                    'content': 'Event scheduling form and calendar.'
                },
                'forecaster': {
                    'name': 'Forecaster',
                    'description': 'Get weather forecasts and rain predictions.',
                    'content': 'Weather prediction tools and results.'
                }
            }.get(current_page, {
                'name': current_page.capitalize(),
                'description': 'No description available.',
                'content': 'No content available.'
            })
            
            # Enhanced system prompt
            system_prompt = f"""You are Rain Assistant, a helpful AI for the 'Rain on Parade' event planning app.

{weather_context}

CURRENT PAGE CONTEXT:
- User is currently on: {page_info['name']}
- Page Description: {page_info['description']}
- Page Content: {page_info['content']}

YOUR CAPABILITIES:
1. **WEATHER PREDICTION**: Predict rainfall for any date up to 1 year ahead using ML trained on NASA data
2. Help users plan events and schedules
3. Provide weather forecast information
4. Navigate users to different pages
5. Answer questions about the current page

WEATHER PREDICTION INSTRUCTIONS:
- When users ask "Will it rain on [date]?", use the WEATHER PREDICTION DATA above if available
- If no date is mentioned, ask them to specify a date
- You use Machine Learning trained on 10+ years of NASA weather data
- Predictions are most accurate for 90 days ahead

NAVIGATION INSTRUCTIONS:
- When users want to go to Scheduler, respond with: "NAVIGATE:scheduler" at the end
- When users want to go to Forecaster/Weather, respond with: "NAVIGATE:forecaster" at the end
- When users want to go to Home, respond with: "NAVIGATE:home" at the end

Be friendly, conversational, and helpful. Keep responses concise (under 100 words) unless detailed information is requested.
Since you have text-to-speech capabilities, speak naturally as if having a conversation."""
            
            print("🤖 Calling Groq API...")
            
            # Check if API key exists
            if not GROQ_API_KEY:
                print("❌ GROQ_API_KEY is missing!")
                return jsonify({
                    'success': False,
                    'error': 'API key not configured'
                }), 500
            
            # Call Groq API
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_message
                    }
                ],
                model="llama-3.1-8b-instant",
                temperature=0.7,
                max_tokens=500
            )
            
            bot_response = chat_completion.choices[0].message.content
            print(f"✅ Bot response: {bot_response[:100]}...")
            
            # IMPORTANT: Parse navigation command from response
            navigate_to = None
            if 'NAVIGATE:' in bot_response:
                nav_match = re.search(r'NAVIGATE:(\w+)', bot_response)
                if nav_match:
                    navigate_to = nav_match.group(1).lower()
                    # Remove NAVIGATE command from response
                    bot_response = re.sub(r'\s*NAVIGATE:\w+\s*', '', bot_response).strip()
                    print(f"🧭 Navigation detected: {navigate_to}")
            
            response_data = {
                'success': True,
                'response': bot_response
            }
            
            # Add navigate_to if navigation was requested
            if navigate_to:
                response_data['navigate_to'] = navigate_to
            
            return jsonify(response_data)
            
        except Exception as e:
            print(f"\n❌ ERROR in chat endpoint:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            traceback.print_exc()
            
            return jsonify({
                'success': False,
                'error': f"{type(e).__name__}: {str(e)}"
            }), 500
    
    
    @app.route('/api/tts', methods=['POST'])
    def text_to_speech():
        try:
            data = request.json
            text = data.get('text', '')
            
            if not text:
                return jsonify({
                    'success': False,
                    'error': 'No text provided'
                }), 400
            
            return jsonify({
                'success': True,
                'message': 'Using browser TTS'
            })
            
        except Exception as e:
            print(f"Error in TTS endpoint: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    
    @app.route('/api/stt', methods=['POST'])
    def speech_to_text():
        try:
            audio_file = request.files.get('audio')
            
            if not audio_file:
                return jsonify({
                    'success': False,
                    'error': 'No audio file provided'
                }), 400
            
            return jsonify({
                'success': True,
                'text': 'Using browser STT'
            })
            
        except Exception as e:
            print(f"Error in STT endpoint: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    print("✅ Chatbot routes registered")


def extract_date_from_message(message):
    """Extract date from user message - IMPROVED VERSION"""
    
    # Try "oct 10" or "10 oct" format FIRST (most common)
    simple_pattern = r'(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)|' \
                    r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+(\d{1,2})'
    match = re.search(simple_pattern, message.lower())
    if match:
        groups = match.groups()
        if groups[0] and groups[1]:  # "10 oct"
            day, month_abbr = groups[0], groups[1]
        elif groups[2] and groups[3]:  # "oct 10"
            month_abbr, day = groups[2], groups[3]
        else:
            month_abbr, day = None, None
            
        if month_abbr and day:
            month_map = {
                'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
                'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
                'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
            }
            month = month_map.get(month_abbr.lower(), '01')
            result = f"2025-{month}-{day.zfill(2)}"
            print(f"✅ Date extraction successful: {result}")
            return result
    
    # Try YYYY-MM-DD format
    iso_pattern = r'(\d{4})-(\d{2})-(\d{2})'
    match = re.search(iso_pattern, message)
    if match:
        return match.group(0)
    
    # Try DD/MM/YYYY or MM/DD/YYYY
    slash_pattern = r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})'
    match = re.search(slash_pattern, message)
    if match:
        day, month, year = match.groups()
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    
    # Try full month names
    month_pattern = r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?'
    match = re.search(month_pattern, message.lower())
    if match:
        month_name, day, year = match.groups()
        month_map = {
            'january': '01', 'february': '02', 'march': '03',
            'april': '04', 'may': '05', 'june': '06',
            'july': '07', 'august': '08', 'september': '09',
            'october': '10', 'november': '11', 'december': '12'
        }
        month = month_map.get(month_name.lower(), '01')
        year = year if year else '2025'
        return f"{year}-{month}-{day.zfill(2)}"
    
    print("⚠️ Could not extract date from message")
    return None


def extract_city_from_message(message):
    """Extract city from user message - WITH KERALA SUPPORT"""
    cities = {
        # Major Indian cities
        'mumbai': 'Mumbai',
        'delhi': 'Delhi',
        'bangalore': 'Bangalore',
        'chennai': 'Chennai',
        'kolkata': 'Kolkata',
        'hyderabad': 'Hyderabad',
        'pune': 'Pune',
        'ahmedabad': 'Ahmedabad',
        'jaipur': 'Jaipur',
        'lucknow': 'Lucknow',
        # Kerala cities
        'kerala': 'Thiruvananthapuram',  # Map state to capital
        'thiruvananthapuram': 'Thiruvananthapuram',
        'trivandrum': 'Thiruvananthapuram',
        'kochi': 'Kochi',
        'cochin': 'Kochi',
        'kozhikode': 'Kozhikode',
        'calicut': 'Kozhikode',
        'thrissur': 'Thrissur',
        'kannur': 'Kannur',
        'kollam': 'Kollam',
        'palakkad': 'Palakkad'
    }
    
    message_lower = message.lower()
    for city_key, city_name in cities.items():
        if city_key in message_lower:
            print(f"✅ City detected: {city_name}")
            return city_name
    
    print("ℹ️ No city detected, using default: Mumbai")
    return 'Mumbai'  # Default
