from flask import jsonify, request, session
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

# ✅ SERVER-SIDE CONVERSATION STORAGE (IN-MEMORY)
conversation_sessions = {}

def register_chatbot_routes(app):
    """Register chatbot routes with the Flask app"""
    
    # Enable sessions
    app.secret_key = os.environ.get("SECRET_KEY", "rain-on-parade-secret-key-2025")
    
    @app.route('/api/chat', methods=['POST'])
    def chat():
        print("\n📨 Received chat request")
        
        try:
            data = request.json
            user_message = data.get('message', '')
            session_id = data.get('session_id', 'default')
            current_page = data.get('current_page', 'home')
            conversation_history = data.get('conversation_history', [])
            
            print(f"💬 User message: {user_message}")
            print(f"🆔 Session ID: {session_id}")
            print(f"📄 Current page: {current_page}")
            print(f"📚 Conversation history length: {len(conversation_history)}")
            
            if not user_message:
                return jsonify({
                    'success': False,
                    'error': 'No message provided'
                }), 400
            
            # ✅ GET OR CREATE CONVERSATION HISTORY FOR THIS SESSION
            if session_id not in conversation_sessions:
                conversation_sessions[session_id] = []
                print(f"🆕 Created new conversation session: {session_id}")
            
            # Use server-side history if frontend didn't send any
            if not conversation_history:
                conversation_history = conversation_sessions[session_id]
                print(f"📥 Using server-side history: {len(conversation_history)} messages")
            
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
            
            # Enhanced system prompt with TRIP PLANNING
            # Enhanced system prompt - TTS OPTIMIZED & TABLE FORMATTED
            system_prompt = f"""You are Rain Assistant, a friendly AI for the 'Rain on Parade' event planning app.

{weather_context}

CURRENT PAGE: {page_info['name']} - {page_info['description']}

YOUR CAPABILITIES:
1. WEATHER PREDICTION: Predict rainfall using ML trained on NASA data
2. TRIP PLANNING: Help plan weather-aware trips
3. EVENT SCHEDULING: Assist with event planning
4. NAVIGATION: Guide users through the app

FORMATTING RULES - SIMPLE & CLEAN:
- Write naturally as if speaking to a friend
- Use numbered lists (1, 2, 3) for places and activities
- Use CAPS LABELS for sections (like "WEATHER:" or "TOP PLACES:")
- NO asterisks, NO bullets, NO markdown formatting
- NO tables - just conversational text with clear sections
- Keep responses flowing and easy to read aloud

RESPONSE FORMAT FOR TRIP PLANNING:
"Great choice! Here's what I'd suggest for [Destination]:

WEATHER: [Brief summary - temps, conditions, rain chance]

TOP PLACES TO VISIT:
1. [Place 1] - [One sentence why]
2. [Place 2] - [One sentence why]
3. [Place 3] - [One sentence why]

ACTIVITIES: [List 3-5 activities in a natural sentence]

WHAT TO PACK: [List 4-6 items in a flowing sentence]

BEST TIMES: [When to visit attractions, written naturally]

For a [X]-day trip, I'd recommend: Day 1 - [activities], Day 2 - [activities], Day 3 - [activities].

Would you like more details about any specific place?"

DESTINATION KNOWLEDGE:
MUNNAR, KERALA (October):
- Weather: 15-22°C, pleasant post-monsoon climate, light rainfall possible
- Peak Season: October to February
- Top Places: Eravikulam National Park (Nilgiri Tahr), Mattupetty Dam (boating), Top Station (sunrise views), Kolukkumalai Tea Estates, Echo Point, Tata Tea Museum
- Activities: Trekking, tea plantation tours, tea tasting, boating, wildlife photography, nature walks
- Pack: Trekking shoes, light jacket, raincoat, warm layers for evenings, camera, power bank
- Duration: Minimum 3-4 days recommended
- Best For: Honeymooners, nature lovers, photographers, adventure seekers

WEATHER RESPONSE FORMAT:
When users ask about weather for a specific date, respond like:
"Great! Let me check the weather for [City] on [Date]. 

FORECAST: [Summary of conditions]
TEMPERATURE: [Range]
RAIN CHANCE: [Percentage]
RECOMMENDATION: [Whether it's good for outdoor activities]

[If trip planning] Based on this forecast, [suggest activities or backup plans]."

MEMORY RULES:
- REMEMBER what the user told you (destination, dates, preferences)
- Reference previous conversation naturally ("As we discussed earlier...")
- Don't ask for info already provided
- Build on previous responses

WEATHER PREDICTION:
- Use the WEATHER PREDICTION DATA above when available
- If no date mentioned, ask politely: "Which date are you planning for?"
- Always mention it's based on ML trained on NASA data
- Give practical recommendations based on rain probability

NAVIGATION COMMANDS (add at END of response):
- To Scheduler: "NAVIGATE:scheduler"
- To Forecaster: "NAVIGATE:forecaster"
- To Home: "NAVIGATE:home"

TONE: Friendly, enthusiastic, helpful - like a knowledgeable travel friend

RESPONSE LENGTH: 100-200 words (150-250 for trip planning)

EXAMPLE GOOD RESPONSE:
"Munnar in October is absolutely perfect! The weather is beautiful with temperatures between 15-22°C, ideal for exploring.

TOP PLACES TO VISIT:
1. Eravikulam National Park - Home to the endangered Nilgiri Tahr, best visited early morning
2. Mattupetty Dam - Perfect for scenic boat rides and photography
3. Top Station - Breathtaking sunrise views of the valley

ACTIVITIES: You can enjoy trekking through tea plantations, tea tasting sessions at local estates, boating at the dam, and wildlife photography.

WHAT TO PACK: Bring comfortable trekking shoes, a light jacket for mornings, warm clothes for evenings, a raincoat just in case, and definitely your camera.

For a 3-day trip: Day 1 - Arrive and explore tea plantations, Day 2 - Visit Eravikulam Park and Top Station for sunrise, Day 3 - Mattupetty Dam and local markets.

Would you like specific recommendations for hotels or restaurants?"

CRITICAL: You have full conversation history. Use it to maintain context."""


            print("🤖 Calling Groq API...")
            
            # Check if API key exists
            if not GROQ_API_KEY:
                print("❌ GROQ_API_KEY is missing!")
                return jsonify({
                    'success': False,
                    'error': 'API key not configured'
                }), 500
            
            # ✅ BUILD MESSAGES WITH FULL CONVERSATION HISTORY
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                }
            ]
            
            # Add conversation history
            for msg in conversation_history:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
            
            # Add current user message
            messages.append({
                "role": "user",
                "content": user_message
            })
            
            print(f"📨 Sending {len(messages)} messages to Groq (including history)")
            
            # Call Groq API
            chat_completion = client.chat.completions.create(
                messages=messages,
                model="llama-3.1-8b-instant",
                temperature=0.7,
                max_tokens=500
            )
            
            bot_response = chat_completion.choices[0].message.content
            print(f"✅ Bot response: {bot_response[:100]}...")
            
            # ✅ SAVE TO SERVER-SIDE HISTORY
            conversation_sessions[session_id].append({
                "role": "user",
                "content": user_message
            })
            conversation_sessions[session_id].append({
                "role": "assistant",
                "content": bot_response
            })
            
            # Keep only last 20 messages (10 exchanges) to prevent memory overflow
            if len(conversation_sessions[session_id]) > 20:
                conversation_sessions[session_id] = conversation_sessions[session_id][-20:]
            
            print(f"💾 Saved to server history. Total messages: {len(conversation_sessions[session_id])}")
            
            # Parse navigation command from response
            navigate_to = None
            forecast_city = None
            forecast_date = None
            
            if 'NAVIGATE:' in bot_response:
                nav_match = re.search(r'NAVIGATE:(\w+)', bot_response)
                if nav_match:
                    navigate_to = nav_match.group(1).lower()
                    
                    # If navigating to forecaster, extract city and date from user message
                    if navigate_to == 'forecaster':
                        forecast_city = extract_city_from_message(user_message)
                        forecast_date = extract_date_from_message(user_message)
                    
                    # Remove NAVIGATE command from response
                    bot_response = re.sub(r'\s*NAVIGATE:\w+\s*', '', bot_response).strip()
                    print(f"🧭 Navigation detected: {navigate_to}")
                    if forecast_city:
                        print(f"  📍 City: {forecast_city}, Date: {forecast_date}")
            
            # Create response data
            response_data = {
                'success': True,
                'response': bot_response,
                'conversation_history': conversation_sessions[session_id]  # ✅ SEND BACK FULL HISTORY
            }
            
            # Add navigate_to if navigation was requested
            if navigate_to:
                response_data['navigate_to'] = navigate_to
                if forecast_city:
                    response_data['forecast_city'] = forecast_city
                if forecast_date:
                    response_data['forecast_date'] = forecast_date
            
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
    
    @app.route('/api/chat/clear', methods=['POST'])
    def clear_conversation():
        """Clear conversation history for a session"""
        try:
            data = request.json
            session_id = data.get('session_id', 'default')
            
            if session_id in conversation_sessions:
                del conversation_sessions[session_id]
                print(f"🗑️ Cleared conversation history for session: {session_id}")
            
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
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
    """Extract date from user message"""
    # Try "oct 10" or "10 oct" format FIRST
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
    
    print("⚠️ Could not extract date from message")
    return None

def extract_city_from_message(message):
    """Extract city from user message - WITH MUNNAR SUPPORT"""
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
        
        # Kerala cities
        'kerala': 'Thiruvananthapuram',
        'thiruvananthapuram': 'Thiruvananthapuram',
        'trivandrum': 'Thiruvananthapuram',
        'kochi': 'Kochi',
        'cochin': 'Kochi',
        'kozhikode': 'Kozhikode',
        'calicut': 'Kozhikode',
        'thrissur': 'Thrissur',
        'kannur': 'Kannur',
        'kollam': 'Kollam',
        'palakkad': 'Palakkad',
        'munnar': 'Kochi',  # ✅ MUNNAR MAPS TO KOCHI (NEAREST CITY)
    }
    
    message_lower = message.lower()
    for city_key, city_name in cities.items():
        if city_key in message_lower:
            print(f"✅ City detected: {city_name}")
            return city_name
    
    print("ℹ️ No city detected, using default: Mumbai")
    return 'Mumbai'  # Default
