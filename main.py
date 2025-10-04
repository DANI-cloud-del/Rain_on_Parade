from flask import Flask, render_template, jsonify, request
import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Store conversation history (in production, use a database)
conversation_history = {}

# Main route for home page
@app.route("/")
def home():
    return render_template("HomePage.html")

# API route for chatbot
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    session_id = data.get('session_id', 'default')
    
    # Initialize conversation history for this session
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    
    # Add user message to history
    conversation_history[session_id].append({
        "role": "user",
        "content": user_message
    })
    
    try:
        # Prepare messages with system prompt
        messages = [
            {
                "role": "system",
                "content": """You are Rain Assistant, a helpful AI assistant for the 'Rain on Parade' event planning app. 
                Your role is to help users:
                - Plan events and schedules
                - Check weather forecasts
                - Navigate through the app
                - Answer questions about event planning
                
                Be friendly, concise, and helpful. If users want to navigate to Scheduler or Forecaster pages, 
                acknowledge their request and let them know you'll help them navigate there.
                Keep responses under 100 words unless specifically asked for detailed information."""
            }
        ] + conversation_history[session_id]
        
        # Call Groq API with updated model
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",  # Updated to current supported model
            temperature=0.7,
            max_tokens=500
        )
        
        bot_response = chat_completion.choices[0].message.content
        
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
            'response': bot_response
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
