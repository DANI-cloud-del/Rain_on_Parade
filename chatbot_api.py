from flask import jsonify, request, current_app
import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Import app from main (or use current_app)
from main import app

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({
                'success': False,
                'error': 'No message provided'
            }), 400
        
        # Call Groq API
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are Rain Assistant, a helpful AI that helps users plan events and check weather forecasts. Be friendly, concise, and helpful. Keep responses under 100 words unless asked for detailed information."
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            model="mixtral-8x7b-32768",
            temperature=0.7,
            max_tokens=500
        )
        
        bot_response = chat_completion.choices[0].message.content
        
        return jsonify({
            'success': True,
            'response': bot_response
        })
    
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
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
        
        # TODO: Implement TTS using Eleven Labs or other service
        # For now, return success (browser TTS will handle it)
        
        return jsonify({
            'success': True,
            'message': 'TTS not implemented yet, using browser TTS'
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
        
        # TODO: Implement STT using Groq Whisper or other service
        # For now, return error (browser STT will handle it)
        
        return jsonify({
            'success': True,
            'text': 'STT not implemented yet, using browser STT'
        })
    
    except Exception as e:
        print(f"Error in STT endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
