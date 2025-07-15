from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import threading
import json
import os
from datetime import datetime
import logging

# Import your existing Jarvis class
from jarvis_ai import JarvisAI  # Make sure this matches your file name

app = Flask(__name__)
CORS(app)  # Enable CORS for web interface

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global Jarvis instance
jarvis = None

def initialize_jarvis():
    """Initialize Jarvis AI with API keys"""
    global jarvis
    
    # Get API keys from environment variables
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    
    if not GEMINI_API_KEY or not GROQ_API_KEY:
        logger.error("GEMINI_API_KEY and GROQ_API_KEY environment variables must be set.")
        return False

    try:
        jarvis = JarvisAI(GEMINI_API_KEY, GROQ_API_KEY)
        logger.info("Jarvis AI initialized successfully!")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Jarvis: {e}")
        return False

@app.route('/')
def index():
    """Serve the web interface"""
    # You can serve the HTML file directly or return it as a string
    # For now, we'll redirect to serve the static HTML file
    return app.send_static_file('index.html')

@app.route('/api/conversation', methods=['POST'])
def handle_conversation():
    """Handle conversation requests from the web interface"""
    global jarvis
    
    if not jarvis:
        return jsonify({'error': 'Jarvis not initialized'}), 500
    
    try:
        data = request.get_json()
        user_input = data.get('user_input', '')
        
        if not user_input:
            return jsonify({'error': 'No input provided'}), 400
        
        # Process the user input with Jarvis
        response = jarvis.process_user_input(user_input)
        
        resp = {
            'user_input': user_input,
            'ai_response': response,
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        return jsonify(resp)
    
    except Exception as e:
        logger.error(f"Error processing conversation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/tasks', methods=['GET', 'POST'])
def handle_tasks():
    """Handle task-related requests"""
    global jarvis

    if not jarvis:
        return jsonify({'error': 'Jarvis not initialized'}), 500

    try:
        if request.method == 'GET':
            # Always use agent_list_tasks to get all tasks as a formatted string and as objects
            tasks = jarvis.get_tasks()
            tasks_list_str = jarvis.agent_list_tasks()
            return jsonify({
                'tasks': tasks,
                'tasks_list': tasks_list_str,
                'status': 'success'
            })

        elif request.method == 'POST':
            data = request.get_json()
            description = data.get('description', '')
            due_date = data.get('due_date')
            priority = data.get('priority', 'medium')
            if not description:
                return jsonify({'error': 'Task description required'}), 400
            result = jarvis.add_task(description, due_date, priority)
            # Return all tasks after adding
            tasks = jarvis.get_tasks()
            tasks_list_str = jarvis.agent_list_tasks()
            return jsonify({
                'message': result,
                'tasks': tasks,
                'tasks_list': tasks_list_str,
                'status': 'success'
            })

    except Exception as e:
        logger.error(f"Error handling tasks: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/tasks/update', methods=['PUT'])
def update_task():
    """Update task status"""
    global jarvis
    
    if not jarvis:
        return jsonify({'error': 'Jarvis not initialized'}), 500
    
    try:
        data = request.get_json()
        task_id = data.get('task_id')
        new_status = data.get('status')
        
        if not task_id or not new_status:
            return jsonify({'error': 'Task ID and status required'}), 400
        
        result = jarvis.update_task_status(int(task_id), new_status)
        
        return jsonify({
            'message': result,
            'status': 'success'
        })
    
    except Exception as e:
        logger.error(f"Error updating task: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/reminders', methods=['GET', 'POST'])
def handle_reminders():
    """Handle reminder-related requests"""
    global jarvis
    
    if not jarvis:
        return jsonify({'error': 'Jarvis not initialized'}), 500
    
    try:
        if request.method == 'GET':
            # Return all reminders, not just active
            return jsonify({
                'reminders': jarvis.reminders,
                'status': 'success'
            })
        
        elif request.method == 'POST':
            # Add new reminder
            data = request.get_json()
            text = data.get('text', '')
            time = data.get('time', '')
            
            if not text:
                return jsonify({'error': 'Reminder text required'}), 400
            
            # If no time provided, use current time + 1 hour as default
            if not time:
                from datetime import datetime, timedelta
                default_time = datetime.now() + timedelta(hours=1)
                time = default_time.strftime("%H:%M")
            
            result = jarvis.add_reminder(text, time)
            # Return updated reminders for instant UI update
            reminders = jarvis.reminders
            
            return jsonify({
                'message': result,
                'reminders': reminders,
                'status': 'success'
            })
    
    except Exception as e:
        logger.error(f"Error handling reminders: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/conversations', methods=['GET'])
def get_conversations():
    """Get conversation history"""
    global jarvis
    
    if not jarvis:
        return jsonify({'error': 'Jarvis not initialized'}), 500
    
    try:
        # Get recent conversations (last 50)
        recent_conversations = jarvis.conversations[-50:] if len(jarvis.conversations) > 50 else jarvis.conversations
        
        return jsonify({
            'conversations': recent_conversations,
            'total': len(jarvis.conversations),
            'status': 'success'
        })
    
    except Exception as e:
        logger.error(f"Error getting conversations: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['POST'])
def search_conversations():
    """Search through conversation history"""
    global jarvis
    
    if not jarvis:
        return jsonify({'error': 'Jarvis not initialized'}), 500
    
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'Search query required'}), 400
        
        # Use Jarvis's vector search functionality
        results = jarvis.search_vector_db(query, k=10)
        
        return jsonify({
            'results': results,
            'query': query,
            'status': 'success'
        })
    
    except Exception as e:
        logger.error(f"Error searching conversations: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status"""
    global jarvis
    
    try:
        status_info = {
            'jarvis_initialized': jarvis is not None,
            'total_conversations': len(jarvis.conversations) if jarvis else 0,
            'total_tasks': len(jarvis.tasks) if jarvis else 0,
            'total_reminders': len(jarvis.reminders) if jarvis else 0,
            'active_reminders': len([r for r in jarvis.reminders if r.get('status') == 'active']) if jarvis else 0,
            'pending_tasks': len([t for t in jarvis.tasks if t.get('status') == 'pending']) if jarvis else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(status_info)
    
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/voice/process', methods=['POST'])
def process_voice():
    """Process voice input (for future voice integration)"""
    global jarvis
    
    if not jarvis:
        return jsonify({'error': 'Jarvis not initialized'}), 500
    
    try:
        # This endpoint can be used for processing voice data
        # For now, we'll just return a placeholder
        return jsonify({
            'message': 'Voice processing endpoint ready',
            'status': 'success'
        })
    
    except Exception as e:
        logger.error(f"Error processing voice: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/user-context', methods=['GET'])
def get_user_context():
    """Get user context and preferences"""
    global jarvis
    
    if not jarvis:
        return jsonify({'error': 'Jarvis not initialized'}), 500
    
    try:
        # Get user context from Jarvis
        context = jarvis.user_context
        
        # Get top frequent topics
        frequent_topics = sorted(context.get('frequent_topics', {}).items(), 
                               key=lambda x: x[1], reverse=True)[:10]
        
        return jsonify({
            'preferences': context.get('preferences', {}),
            'frequent_topics': frequent_topics,
            'conversation_count': len(context.get('conversation_history', [])),
            'status': 'success'
        })
    
    except Exception as e:
        logger.error(f"Error getting user context: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/tasks/all', methods=['GET'])
def get_all_tasks():
    """Get all tasks (no filtering, always returns all tasks as objects and as formatted string)"""
    global jarvis

    if not jarvis:
        return jsonify({'error': 'Jarvis not initialized'}), 500

    try:
        tasks = jarvis.get_tasks()
        tasks_list_str = jarvis.agent_list_tasks()
        return jsonify({
            'tasks': tasks,
            'tasks_list': tasks_list_str,
            'status': 'success'
        })
    except Exception as e:
        logger.error(f"Error getting all tasks: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

def run_jarvis_continuous():
    """Run Jarvis in continuous mode in a separate thread"""
    global jarvis
    if jarvis:
        try:
            # You can uncomment this if you want voice mode running alongside web interface
            # jarvis.run_continuous_mode()
            pass
        except Exception as e:
            logger.error(f"Error running Jarvis continuous mode: {e}")

if __name__ == '__main__':
    # Initialize Jarvis
    if initialize_jarvis():
        logger.info("Starting Flask server...")
        
        # Optionally start Jarvis in continuous mode in a separate thread
        # jarvis_thread = threading.Thread(target=run_jarvis_continuous, daemon=True)
        # jarvis_thread.start()
        
        # Run Flask app
        app.run(host='0.0.0.0', port=4000, debug=True)
    else:
        logger.error("Failed to initialize Jarvis. Please check your API keys.")
        print("Please set your API keys:")
        print("export GEMINI_API_KEY='your_gemini_api_key'")
        print("export GROQ_API_KEY='your_groq_api_key'")
        print("Or modify the keys directly in the code.")