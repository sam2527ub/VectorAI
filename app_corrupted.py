# Multi-Agent Chat Interface - Flask App
# File: app.py

from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
import uuid
import json
import os
from datetime import datetime
from typing import Dict, List, Any
import sqlite3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the agent system
from chatbot6 import (
    CollaborationFixedSession, 
    domain_classifier,
    therapist_kb,
                # Clean main content - remove all metadata
            main_content = content
            metadata_patterns = [
                r'🎬\s*\*\*Citation:?\*\*.*?(?=\n💭|\n🔗|\n🎬|$)',
                r'🎬\s*\*\*Source Inspiration\*\*:.*?(?=\n💭|\n🔗|$)',
                r'💭\s*\*\*(?:Reasoning(?:\s+Process)?|My approach|Thought process|Wisdom approach):?\*\*.*?(?=\n🔗|\n🎬|$)',
                r'🔗\s*\*\*(?:Attribution|Building on|Connecting to|Drawing from).*?(?=\n💭|\n🎬|$)',
                r'💬.*?(?=\n|$)',
                r'🔗\s*Shared.*insights.*'
            ]
            
            for pattern in metadata_patterns:
                main_content = re.sub(pattern, '', main_content, flags=re.DOTALL | re.IGNORECASE)b, 
    wise_kb,
    knowledge_manager,
    llm
)
from langchain_core.messages import SystemMessage

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# Database setup for chat sessions
def init_db():
    conn = sqlite3.connect('chat_sessions.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            agent_type TEXT,
            inspiration TEXT,
            movie TEXT,
            citation TEXT,
            reasoning TEXT,
            attribution TEXT,
            source_inspiration TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions (id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database
init_db()

# Global storage for active chat sessions
active_sessions: Dict[str, CollaborationFixedSession] = {}

class ChatSessionManager:
    @staticmethod
    def create_session() -> str:
        session_id = str(uuid.uuid4())
        active_sessions[session_id] = CollaborationFixedSession()
        
        # Save to database
        conn = sqlite3.connect('chat_sessions.db')
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO sessions (id, title) VALUES (?, ?)',
            (session_id, f"Chat {datetime.now().strftime('%m/%d %H:%M')}")
        )
        conn.commit()
        conn.close()
        
        return session_id
    
    @staticmethod
    def get_session(session_id: str) -> CollaborationFixedSession:
        if session_id not in active_sessions:
            # Load from database and recreate session
            active_sessions[session_id] = CollaborationFixedSession()
            ChatSessionManager.load_session_history(session_id)
        return active_sessions[session_id]
    
    @staticmethod
    def save_message(session_id: str, message_data: dict):
        conn = sqlite3.connect('chat_sessions.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO messages (session_id, role, content, agent_type, inspiration, movie, citation, reasoning, attribution, source_inspiration)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id,
            message_data.get('role', ''),
            message_data.get('content', ''),
            message_data.get('agent_type', ''),
            message_data.get('inspiration', ''),
            message_data.get('movie', ''),
            message_data.get('citation', ''),
            message_data.get('reasoning', ''),
            message_data.get('attribution', ''),
            message_data.get('source_inspiration', '')
        ))
        
        # Update session timestamp
        cursor.execute(
            'UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?',
            (session_id,)
        )
        
        conn.commit()
        conn.close()
    
    @staticmethod
    def load_session_history(session_id: str):
        conn = sqlite3.connect('chat_sessions.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT role, content, agent_type, inspiration, movie, citation, reasoning, timestamp
            FROM messages WHERE session_id = ? ORDER BY timestamp
        ''', (session_id,))
        
        messages = cursor.fetchall()
        conn.close()
        
        if session_id in active_sessions:
            session = active_sessions[session_id]
            for msg in messages:
                role, content, agent_type, inspiration, movie, citation, reasoning, timestamp = msg
                session.conversation.append({
                    'role': role,
                    'content': content,
                    'agent_type': agent_type,
                    'inspiration': inspiration,
                    'movie': movie,
                    'citation': citation,
                    'reasoning': reasoning,
                    'timestamp': timestamp
                })
    
    @staticmethod
    def get_all_sessions():
        conn = sqlite3.connect('chat_sessions.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT s.id, s.title, s.updated_at,
                   (SELECT content FROM messages WHERE session_id = s.id AND role = 'user' ORDER BY timestamp DESC LIMIT 1) as last_message
            FROM sessions s ORDER BY s.updated_at DESC
        ''')
        
        sessions = cursor.fetchall()
        conn.close()
        
        return [{'id': s[0], 'title': s[1], 'updated_at': s[2], 'last_message': s[3]} for s in sessions]

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    sessions = ChatSessionManager.get_all_sessions()
    return jsonify(sessions)

@app.route('/api/sessions', methods=['POST'])
def create_session():
    session_id = ChatSessionManager.create_session()
    return jsonify({'session_id': session_id})

@app.route('/api/sessions/<session_id>/messages', methods=['GET'])
def get_session_messages(session_id):
    conn = sqlite3.connect('chat_sessions.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT role, content, agent_type, inspiration, movie, citation, reasoning, timestamp
        FROM messages WHERE session_id = ? ORDER BY timestamp
    ''', (session_id,))
    
    messages = cursor.fetchall()
    conn.close()
    
    formatted_messages = []
    for msg in messages:
        role, content, agent_type, inspiration, movie, citation, reasoning, timestamp = msg
        formatted_messages.append({
            'role': role,
            'content': content,
            'agent_type': agent_type,
            'inspiration': inspiration,
            'movie': movie,
            'citation': citation,
            'reasoning': reasoning,
            'timestamp': timestamp
        })
    
    return jsonify(formatted_messages)

@socketio.on('send_message')
def handle_message(data):
    session_id = data['session_id']
    message = data['message']
    mode = data.get('mode', 'auto')  # auto, therapist, expert, wise, collaborative
    
    # Get or create session
    chat_session = ChatSessionManager.get_session(session_id)
    
    # Save user message
    user_message = {
        'role': 'user',
        'content': message,
        'agent_type': 'user',
        'timestamp': datetime.now().isoformat()
    }
    
    ChatSessionManager.save_message(session_id, user_message)
    
    # Emit user message to client
    emit('message_received', {
        'type': 'user',
        'content': message,
        'timestamp': user_message['timestamp']
    })
    
    # Process message based on mode
    if mode == 'auto':
        # Auto-detect domain
        domain_analysis = domain_classifier.analyze_query_domain(message)
        
        if domain_analysis['collaboration_needed']:
            # Collaborative mode - all agents respond
            emit('typing', {'agents': ['therapist', 'expert', 'wise_mentor']})
            
            # Show inter-agent collaboration discussion first
            emit('agent_collaboration', {
                'type': 'collaboration_start',
                'message': 'Agents are collaborating on your question...'
            })
            
            # Process through all agents
            agents = ['therapist', 'intelligent_expert', 'wise_mentor']
            agent_responses = []
            
            for agent in agents:
                response = process_agent_response(chat_session, message, agent, session_id)
                if response:
                    agent_responses.append(response)
                    emit('agent_response', response)
            
            # Generate inter-agent conversation after all responses
            inter_agent_conversation = generate_inter_agent_conversation(agent_responses, message)
            if inter_agent_conversation:
                emit('inter_agent_conversation', inter_agent_conversation)
                
        else:
            # Single agent mode
            primary_agent = domain_analysis['primary_domain']
            emit('typing', {'agents': [primary_agent]})
            
            response = process_agent_response(chat_session, message, primary_agent, session_id)
            emit('agent_response', response)
    
    elif mode in ['therapist', 'expert', 'wise']:
        # Explicit agent selection
        agent_map = {
            'therapist': 'therapist',
            'expert': 'intelligent_expert', 
            'wise': 'wise_mentor'
        }
        
        selected_agent = agent_map[mode]
        emit('typing', {'agents': [selected_agent]})
        
        response = process_agent_response(chat_session, message, selected_agent, session_id)
        emit('agent_response', response)
    
    elif mode == 'collaborative':
        # Force collaborative mode
        emit('typing', {'agents': ['therapist', 'expert', 'wise_mentor']})
        
        # Show collaboration start
        emit('agent_collaboration', {
            'type': 'collaboration_start',
            'message': 'All agents are working together on your question...'
        })
        
        agents = ['therapist', 'intelligent_expert', 'wise_mentor']
        agent_responses = []
        
        for agent in agents:
            response = process_agent_response(chat_session, message, agent, session_id)
            if response:
                agent_responses.append(response)
                emit('agent_response', response)
        
        # Generate inter-agent conversation
        inter_agent_conversation = generate_inter_agent_conversation(agent_responses, message)
        if inter_agent_conversation:
            emit('inter_agent_conversation', inter_agent_conversation)

def generate_inter_agent_conversation(agent_responses, user_message):
    """Generate a natural conversation between agents about their responses"""
    if len(agent_responses) < 2:
        return None
    
    # Create a conversation prompt
    conversation_prompt = f"""
You are simulating a natural conversation between three professional agents who just responded to a user's question: "{user_message}"

The agents are:
- 😊 Therapist: Focuses on emotional support
- 🎓 Expert: Focuses on practical solutions  
- 🧠 Wise Mentor: Focuses on life wisdom

Here's what each agent told the user:

THERAPIST: {agent_responses[0]['content'][:200] if len(agent_responses) > 0 else ''}...

EXPERT: {agent_responses[1]['content'][:200] if len(agent_responses) > 1 else ''}...

WISE MENTOR: {agent_responses[2]['content'][:200] if len(agent_responses) > 2 else ''}...

Now create a natural, brief conversation where the agents discuss their approaches and build on each other's insights. Make it feel like colleagues collaborating.

Format as:
THERAPIST: [comment about expert/wise mentor's approach]
EXPERT: [building on therapist's point, adding practical angle]
WISE MENTOR: [synthesizing both perspectives with wisdom]
THERAPIST: [brief agreement or additional insight]

Keep it conversational and under 300 words total.
"""
    
    try:
        response = llm.invoke([SystemMessage(content=conversation_prompt)])
        conversation_text = response.content.strip()
        
        # Parse the conversation into individual messages
        conversation_messages = []
        lines = conversation_text.split('\n')
        
        current_agent = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('THERAPIST:'):
                if current_agent and current_content:
                    conversation_messages.append({
                        'agent': current_agent,
                        'content': ' '.join(current_content)
                    })
                current_agent = 'therapist'
                current_content = [line[11:].strip()]
            elif line.startswith('EXPERT:'):
                if current_agent and current_content:
                    conversation_messages.append({
                        'agent': current_agent,
                        'content': ' '.join(current_content)
                    })
                current_agent = 'intelligent_expert'
                current_content = [line[7:].strip()]
            elif line.startswith('WISE MENTOR:'):
                if current_agent and current_content:
                    conversation_messages.append({
                        'agent': current_agent,
                        'content': ' '.join(current_content)
                    })
                current_agent = 'wise_mentor'
                current_content = [line[12:].strip()]
            elif line and current_agent:
                current_content.append(line)
        
        # Add the last message
        if current_agent and current_content:
            conversation_messages.append({
                'agent': current_agent,
                'content': ' '.join(current_content)
            })
        
        return {
            'type': 'inter_agent_conversation',
            'messages': conversation_messages,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error generating inter-agent conversation: {e}")
        return None

def process_agent_response(chat_session, message, agent_type, session_id):
    """Process agent response and return formatted data"""
    
    # Create the state for the agent
    state = {
        'user_input': message,
        'conversation': chat_session.conversation,
        'collaboration_mode': False,
        'messages': [],
        'turn': 1,
        'last_speaker': 'user',
        'target_agent': agent_type,
        'original_query': message,
        'shared_knowledge': {},
        'agent_questions': [],
        'inter_agent_dialogue': [],
        'agents_completed': []
    }
    
    try:
        # Process based on agent type
        if agent_type == 'therapist':
            from chatbot6 import therapist_agent
            result = therapist_agent(state)
        elif agent_type == 'intelligent_expert':
            from chatbot6 import intelligent_expert_agent
            result = intelligent_expert_agent(state)
        elif agent_type == 'wise_mentor':
            from chatbot6 import wise_mentor_agent
            result = wise_mentor_agent(state)
        else:
            return None
        
        # Extract response data
        if result and 'messages' in result and result['messages']:
            agent_msg = result['messages'][-1]
            
            # Parse citation and reasoning from content
            content = agent_msg['content']
            citation = ""
            reasoning = ""
            attribution = ""
            source_inspiration = ""
            
            # Extract all metadata sections
            import re
            
            # Extract Citation (more flexible pattern)
            citation_match = re.search(r'🎬\s*\*\*Citation:?\*\*.*?(?=\n💭|\n🔗|\n🎬|$)', content, re.DOTALL | re.IGNORECASE)
            if citation_match:
                citation = citation_match.group(0).strip()
            
            # Extract Source Inspiration
            source_match = re.search(r'🎬\s*\*\*Source Inspiration\*\*:.*?(?=\n💭|\n🔗|$)', content, re.DOTALL | re.IGNORECASE)
            if source_match:
                source_inspiration = source_match.group(0).strip()
            
            # Extract Reasoning (more flexible patterns)
            reasoning_patterns = [
                r'💭\s*\*\*Reasoning(?:\s+Process)?:?\*\*.*?(?=\n🔗|\n🎬|$)',
                r'💭\s*\*\*My approach:?\*\*.*?(?=\n🔗|\n🎬|$)',
                r'💭\s*\*\*Thought process:?\*\*.*?(?=\n🔗|\n🎬|$)',
                r'💭\s*\*\*Wisdom approach:?\*\*.*?(?=\n🔗|\n🎬|$)'
            ]
            for pattern in reasoning_patterns:
                reasoning_match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
                if reasoning_match:
                    reasoning = reasoning_match.group(0).strip()
                    break
            
            # Extract Attribution (more flexible patterns)
            attribution_patterns = [
                r'🔗\s*\*\*(?:Attribution|Building on).*?(?=\n💭|\n🎬|$)',
                r'🔗\s*\*\*Connecting to.*?(?=\n💭|\n🎬|$)',
                r'🔗\s*\*\*Drawing from.*?(?=\n💭|\n🎬|$)'
            ]
            for pattern in attribution_patterns:
                attribution_match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
                if attribution_match:
                    attribution = attribution_match.group(0).strip()
                    break
            
            # Clean main content - remove all metadata
            main_content = content
            metadata_patterns = [
                r'🎬 \*\*Citation:\*\*.*?(?=\n�|\n🔗|\n🎬|$)',
                r'🎬 \*\*Source Inspiration\*\*:.*?(?=\n�|\n🔗|$)',
                r'💭 \*\*Reasoning(?:\sProcess)?:\*\*.*?(?=\n🔗|\n🎬|$)',
                r'🔗 \*\*(?:Attribution|Building on).*?(?=\n💭|\n🎬|$)',
                r'💬.*?(?=\n|$)',
                r'🔗 Shared.*insights.*'
            ]
            
            for pattern in metadata_patterns:
                main_content = re.sub(pattern, '', main_content, flags=re.DOTALL)
            
            # Clean up extra whitespace
            main_content = re.sub(r'\n\s*\n+', '\n\n', main_content).strip()
            
            # Update chat session conversation
            chat_session.conversation.append({
                'role': agent_type,
                'content': main_content,
                'agent_type': agent_type,
                'inspiration': agent_msg.get('inspiration', ''),
                'movie': agent_msg.get('movie', ''),
                'citation': citation,
                'reasoning': reasoning,
                'attribution': attribution,
                'source_inspiration': source_inspiration
            })
            
            # Save to database
            message_data = {
                'role': agent_type,
                'content': main_content,
                'agent_type': agent_type,
                'inspiration': agent_msg.get('inspiration', ''),
                'movie': agent_msg.get('movie', ''),
                'citation': citation,
                'reasoning': reasoning,
                'attribution': attribution,
                'source_inspiration': source_inspiration
            }
            
            ChatSessionManager.save_message(session_id, message_data)
            
            # Return formatted response
            return {
                'type': 'agent',
                'agent_type': agent_type,
                'content': main_content,
                'inspiration': agent_msg.get('inspiration', ''),
                'movie': agent_msg.get('movie', ''),
                'citation': citation,
                'reasoning': reasoning,
                'attribution': attribution,
                'source_inspiration': source_inspiration,
                'timestamp': datetime.now().isoformat()
            }
        
    except Exception as e:
        print(f"Error processing {agent_type} response: {e}")
        import traceback
        traceback.print_exc()
    
    return None

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
