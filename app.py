from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import sqlite3
import uuid
from datetime import datetime
import json
import re

# Import the agent system
from chatbot6 import (
    CollaborationFixedSession, 
    domain_classifier,
    therapist_kb,
    wise_kb,
    intelligent_kb,
    knowledge_manager,
    llm,
    app as chatbot6_app  # ✅ Import the workflow app
)
from langchain_core.messages import SystemMessage

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# Database setup for chat sessions
def init_db():
    conn = sqlite3.connect('chat_sessions.db')
    cursor = conn.cursor()
    
    # Create sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create messages table with new columns
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

# Store active sessions for memory
active_sessions = {}

class ChatSessionManager:
    @staticmethod
    def create_session():
        session_id = str(uuid.uuid4())
        conn = sqlite3.connect('chat_sessions.db')
        cursor = conn.cursor()
        
        cursor.execute(
            'INSERT INTO sessions (id, title) VALUES (?, ?)',
            (session_id, 'New Chat')
        )
        
        conn.commit()
        conn.close()
        
        # Create in-memory session
        active_sessions[session_id] = CollaborationFixedSession()
        
        return session_id
    
    @staticmethod
    def save_message(session_id, role, content, agent_type=None, inspiration=None, movie=None, 
                    citation=None, reasoning=None, attribution=None, source_inspiration=None):
        conn = sqlite3.connect('chat_sessions.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO messages (session_id, role, content, agent_type, inspiration, movie, 
                                citation, reasoning, attribution, source_inspiration)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, role, content, agent_type, inspiration, movie, 
              citation, reasoning, attribution, source_inspiration))
        
        # Update session title if it's the first user message
        if role == 'user':
            cursor.execute(
                'SELECT COUNT(*) FROM messages WHERE session_id = ? AND role = "user"',
                (session_id,)
            )
            count = cursor.fetchone()[0]
            if count == 1:  # First user message
                title = content[:50] + "..." if len(content) > 50 else content
                cursor.execute(
                    'UPDATE sessions SET title = ? WHERE id = ?',
                    (title, session_id)
                )
        
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
            SELECT role, content, agent_type, inspiration, movie, citation, reasoning, attribution, source_inspiration, timestamp
            FROM messages WHERE session_id = ? ORDER BY timestamp
        ''', (session_id,))
        
        messages = cursor.fetchall()
        conn.close()
        
        if session_id in active_sessions:
            session = active_sessions[session_id]
            for msg in messages:
                role, content, agent_type, inspiration, movie, citation, reasoning, attribution, source_inspiration, timestamp = msg
                session.conversation.append({
                    'role': role,
                    'content': content,
                    'agent_type': agent_type,
                    'inspiration': inspiration,
                    'movie': movie,
                    'citation': citation,
                    'reasoning': reasoning,
                    'attribution': attribution,
                    'source_inspiration': source_inspiration,
                    'timestamp': timestamp
                })
        
        return messages
    
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
        SELECT role, content, agent_type, inspiration, movie, citation, reasoning, attribution, source_inspiration, timestamp
        FROM messages WHERE session_id = ? ORDER BY timestamp
    ''', (session_id,))
    
    messages = cursor.fetchall()
    conn.close()
    
    formatted_messages = []
    for msg in messages:
        role, content, agent_type, inspiration, movie, citation, reasoning, attribution, source_inspiration, timestamp = msg
        formatted_messages.append({
            'role': role,
            'content': content,
            'agent_type': agent_type or '',
            'inspiration': inspiration or '',
            'movie': movie or '',
            'citation': citation or '',
            'reasoning': reasoning or '',
            'attribution': attribution or '',
            'source_inspiration': source_inspiration or '',
            'timestamp': timestamp
        })
    
    return jsonify(formatted_messages)

def generate_inter_agent_conversation(query, agents_data):
    """Generate a natural conversation between agents after their responses"""
    
    # Build conversation context
    agent_responses = []
    for agent_name, data in agents_data.items():
        agent_responses.append(f"{agent_name}: {data['main_content']}")
    
    conversation_prompt = f"""
You are facilitating a natural discussion between AI agents who have each provided their perspective on the user's query: "{query}"

The agents and their responses are:
{chr(10).join(agent_responses)}

Generate a brief, natural conversation (3-4 exchanges) where the agents discuss their different approaches, acknowledge each other's insights, and find common ground or respectfully note differences. Make it feel like a genuine discussion between colleagues.

Format as a JSON array with objects containing 'agent' and 'content' fields. Use these agent names exactly: {list(agents_data.keys())}

Keep each response conversational and under 100 words. Focus on:
- Acknowledging other agents' valuable points
- Highlighting complementary approaches
- Building on shared themes
- Respectful discussion of different perspectives

Return only the JSON array, no other text.
"""
    
    try:
        response = llm.invoke([SystemMessage(content=conversation_prompt)])
        conversation_text = response.content.strip()
        
        # Clean up any markdown formatting
        if conversation_text.startswith('```json'):
            conversation_text = conversation_text[7:]
        if conversation_text.endswith('```'):
            conversation_text = conversation_text[:-3]
        
        conversation = json.loads(conversation_text)
        
        # Validate the format
        if isinstance(conversation, list) and all(
            isinstance(item, dict) and 'agent' in item and 'content' in item 
            for item in conversation
        ):
            return conversation
        
    except Exception as e:
        print(f"Error generating inter-agent conversation: {e}")
    
    # Fallback: generate a simple conversation
    agents = list(agents_data.keys())
    if len(agents) >= 2:
        return [
            {
                "agent": agents[0],
                "content": f"I appreciate {agents[1]}'s perspective on this. We both emphasize the importance of understanding, though from different angles."
            },
            {
                "agent": agents[1], 
                "content": f"Exactly, {agents[0]}. Your approach complements mine well. Together we can provide more comprehensive support."
            }
        ]
    
    return []

def process_collaborative_response(chat_session, message, session_id):
    """Process collaborative response using chatbot6 collaboration system"""
    
    # Create the state for collaborative mode
    state = {
        'user_input': message,
        'conversation': chat_session.conversation,
        'collaboration_mode': True,  # ✅ Enable collaboration
        'messages': [],
        'turn': 1,
        'last_speaker': 'user',
        'target_agent': '',
        'original_query': message,
        'shared_knowledge': {},
        'agent_questions': [],
        'inter_agent_dialogue': [],
        'agents_completed': []
    }
    
    try:
        from chatbot6 import therapist_agent, intelligent_expert_agent, wise_mentor_agent
        
        agent_responses = []
        agents_data = {}
        
        # Process each agent in sequence for collaboration
        agent_functions = [
            ('therapist', therapist_agent),
            ('intelligent_expert', intelligent_expert_agent), 
            ('wise_mentor', wise_mentor_agent)
        ]
        
        for agent_name, agent_func in agent_functions:
            # Update state for current agent
            state['target_agent'] = agent_name
            state['last_speaker'] = 'user'
            
            result = agent_func(state)
            
            if result and 'messages' in result and result['messages']:
                agent_msg = result['messages'][-1]
                
                # Extract metadata like the single agent processing
                processed_result = extract_agent_metadata(agent_msg, agent_name)
                if processed_result:
                    # Update chat session conversation
                    chat_session.conversation.append({
                        'role': agent_name,
                        'content': processed_result['content'],
                        'agent_type': agent_name,
                        'inspiration': processed_result.get('inspiration', ''),
                        'movie': processed_result.get('movie', ''),
                        'citation': processed_result.get('citation', ''),
                        'reasoning': processed_result.get('reasoning', ''),
                        'attribution': processed_result.get('attribution', ''),
                        'source_inspiration': processed_result.get('source_inspiration', '')
                    })
                    
                    # Save to database
                    ChatSessionManager.save_message(
                        session_id, agent_name, processed_result['content'], agent_name,
                        processed_result.get('inspiration', ''), processed_result.get('movie', ''),
                        processed_result.get('citation', ''), processed_result.get('reasoning', ''), 
                        processed_result.get('attribution', ''), processed_result.get('source_inspiration', '')
                    )
                    
                    # Add to response list
                    agent_responses.append({
                        'role': 'assistant',      # ✅ Add role field
                        'content': processed_result['content'],
                        'agent_type': agent_name,  # ✅ Use agent_type instead of agent
                        'agent': agent_name,       # Keep for backward compatibility  
                        'inspiration': processed_result.get('inspiration', ''),
                        'movie': processed_result.get('movie', ''),
                        'citation': processed_result.get('citation', ''),
                        'reasoning': processed_result.get('reasoning', ''),
                        'attribution': processed_result.get('attribution', ''),
                        'source_inspiration': processed_result.get('source_inspiration', '')
                    })
                    
                    agents_data[agent_name] = processed_result
                    
                    # Mark agent as completed
                    state['agents_completed'].append(agent_name)
                    state['last_speaker'] = agent_name
                    
                    # Update conversation for next agent
                    state['conversation'] = chat_session.conversation
        
        # Generate inter-agent conversation
        conversation = []
        if len(agents_data) >= 2:
            conversation = generate_inter_agent_conversation(message, agents_data)
        
        return {
            'agent_responses': agent_responses,
            'conversation': conversation
        }
        
    except Exception as e:
        print(f"Error in collaborative processing: {e}")
        return None

def extract_agent_metadata(agent_msg, agent_name):
    """Extract metadata from agent message content"""
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
        r'🎬\s*\*\*Citation:?\*\*.*?(?=\n💭|\n🔗|\n🎬|$)',
        r'🎬\s*\*\*Source Inspiration\*\*:.*?(?=\n💭|\n🔗|$)',
        r'💭\s*\*\*(?:Reasoning(?:\s+Process)?|My approach|Thought process|Wisdom approach):?\*\*.*?(?=\n🔗|\n🎬|$)',
        r'🔗\s*\*\*(?:Attribution|Building on|Connecting to|Drawing from).*?(?=\n💭|\n🎬|$)',
        r'💬.*?(?=\n|$)',
        r'🔗\s*Shared.*insights.*'
    ]
    
    for pattern in metadata_patterns:
        main_content = re.sub(pattern, '', main_content, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up extra whitespace
    main_content = re.sub(r'\n\s*\n+', '\n\n', main_content).strip()
    
    return {
        'content': main_content,
        'agent': agent_name,
        'inspiration': agent_msg.get('inspiration', ''),
        'movie': agent_msg.get('movie', ''),
        'citation': citation,
        'reasoning': reasoning,
        'attribution': attribution,
        'source_inspiration': source_inspiration,
        'main_content': main_content
    }

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
                r'🎬\s*\*\*Citation:?\*\*.*?(?=\n💭|\n🔗|\n🎬|$)',
                r'🎬\s*\*\*Source Inspiration\*\*:.*?(?=\n💭|\n🔗|$)',
                r'💭\s*\*\*(?:Reasoning(?:\s+Process)?|My approach|Thought process|Wisdom approach):?\*\*.*?(?=\n🔗|\n🎬|$)',
                r'🔗\s*\*\*(?:Attribution|Building on|Connecting to|Drawing from).*?(?=\n💭|\n🎬|$)',
                r'💬.*?(?=\n|$)',
                r'🔗\s*Shared.*insights.*'
            ]
            
            for pattern in metadata_patterns:
                main_content = re.sub(pattern, '', main_content, flags=re.DOTALL | re.IGNORECASE)
            
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
            ChatSessionManager.save_message(
                session_id, agent_type, main_content, agent_type,
                agent_msg.get('inspiration', ''), agent_msg.get('movie', ''),
                citation, reasoning, attribution, source_inspiration
            )
            
            return {
                'content': main_content,
                'agent': agent_type,
                'inspiration': agent_msg.get('inspiration', ''),
                'movie': agent_msg.get('movie', ''),
                'citation': citation,
                'reasoning': reasoning,
                'attribution': attribution,
                'source_inspiration': source_inspiration,
                'main_content': main_content
            }
    
    except Exception as e:
        print(f"Error processing {agent_type} response: {e}")
        return None

@socketio.on('send_message')
def handle_message(data):
    message = data['message']
    session_id = data.get('session_id')
    mode = data.get('mode', 'single')
    agent_type = data.get('agent_type', 'therapist')
    
    # Get or create session
    if session_id and session_id in active_sessions:
        chat_session = active_sessions[session_id]
        # Load existing conversation
        ChatSessionManager.load_session_history(session_id)
    else:
        session_id = ChatSessionManager.create_session()
        chat_session = active_sessions[session_id]
    
    # Save user message
    chat_session.conversation.append({'role': 'user', 'content': message})
    ChatSessionManager.save_message(session_id, 'user', message)
    
    emit('session_created', {'session_id': session_id})
    
    if mode == 'collaborative':
        print(f"🔥 COLLABORATIVE MODE TRIGGERED")
        # Use the original chatbot6 collaboration workflow
        try:
            # Set up for collaboration mode  
            if not hasattr(chat_session, 'original_problem'):
                chat_session.original_problem = message
            
            # Add user message to session
            chat_session.conversation.append({'role': 'user', 'content': message})
            
            # Use the chatbot6 collaboration system
            initial_state = {
                'user_input': message,
                'conversation': chat_session.conversation.copy(),
                'turn': 1,
                'last_speaker': 'user',
                'target_agent': '',
                'original_query': message,
                'shared_knowledge': {},
                'agent_questions': [],
                'collaboration_mode': True,  # ✅ Enable collaboration
                'collaboration_step': 1,
                'inter_agent_dialogue': [],
                'agents_completed': []
            }
            
            # Import the workflow from chatbot6
            result = chatbot6_app.invoke(initial_state)
            
            # Extract all agent responses from the result
            if result and 'conversation' in result:
                chat_session.conversation = result['conversation']
                
                # Find new agent messages and emit them
                agent_responses = []
                for msg in result['conversation']:
                    if msg['role'] in ['therapist', 'intelligent_expert', 'wise_mentor']:
                        # Extract metadata and emit response
                        processed = extract_agent_metadata(msg, msg['role'])
                        if processed:
                            # Save to database
                            ChatSessionManager.save_message(
                                session_id, msg['role'], processed['content'], msg['role'],
                                processed.get('inspiration', ''), processed.get('movie', ''),
                                processed.get('citation', ''), processed.get('reasoning', ''), 
                                processed.get('attribution', ''), processed.get('source_inspiration', '')
                            )
                            
                            # Emit response
                            emit('agent_response', {
                                'role': 'assistant',
                                'content': processed['content'],
                                'agent_type': msg['role'],
                                'agent': msg['role'],
                                'inspiration': processed.get('inspiration', ''),
                                'movie': processed.get('movie', ''),
                                'citation': processed.get('citation', ''),
                                'reasoning': processed.get('reasoning', ''),
                                'attribution': processed.get('attribution', ''),
                                'source_inspiration': processed.get('source_inspiration', '')
                            })
                            
                            agent_responses.append(processed)
                
                print(f"🔥 COLLABORATIVE RESULT: {len(agent_responses)} agents responded")
                
                # Generate inter-agent conversation if we have multiple responses
                if len(agent_responses) >= 2:
                    agents_data = {resp['agent']: resp for resp in agent_responses}
                    conversation = generate_inter_agent_conversation(message, agents_data)
                    if conversation:
                        print(f"🔥 EMITTING CONVERSATION: {len(conversation)} exchanges")
                        emit('inter_agent_conversation', {'conversation': conversation})
            else:
                print(f"🔥 COLLABORATIVE FAILED - NO RESULT")
                
        except Exception as e:
            print(f"🔥 COLLABORATION ERROR: {e}")
            # Fallback to single agent
            result = process_agent_response(chat_session, message, 'therapist', session_id)
            if result:
                emit('agent_response', {
                    'role': 'assistant',
                    'content': result['content'],
                    'agent_type': 'therapist',
                    'agent': 'therapist',
                    'inspiration': result['inspiration'],
                    'movie': result['movie'],
                    'citation': result['citation'],
                    'reasoning': result['reasoning'],
                    'attribution': result['attribution'],
                    'source_inspiration': result['source_inspiration']
                })
    
    else:
        # Single agent mode
        print(f"🔥 SINGLE AGENT MODE: {agent_type}")
        result = process_agent_response(chat_session, message, agent_type, session_id)
        if result:
            emit('agent_response', {
                'role': 'assistant',          # ✅ Add role field
                'content': result['content'],
                'agent_type': agent_type,     # ✅ Use agent_type for proper naming
                'agent': agent_type,
                'inspiration': result['inspiration'],
                'movie': result['movie'],
                'citation': result['citation'],
                'reasoning': result['reasoning'],
                'attribution': result['attribution'],
                'source_inspiration': result['source_inspiration']
            })
        else:
            print(f"🔥 SINGLE AGENT FAILED - NO RESULT")

@socketio.on('load_session')
def handle_load_session(data):
    session_id = data['session_id']
    
    if session_id not in active_sessions:
        active_sessions[session_id] = CollaborationFixedSession()
    
    messages = ChatSessionManager.load_session_history(session_id)
    
    formatted_messages = []
    for msg in messages:
        role, content, agent_type, inspiration, movie, citation, reasoning, attribution, source_inspiration, timestamp = msg
        formatted_messages.append({
            'role': role,
            'content': content,
            'agent': agent_type or '',
            'inspiration': inspiration or '',
            'movie': movie or '',
            'citation': citation or '',
            'reasoning': reasoning or '',
            'attribution': attribution or '',
            'source_inspiration': source_inspiration or '',
            'timestamp': timestamp
        })
    
    emit('session_loaded', {'messages': formatted_messages, 'session_id': session_id})

if __name__ == '__main__':
    init_db()
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
