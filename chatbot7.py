
from matplotlib.style import context
import pandas as pd
import numpy as np
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
import operator
import random
from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

print("🎭 BUILDING COLLABORATION SEQUENCE SYSTEM")
print(" All 3 agents respond in collaboration mode")
print("="*80)

# ================================================================================
# STATE DEFINITION
# ================================================================================

class ChatState(TypedDict):
    messages: Annotated[List[dict], operator.add]
    user_input: str
    conversation: List[dict]
    turn: int
    last_speaker: str
    target_agent: str
    original_query: str
    shared_knowledge: Dict[str, List[str]]
    agent_questions: List[dict]
    collaboration_mode: bool
    inter_agent_dialogue: List[dict]
    collaboration_step: int
    agents_completed: List[str]  # NEW: Track which agents have responded

# ================================================================================
# GEMINI LLM
# ================================================================================

class GeminiLLM:
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        self.model = genai.GenerativeModel(model_name)
    
    def invoke(self, messages: List[BaseMessage], **kwargs) -> AIMessage:
        try:
            prompt_parts = []
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    prompt_parts.append(f"SYSTEM: {msg.content}")
                elif isinstance(msg, HumanMessage):
                    prompt_parts.append(f"USER: {msg.content}")
                elif isinstance(msg, AIMessage):
                    prompt_parts.append(f"ASSISTANT: {msg.content}")
                else:
                    prompt_parts.append(f"USER: {str(msg.content)}")
            
            full_prompt = "\n\n".join(prompt_parts)
            
            response = self.model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=500,
                    temperature=0.7
                )
            )
            
            return AIMessage(content=response.text)
            
        except Exception as e:
            print(f"❌ Gemini API Error: {e}")
            return AIMessage(content="I'm having trouble connecting right now.")

llm = GeminiLLM("gemini-1.5-flash")
print("✅ Gemini LLM ready!")

# Load personas
therapist_personas = pd.read_csv('semantic_therapist_personas.csv')
wise_personas = pd.read_csv('semantic_wise_personas.csv')
intelligent_personas = pd.read_csv('semantic_intelligent_personas.csv')

print(f"✅ Loaded personas:")
# ================================================================================
# DOMAIN CLASSIFIER
# ================================================================================

class AdvancedDomainClassifier:
    def __init__(self, llm):
        self.llm = llm
    
    def analyze_query_domain(self, user_query: str) -> dict:
        analysis_prompt = f"""
Analyze this user query and determine which agent domain(s) it belongs to:

USER QUERY: "{user_query}"

AGENT DOMAINS:
1. THERAPIST: Emotions, anxiety, stress, mental health, confidence, feelings, emotional support, psychological well-being
2. EXPERT: Practical solutions, how-to steps, strategies, technical advice, preparation methods, actionable plans
3. WISE_MENTOR: Life wisdom, meaning, philosophy, big picture perspective, values, life decisions, existential questions

Respond in this EXACT format:
THERAPIST_RELEVANCE: [0-10]
EXPERT_RELEVANCE: [0-10]
WISE_RELEVANCE: [0-10]
PRIMARY_DOMAIN: [therapist/expert/wise_mentor/collaborative]
COLLABORATION_NEEDED: [yes/no]
REASONING: [brief explanation]
"""

        try:
            response = self.llm.invoke([SystemMessage(content=analysis_prompt)])
            return self._parse_domain_analysis(response.content)
        except:
            return self._fallback_analysis(user_query)
    
    def _parse_domain_analysis(self, analysis_text: str) -> dict:
        result = {
            'therapist_relevance': 0,
            'expert_relevance': 0,
            'wise_relevance': 0,
            'primary_domain': 'expert',
            'collaboration_needed': False,
            'reasoning': 'Fallback analysis'
        }
        
        lines = analysis_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('THERAPIST_RELEVANCE:'):
                try:
                    result['therapist_relevance'] = int(line.split(':')[1].strip())
                except:
                    pass
            elif line.startswith('EXPERT_RELEVANCE:'):
                try:
                    result['expert_relevance'] = int(line.split(':')[1].strip())
                except:
                    pass
            elif line.startswith('WISE_RELEVANCE:'):
                try:
                    result['wise_relevance'] = int(line.split(':')[1].strip())
                except:
                    pass
            elif line.startswith('PRIMARY_DOMAIN:'):
                domain = line.split(':')[1].strip().lower()
                if domain in ['therapist', 'intelligent_expert', 'wise_mentor', 'collaborative']:  # ✅ FIXED: Use correct agent names
                   result['primary_domain'] = domain
                elif domain == 'expert':  # ✅ HANDLE: Map 'expert' to 'intelligent_expert'
                   result['primary_domain'] = 'intelligent_expert'
            elif line.startswith('COLLABORATION_NEEDED:'):
                result['collaboration_needed'] = 'yes' in line.lower()
            elif line.startswith('REASONING:'):
                result['reasoning'] = line.split(':', 1)[1].strip()
        
        return result
    

    def _fallback_analysis(self, user_query: str) -> dict:
        query_lower = user_query.lower()
        
        therapist_words = ['anxious', 'nervous', 'scared', 'worried', 'stress', 'freaking out', 'emotional', 'feelings']
        expert_words = ['interview', 'job', 'prepare', 'strategy', 'steps', 'plan', 'what do i do', 'how do i']
        wise_words = ['worth', 'meaning', 'decision', 'should i', 'path', 'life']
        
        therapist_score = sum(2 for word in therapist_words if word in query_lower)
        expert_score = sum(2 for word in expert_words if word in query_lower)  
        wise_score = sum(2 for word in wise_words if word in query_lower)
        
        # Check for collaboration triggers
        collaboration_triggers = ['interview', 'ex', 'breakup', 'job', 'emotional', 'freaking out']
        trigger_count = sum(1 for trigger in collaboration_triggers if trigger in query_lower)
        
        if trigger_count >= 3 or (therapist_score > 0 and expert_score > 0):
            return {
                'therapist_relevance': min(therapist_score, 10),
                'expert_relevance': min(expert_score, 10),
                'wise_relevance': min(wise_score, 10),
                'primary_domain': 'collaborative',
                'collaboration_needed': True,
                'reasoning': 'Multiple domains detected - emotional + practical situation'
            }
        elif therapist_score > expert_score and therapist_score > wise_score:
            return {
                'therapist_relevance': min(therapist_score, 10),
                'expert_relevance': 0,
                'wise_relevance': 0,
                'primary_domain': 'therapist',
                'collaboration_needed': False,
                'reasoning': 'Emotional content detected'
            }
        elif expert_score > wise_score:
            return {
                'therapist_relevance': 0,
                'expert_relevance': min(expert_score, 10),
                'wise_relevance': 0,
                'primary_domain': 'intelligent_expert',  # ✅ FIXED: Use correct agent name
                'collaboration_needed': False,
                'reasoning': 'Practical content detected'
            }
        else:
            return {
                'therapist_relevance': 0,
                'expert_relevance': 0,
                'wise_relevance': min(wise_score, 10),
                'primary_domain': 'wise_mentor',
                'collaboration_needed': False,
                'reasoning': 'Wisdom content detected'
            }



domain_classifier = AdvancedDomainClassifier(llm)

# ================================================================================
# KNOWLEDGE MANAGER
# ================================================================================

class CollaborativeKnowledgeManager:
    def __init__(self):
        self.shared_knowledge = {
            'therapist': [],
            'wise_mentor': [],
            'intelligent_expert': []
        }
        self.agent_insights = {
            'therapist': [],
            'wise_mentor': [],
            'intelligent_expert': []
        }
        self.inter_agent_questions = []
        self.collaboration_history = []
        self.inter_agent_dialogue = []  # Track conversations between agents
    
    def share_knowledge(self, from_agent: str, to_agent: str, knowledge: str, knowledge_type: str):
        self.shared_knowledge[to_agent].append({
            'knowledge': knowledge,
            'source_agent': from_agent,
            'type': knowledge_type,
            'shared_at': len(self.shared_knowledge[to_agent])
        })
    
    def ask_question_to_agent(self, from_agent: str, to_agent: str, question: str):
        self.inter_agent_questions.append({
            'from': from_agent,
            'to': to_agent,
            'question': question,
            'timestamp': len(self.inter_agent_questions)
        })
        return f"❓ {from_agent} asked {to_agent}: {question}"
    
    def add_inter_agent_dialogue(self, from_agent: str, to_agent: str, message: str):
        self.inter_agent_dialogue.append({
            'from': from_agent,
            'to': to_agent,
            'message': message,
            'timestamp': len(self.inter_agent_dialogue),
            'responded': False  # Track if this has been responded to
        })
    
    def get_agent_questions(self, agent: str) -> List[dict]:
        return [q for q in self.inter_agent_questions if q['to'] == agent]
    
    def get_shared_knowledge_context(self, agent: str) -> str:
        if not self.shared_knowledge[agent]:
            return ""
        
        context = "\n🔗 SHARED KNOWLEDGE FROM OTHER AGENTS:\n"
        for item in self.shared_knowledge[agent][-3:]:
            context += f"- {item['source_agent']}: {item['knowledge']}\n"
        return context
    
    def get_inter_agent_context(self, agent: str) -> str:
        # Get questions directed to this agent
        questions = self.get_agent_questions(agent)
        context = ""
        
        if questions:
            context += "\n🤔 QUESTIONS FROM OTHER AGENTS:\n"
            for q in questions[-2:]:  # Show last 2 questions
                context += f"- {q['from']} asks: {q['question']}\n"
        
        # Get recent inter-agent dialogue
        if self.inter_agent_dialogue:
            context += "\n💬 RECENT AGENT DISCUSSION:\n"
            for msg in self.inter_agent_dialogue[-3:]:  # Show last 3 messages
                context += f"- {msg['from']} → {msg['to']}: {msg['message']}\n"
        
        return context
    
    def get_collaboration_summary(self) -> str:
        return f"Knowledge shared: {sum(len(k) for k in self.shared_knowledge.values())}, Questions asked: {len(self.inter_agent_questions)}, Dialogues: {len(self.inter_agent_dialogue)}"
knowledge_manager = CollaborativeKnowledgeManager()
def initiate_inter_agent_communication(current_agent: str, state: ChatState) -> str:
    """Agents automatically communicate with each other based on the context"""
    conversation = state.get('conversation', [])
    user_query = state.get('user_input', '')
    collaboration_mode = state.get('collaboration_mode', False)
    
    # Only initiate communication in collaboration mode or complex scenarios
    if not collaboration_mode and random.random() < 0.7:  # 70% chance to skip in single mode
        return ""
    
    agents = ['therapist', 'intelligent_expert', 'wise_mentor']
    other_agents = [a for a in agents if a != current_agent]
    
    # Choose which agent to communicate with
    target_agent = random.choice(other_agents)
    
    # Create communication prompt based on current context
    communication_prompt = f"""
As {current_agent}, you are currently helping a user with: "{user_query}"

You want to communicate with {target_agent} to get their perspective or share insights.

What would you like to say to {target_agent}? Keep it brief and focused on collaboration.
Respond ONLY with your message to {target_agent}, nothing else.
"""
    
    try:
        response = llm.invoke([SystemMessage(content=communication_prompt)])
        message = response.content.strip()
        
        # Add to inter-agent dialogue
        knowledge_manager.add_inter_agent_dialogue(current_agent, target_agent, message)
        
        print(f"💬 {current_agent} → {target_agent}: {message}")
        return f"\n[To {target_agent}: {message}]"
    
    except Exception as e:
        print(f"❌ Inter-agent communication error: {e}")
        return ""
# ================================================================================
# PERSONA SYSTEM
# ================================================================================

class PersonaKnowledge:
    def __init__(self, persona_df: pd.DataFrame, persona_type: str):
        self.persona_type = persona_type
        self.characters = persona_df
    
    def get_character_inspiration(self) -> dict:
        char = self.characters.sample(1).iloc[0]
        return {
            'name': char['character_name'],
            'dialogue': char['dialogue'],
            'movie': char['movie_title'],
            'genre': char.get('genre', ''),
            'context': char.get('context', ''),
            'full_citation': f"{char['character_name']} in {char['movie_title']}: \"{char['dialogue']}\""
        }
therapist_kb = PersonaKnowledge(therapist_personas, "Therapist")
wise_kb = PersonaKnowledge(wise_personas, "Wise Mentor")
intelligent_kb = PersonaKnowledge(intelligent_personas, "Expert")

# Citation System
class CitationSystem:
    def __init__(self):
        self.citations = {}
    
    def generate_citation(self, character_data: dict, reasoning: str = "") -> str:
        """Generate a formatted citation for agent responses"""
        citation = f"\n\n🎬 **Source Inspiration**: {character_data['full_citation']}"
        if reasoning:
            citation += f"\n💭 **Reasoning**: {reasoning}"
        return citation
    
    def generate_attribution(self, source_agent: str, knowledge: str) -> str:
        """Generate attribution when using knowledge from other agents"""
        return f"\n🔗 **Building on {source_agent}'s insight**: {knowledge}"
# ================================================================================
# DOMAIN-SPECIFIC PROMPTS
# ================================================================================

def create_domain_prompt(agent_type: str, character_inspiration: str, character_dialogue: str, 
                        user_query: str, original_problem: str, conversation: List[dict], 
                        is_collaboration: bool = False) -> str:
    
    # Get shared knowledge and inter-agent context
    shared_knowledge = knowledge_manager.get_shared_knowledge_context(agent_type)
    inter_agent_context = knowledge_manager.get_inter_agent_context(agent_type)
    
    collaboration_note = ""
    if is_collaboration:
        collaboration_note = "\n🌟 COLLABORATION MODE: You are working with other agents. Build upon their insights and share your own!"
    
    if agent_type == 'therapist':
        return f"""You are a compassionate Therapist inspired by the speaking style of {character_inspiration}.

SPEAKING STYLE REFERENCE: "{character_dialogue}"

🎯 YOUR DOMAIN: Emotional support, anxiety management, stress relief, confidence building

USER'S SITUATION: {user_query}
{collaboration_note}

{shared_knowledge}
{inter_agent_context}

⚠️ IMPORTANT RULES:
- Stay focused on EMOTIONAL SUPPORT only
- Help them manage anxiety and stress about the situation
- DO NOT give practical advice (that's Expert's job)
- DO NOT give life philosophy (that's Wise Mentor's job)
- Address the USER directly, not fictional characters
- Use the character's speaking style as inspiration but talk TO THE USER
- Collaborate with other agents when appropriate
- If other agents have asked you questions or shared insights, respond to them briefly at the end of your message
- Keep inter-agent communication concise and focused on helping the user

Your emotional support response (speak directly to the user):"""
    
    elif agent_type == 'intelligent_expert':
        return f"""You are a practical Expert inspired by the speaking style of {character_inspiration}.

SPEAKING STYLE REFERENCE: "{character_dialogue}"

🎯 YOUR DOMAIN: Practical solutions, strategies, step-by-step plans, actionable advice

USER'S SITUATION: {user_query}
{collaboration_note}

{shared_knowledge}
{inter_agent_context}

⚠️ IMPORTANT RULES:
- Focus on PRACTICAL SOLUTIONS only
- Give concrete steps for handling the interview professionally
- DO NOT provide emotional support (that's Therapist's job)
- DO NOT give life philosophy (that's Wise Mentor's job)
- Address the USER directly, not fictional characters
- Use the character's speaking style as inspiration but talk TO THE USER
- Collaborate with other agents when appropriate
- If other agents have asked you questions or shared insights, respond to them briefly at the end of your message
- Keep inter-agent communication concise and focused on helping the user

Your practical solution response (speak directly to the user):"""

    else:  # wise_mentor
        return f"""You are a wise Mentor inspired by the speaking style of {character_inspiration}.

SPEAKING STYLE REFERENCE: "{character_dialogue}"

🎯 YOUR DOMAIN: Life wisdom, perspective, values, big picture thinking

USER'S SITUATION: {user_query}
{collaboration_note}

{shared_knowledge}
{inter_agent_context}

⚠️ IMPORTANT RULES:
- Focus on WISDOM and PERSPECTIVE only
- Help them see the bigger picture and long-term view
- DO NOT give practical steps (that's Expert's job)
- DO NOT provide emotional support techniques (that's Therapist's job)
- Address the USER directly, not fictional characters
- Use the character's speaking style as inspiration but talk TO THE USER
- Collaborate with other agents when appropriate
- If other agents have asked you questions or shared insights, respond to them briefly at the end of your message
- Keep inter-agent communication concise and focused on helping the user

Your wisdom response (speak directly to the user):"""

# ================================================================================
# AGENT FUNCTIONS
# ================================================================================
def create_inter_agent_conversation(current_agent: str, state: ChatState) -> str:
    """Create natural conversation between agents based on current context"""
    conversation = state.get('conversation', [])
    user_query = state.get('user_input', '')
    collaboration_mode = state.get('collaboration_mode', False)
    
    if not collaboration_mode and random.random() < 0.8:  # Less frequent in single mode
        return ""
    
    agents = ['therapist', 'intelligent_expert', 'wise_mentor']
    other_agents = [a for a in agents if a != current_agent]
    target_agent = random.choice(other_agents)
    
    # Create dynamic conversation prompts based on current context
    conversation_prompt = f"""
As the {current_agent}, you are currently helping a user with this situation: "{user_query}"

You want to reach out to the {target_agent} to either:
1. Ask for their expertise in their domain
2. Share an insight that might help them
3. Get their perspective on something

Based on the user's current situation, what would you naturally say to the {target_agent}? 
Make it conversational and specific to this situation.

Your domains:
- therapist: emotions, anxiety, stress, mental health support
- intelligent_expert: practical solutions, strategies, step-by-step plans
- wise_mentor: life wisdom, perspective, big picture thinking

Respond ONLY with your natural message to {target_agent}, keep it brief and collaborative.
"""
    
    try:
        response = llm.invoke([SystemMessage(content=conversation_prompt)])
        message = response.content.strip()
        
        # Add to inter-agent dialogue
        knowledge_manager.add_inter_agent_dialogue(current_agent, target_agent, message)
        
        print(f"💬 {current_agent} → {target_agent}: {message}")
        return f"\n\n💬 **To {target_agent}**: {message}"
    
    except Exception as e:
        print(f"❌ Inter-agent communication error: {e}")
        return ""

def respond_to_agent_conversations(agent: str, state: ChatState) -> str:
    """Respond to natural conversations from other agents"""
    recent_dialogue = knowledge_manager.inter_agent_dialogue[-5:]  # Last 5 messages
    
    # Find messages directed to this agent that haven't been responded to
    messages_to_me = [msg for msg in recent_dialogue 
                     if msg['to'] == agent and not msg.get('responded', False)]
    
    if not messages_to_me:
        return ""
    
    response_text = "\n\n💬 **Agent Responses**:\n"
    
    for msg in messages_to_me[-2:]:  # Respond to last 2 messages
        from_agent = msg['from']
        message = msg['message']
        user_query = state.get('user_input', '')
        
        # Create dynamic response prompt
        response_prompt = f"""
Another agent ({from_agent}) just said to you: "{message}"

The user's current situation is: "{user_query}"

As the {agent}, give a natural, helpful response. Base your response on your domain expertise:
- therapist: Focus on emotional support, anxiety management, confidence building
- intelligent_expert: Focus on practical solutions, strategies, actionable advice  
- wise_mentor: Focus on wisdom, perspective, big picture thinking

Keep your response conversational, brief, and helpful to both the other agent and the user.
Respond naturally as if you're having a professional conversation with a colleague.
"""
        
        try:
            response = llm.invoke([SystemMessage(content=response_prompt)])
            answer = response.content.strip()
            
            # Add to inter-agent dialogue as response
            knowledge_manager.add_inter_agent_dialogue(agent, from_agent, f"Thanks {from_agent}! {answer}")
            
            response_text += f"**To {from_agent}**: {answer}\n\n"
            
            # Mark as responded
            msg['responded'] = True
            
        except Exception as e:
            print(f"❌ Error responding to agent conversation: {e}")
    
    return response_text

def respond_to_agent_questions(agent: str, state: ChatState) -> str:
    """Check if this agent has questions from other agents and respond to them"""
    questions = knowledge_manager.get_agent_questions(agent)
    
    if not questions:
        return ""
    
    response_text = "\n\n🔍 RESPONSES TO OTHER AGENTS:\n"
    
    for question in questions[-2:]:  # Respond to last 2 questions
        response_prompt = f"""
As {agent}, another agent ({question['from']}) asked you: "{question['question']}"

How would you respond to them? Keep it brief and helpful.
Respond ONLY with your answer to {question['from']}, nothing else.
"""
        
        try:
            response = llm.invoke([SystemMessage(content=response_prompt)])
            answer = response.content.strip()
            
            # Add to inter-agent dialogue
            knowledge_manager.add_inter_agent_dialogue(agent, question['from'], f"Re: {question['question']} - {answer}")
            
            response_text += f"- To {question['from']}: {answer}\n"
            
            # Remove the answered question
            knowledge_manager.inter_agent_questions = [q for q in knowledge_manager.inter_agent_questions if q != question]
            
        except Exception as e:
            print(f"❌ Error responding to agent question: {e}")
    
    return response_text
def build_conversation_messages(state: ChatState, prompt_text: str) -> List[BaseMessage]:
    """Build full conversation context for LLM"""
    messages = []
    
    # Add system prompt
    messages.append(SystemMessage(content=prompt_text))
    
    # Add conversation history
    conversation = state.get('conversation', [])
    for msg in conversation:
        if msg['role'] == 'user':
            messages.append(HumanMessage(content=msg['content']))
        elif msg['role'] in ['therapist', 'intelligent_expert', 'wise_mentor']:
            # Add agent responses as AI messages
            agent_content = f"[{msg['role'].upper()}]: {msg['content']}"
            messages.append(AIMessage(content=agent_content))
    
    return messages


# CONVERSATION FLOW - COLLABORATION FIX
# ================================================================================

# Add this new function after the knowledge_manager definition
# Add these functions after the conduct_agent_discussion function

# Replace the generate_agent_insight and generate_agent_discussion_response functions

# Replace the generate_agent_insight and generate_agent_discussion_response functions

def generate_agent_insight(agent: str, user_query: str, state: ChatState) -> str:
    """Generate an agent's initial insight about the user's situation - DOMAIN FOCUSED"""
    
    # Get character inspiration for the agent
    if agent == 'therapist':
        inspiration = therapist_kb.get_character_inspiration()
        domain_focus = "emotional support, anxiety management, stress relief"
        domain_instruction = "Focus on the user's EMOTIONAL state - their anxiety, fear, stress. How are they feeling? What emotional support do they need?"
    elif agent == 'intelligent_expert':
        inspiration = intelligent_kb.get_character_inspiration()
        domain_focus = "practical solutions, strategic planning, actionable steps"
        domain_instruction = "Focus on PRACTICAL solutions - what specific actions can they take? What's the strategic approach? What are the concrete steps?"
    else:  # wise_mentor
        inspiration = wise_kb.get_character_inspiration()
        domain_focus = "life wisdom, big picture perspective, meaning and growth"
        domain_instruction = "Focus on LIFE WISDOM - what's the bigger picture? What can they learn? How does this fit into their life journey?"
    
    insight_prompt = f"""
You are a {agent} inspired by {inspiration['name']} from {inspiration['movie']}.
Character speaking style: "{inspiration['dialogue']}"

CRITICAL: You MUST stay in your domain: {domain_focus}
{domain_instruction}

The user said: "{user_query}"

TASK: Share your professional insight about this situation from YOUR DOMAIN ONLY.
- Use your character's speaking style for flavor
- BUT focus entirely on your domain expertise
- Be specific and actionable within your domain
- Keep it brief but substantial (2-3 sentences)
- Make it relevant to solving their problem

Your {domain_focus} insight:"""
    
    try:
        response = llm.invoke([SystemMessage(content=insight_prompt)])
        return response.content.strip()
    except Exception as e:
        print(f"❌ Error generating insight for {agent}: {e}")
        return f"I see this needs {domain_focus} attention."

def generate_agent_discussion_response(agent: str, discussion_log: list, user_query: str, state: ChatState) -> str:
    """Generate an agent's response to other agents' insights - PRODUCTIVE COLLABORATION"""
    
    # Get recent insights from other agents
    other_insights = [log for log in discussion_log if log['agent'] != agent and log['type'] in ['initial_insight', 'discussion_response']]
    
    if not other_insights:
        return ""
    
    # Get character inspiration and domain
    if agent == 'therapist':
        inspiration = therapist_kb.get_character_inspiration()
        domain_focus = "emotional support and mental wellbeing"
        collaboration_instruction = """SPECIFIC TASK: Build on what others said by adding emotional support strategies.
        - How can you help them feel more confident? 
        - What emotional preparation complements the practical advice?
        - How can they manage anxiety while implementing the suggested actions?"""
    elif agent == 'intelligent_expert':
        inspiration = intelligent_kb.get_character_inspiration()
        domain_focus = "practical execution and strategy"
        collaboration_instruction = """SPECIFIC TASK: Build on what others said by adding concrete practical steps.
        - What specific actions should they take based on the emotional and wisdom insights?
        - How can they implement the advice practically?
        - What's the step-by-step execution plan?"""
    else:  # wise_mentor
        inspiration = wise_kb.get_character_inspiration()
        domain_focus = "life perspective and wisdom"
        collaboration_instruction = """SPECIFIC TASK: Build on what others said by adding deeper perspective and meaning.
        - What's the bigger lesson here that connects the emotional and practical aspects?
        - How does this challenge help them grow?
        - What wisdom ties everything together?"""
    
    # Build context of other agents' insights
    insights_context = "\n".join([f"- {log['agent']}: {log['content']}" for log in other_insights[-2:]])
    
    discussion_prompt = f"""
You are a {agent} inspired by {inspiration['name']}.
Character style: "{inspiration['dialogue']}"

User's situation: "{user_query}"
Your domain: {domain_focus}

Other agents shared these insights:
{insights_context}

{collaboration_instruction}

RULES FOR PRODUCTIVE COLLABORATION:
- Stay in character style but be PROFESSIONAL
- Focus on YOUR domain expertise only
- SPECIFICALLY respond to what others said
- Add VALUE that builds on their insights
- Don't just repeat - ENHANCE their advice
- Make it actionable for solving the user's problem
- Keep it conversational but substantial (2-3 sentences)

Your collaborative response building on their insights:"""
    
    try:
        response = llm.invoke([SystemMessage(content=discussion_prompt)])
        content = response.content.strip()
        return content if len(content) > 20 else ""  # Ensure substantial response
    except Exception as e:
        print(f"❌ Error generating discussion response for {agent}: {e}")
        return ""

def conduct_agent_discussion(state: ChatState) -> dict:
    """Agents have a PRODUCTIVE NATURAL conversation focused on their domains to solve the problem TOGETHER"""
    user_query = state['user_input']
    collaboration_mode = state.get('collaboration_mode', False)
    
    # ✅ ONLY conduct discussion if collaboration mode is detected
    if not collaboration_mode:
        print(f"🎯 **SINGLE DOMAIN DETECTED** - No discussion needed")
        return {
            'discussion_log': [],
            'collaboration_mode': False,
            'agents_ready': True
        }
    
    print(f"\n🗣️ **AGENT COLLABORATION DISCUSSION**")
    print(f"💭 Multiple domains needed - Agents building solution together: '{user_query}'")
    print("-" * 60)
    
    discussion_log = []
    
    # All 3 agents participate in PRODUCTIVE discussion
    agents = ['therapist', 'intelligent_expert', 'wise_mentor']
    
    # Phase 1: Each agent shares their domain expertise
    print("💡 **Initial Domain Insights**")
    for agent in agents:
        insight = generate_agent_insight(agent, user_query, state)
        discussion_log.append({
            'agent': agent,
            'type': 'initial_insight',
            'content': insight
        })
        print(f"💡 {agent} ({get_domain_name(agent)}): {insight}")
    
    print("\n🤝 **Building Solutions Together**")
    
    # Phase 2: Productive collaboration - agents respond to each other's insights
    for round_num in range(2):  # 2 rounds of building on each other
        print(f"\n🔄 **Round {round_num + 1}: Cross-Domain Collaboration**")
        for agent in agents:
            response = generate_agent_discussion_response(agent, discussion_log, user_query, state)
            if response:
                discussion_log.append({
                    'agent': agent,
                    'type': 'collaboration_response',
                    'content': response,
                    'round': round_num + 1
                })
                print(f"🤝 {agent}: {response}")
    
    print(f"\n✅ **Integrated Solution Ready** - {len(discussion_log)} collaborative insights")
    print("-" * 60)
    
    return {
        'discussion_log': discussion_log,
        'collaboration_mode': collaboration_mode,
        'agents_ready': True
    }
def get_domain_name(agent: str) -> str:
    """Get readable domain name for display"""
    domain_names = {
        'therapist': 'Emotional Support',
        'intelligent_expert': 'Practical Solutions', 
        'wise_mentor': 'Life Wisdom'
    }
    return domain_names.get(agent, agent)
def generate_agent_context_learning(agent: str, user_query: str, primary_insight: str, state: ChatState) -> str:
    """Generate how a non-primary agent learns from the situation"""
    
    learning_prompt = f"""
You are {agent}. Another agent is handling this situation: "{user_query}"

Their insight: "{primary_insight}"

As {agent}, what do you learn from observing this situation? 
How does this add to your understanding? Keep it brief and note-taking style (1 sentence).

Learning note:"""
    
    try:
        response = llm.invoke([SystemMessage(content=learning_prompt)])
        return response.content.strip()
    except Exception as e:
        print(f"❌ Error generating context learning for {agent}: {e}")
        return f"Noted: {agent} perspective on this situation."

# Update the start_conversation function to include discussion phase
def start_conversation(state: ChatState) -> ChatState:
    user_input = state['user_input']
    conversation = state.get('conversation', [])
    
    # Add user message
    conversation.append({
        'role': 'user', 
        'content': user_input
    })
    
    # Store original query
    original_query = state.get('original_query', '')
    if not original_query:
        original_query = user_input
    
    # Check for explicit agent targeting
    target_agent = ""
    collaboration_mode = False
    collaboration_step = 1
    agents_completed = []
    
    if user_input.startswith('therapist '):
        target_agent = 'therapist'
        collaboration_mode = False  # ✅ Explicit targeting = no collaboration
        print(f"🎯 EXPLICIT: User chose Therapist - single response")
    elif user_input.startswith('expert '):
        target_agent = 'intelligent_expert'
        collaboration_mode = False  # ✅ Explicit targeting = no collaboration
        print(f"🎯 EXPLICIT: User chose Expert - single response")
    elif user_input.startswith('wise ') or user_input.startswith('mentor '):
        target_agent = 'wise_mentor'
        collaboration_mode = False  # ✅ Explicit targeting = no collaboration
        print(f"🎯 EXPLICIT: User chose Wise Mentor - single response")
    else:
        # Use domain analysis
        print(f"🔍 Analyzing domain for: '{user_input}'")
        domain_analysis = domain_classifier.analyze_query_domain(user_input)
        
        print(f"📊 Domain scores - Therapist: {domain_analysis['therapist_relevance']}, Expert: {domain_analysis['expert_relevance']}, Wise: {domain_analysis['wise_relevance']}")
        print(f"🎯 Analysis: {domain_analysis['reasoning']}")
        
        if domain_analysis['collaboration_needed']:
            target_agent = 'therapist'  # Start with therapist in collaboration
            collaboration_mode = True   # ✅ Multiple domains = collaboration
            print(f"🤝 COLLABORATION MODE: All 3 agents will respond")
        else:
            target_agent = domain_analysis['primary_domain']
            collaboration_mode = False  # ✅ Single domain = no collaboration
            print(f"🎯 SINGLE AGENT: {target_agent} will respond")
    
    # ✅ NEW: Conduct agent discussion ONLY if collaboration_mode is True
    discussion_result = conduct_agent_discussion({
        **state,
        'user_input': user_input,
        'collaboration_mode': collaboration_mode,
        'target_agent': target_agent
    })
    
    return {
        **state,
        'conversation': conversation,
        'turn': 1,
        'last_speaker': 'user',
        'target_agent': target_agent,
        'original_query': original_query,
        'collaboration_mode': collaboration_mode,
        'collaboration_step': collaboration_step,
        'shared_knowledge': {},
        'agent_questions': [],
        'inter_agent_dialogue': [],
        'agents_completed': agents_completed,
        'discussion_log': discussion_result['discussion_log'],  # ✅ NEW: Store discussion
        'agents_ready': discussion_result['agents_ready']       # ✅ NEW: Agents are ready
    }

# Update agent functions to use discussion insights ONLY in collaboration mode
def therapist_agent(state: ChatState) -> ChatState:
    inspiration = therapist_kb.get_character_inspiration()
    is_collaboration = state.get('collaboration_mode', False)
    discussion_log = state.get('discussion_log', [])
    
    # ✅ Build discussion context ONLY if collaboration mode
    discussion_context = ""
    if is_collaboration and discussion_log:
        discussion_context = "\n🗣️ AGENT DISCUSSION INSIGHTS:\n"
        for log in discussion_log:
            if log['agent'] != 'therapist':  # Other agents' insights
                discussion_context += f"- {log['agent']}: {log['content']}\n"
        discussion_context += "\nBased on this discussion, provide your response to the user.\n"
    
    # Get inter-agent context
    inter_agent_context = knowledge_manager.get_inter_agent_context('therapist')
    
    # Build the complete prompt
    prompt_text = create_domain_prompt(
        'therapist',
        f"{inspiration['name']} from {inspiration['movie']}",
        inspiration['dialogue'],
        state['user_input'],
        state.get('original_query', state['user_input']),
        state.get('conversation', []),
        is_collaboration
    )
    
    # ✅ Add discussion context ONLY if collaboration mode
    if is_collaboration:
        prompt_text += discussion_context
    
    # Add reasoning and citation instruction to prompt
    prompt_text += f"""
    
IMPORTANT: After your response, include:
1. 🎬 Citation to your character inspiration with specific dialogue reference
2. 💭 Brief explanation of your reasoning process
3. 🔗 Attribution to other agents if using their insights
"""

    # Add inter-agent context to prompt
    prompt_text += f"\n{inter_agent_context}"
    
    # Build conversation messages with the COMPLETE prompt
    messages = build_conversation_messages(state, prompt_text)
    
    # Single LLM call with full context
    response = llm.invoke(messages)
    
    # Generate citation
    citation_system = CitationSystem()
    citation = citation_system.generate_citation(inspiration, "Character-inspired emotional support approach")
    
    # ✅ Only do inter-agent communication in collaboration mode
    agent_responses = ""
    agent_communication = ""
    
    if is_collaboration:
        agent_responses = respond_to_agent_conversations('therapist', state)
        agent_communication = create_inter_agent_conversation('therapist', state)
    
    # Build full response
    full_response = response.content + agent_responses + agent_communication + citation
    
    new_message = {
        'role': 'therapist',
        'content': full_response,
        'inspiration': inspiration['name'],
        'movie': inspiration['movie'],
        'domain': 'emotional_support',
        'citation': inspiration['full_citation']
    }
    
    print(f"\n😊 Therapist (inspired by {inspiration['name']}) [EMOTIONAL SUPPORT DOMAIN]:")
    print(f"{full_response}")
    
    # Share knowledge for collaboration ONLY in collaboration mode
    if is_collaboration:
        knowledge_manager.share_knowledge('therapist', 'intelligent_expert', 
                                        f"User emotional state: high anxiety about interview", 'emotional_insight')
        knowledge_manager.share_knowledge('therapist', 'wise_mentor', 
                                        f"User needs confidence and perspective", 'emotional_insight')
        print("🔗 Shared 2 emotional insights")
    
    print("-" * 60)
    
    # Update agents_completed list
    agents_completed = state.get('agents_completed', [])
    if 'therapist' not in agents_completed:
        agents_completed.append('therapist')
    
    return {
        **state,
        'messages': [new_message],
        'agents_completed': agents_completed
    }

# Apply similar changes to intelligent_expert_agent and wise_mentor_agent
def intelligent_expert_agent(state: ChatState) -> ChatState:
    inspiration = intelligent_kb.get_character_inspiration()
    is_collaboration = state.get('collaboration_mode', False)
    discussion_log = state.get('discussion_log', [])
    user_query = state['user_input']
    
    # Get context from user query
    context = user_query[:50].split('.')[0] + "..."
    
    # ✅ Build discussion context ONLY if collaboration mode
    discussion_context = ""
    if is_collaboration and discussion_log:
        discussion_context = "\n🗣️ AGENT DISCUSSION INSIGHTS:\n"
        for log in discussion_log:
            if log['agent'] != 'intelligent_expert':  # Other agents' insights
                discussion_context += f"- {log['agent']}: {log['content']}\n"
        discussion_context += "\nBased on this discussion, provide your response to the user.\n"
    
    # Get inter-agent context
    inter_agent_context = knowledge_manager.get_inter_agent_context('intelligent_expert')
    
    # Build the complete prompt
    prompt_text = create_domain_prompt(
        'intelligent_expert',
        f"{inspiration['name']} from {inspiration['movie']}",
        inspiration['dialogue'],
        state['user_input'],
        state.get('original_query', state['user_input']),
        state.get('conversation', []),
        is_collaboration
    )
    
    # ✅ Add discussion context ONLY if collaboration mode
    if is_collaboration:
        prompt_text += discussion_context
    
    # Add reasoning and citation instruction to prompt
    prompt_text += f"""
    
IMPORTANT: After your response, include:
1. 🎬 Citation to your character inspiration with specific dialogue reference
2. 💭 Brief explanation of your reasoning process
3. 🔗 Attribution to other agents if using their insights
"""
    
    # Add inter-agent context to prompt
    prompt_text += f"\n{inter_agent_context}"
    
    # Build conversation messages with the COMPLETE prompt
    messages = build_conversation_messages(state, prompt_text)
    
    # Single LLM call with full context
    response = llm.invoke(messages)
    
    # Generate citation
    citation_system = CitationSystem()
    citation = citation_system.generate_citation(inspiration, "Character-inspired practical solutions approach")

    # ✅ Only do inter-agent communication in collaboration mode
    agent_responses = ""
    agent_communication = ""
    
    if is_collaboration:
        agent_responses = respond_to_agent_conversations('intelligent_expert', state)
        agent_communication = create_inter_agent_conversation('intelligent_expert', state)
    
    # Build full response
    full_response = response.content + agent_responses + agent_communication + citation

    new_message = {
        'role': 'intelligent_expert',
        'content': full_response,
        'inspiration': inspiration['name'],
        'movie': inspiration['movie'],
        'domain': 'practical_solutions',
        'citation': inspiration['full_citation']
    }
    
    print(f"\n🎓 Expert (inspired by {inspiration['name']}) [PRACTICAL SOLUTIONS DOMAIN]:")
    print(f"{full_response}")
    
    # Share knowledge for collaboration ONLY in collaboration mode
    if is_collaboration:
        knowledge_manager.share_knowledge('intelligent_expert', 'therapist', 
                                        f"Practical strategies needed for: {context}", 'strategic_insight')
        knowledge_manager.share_knowledge('intelligent_expert', 'wise_mentor', 
                                        f"Implementation planning for: {context}", 'strategic_insight')
        print("🔗 Shared 2 strategic insights")
    
    print("-" * 60)
    
    # Update agents_completed list
    agents_completed = state.get('agents_completed', [])
    if 'intelligent_expert' not in agents_completed:
        agents_completed.append('intelligent_expert')
    
    return {
        **state,
        'messages': [new_message],
        'agents_completed': agents_completed
    }

def wise_mentor_agent(state: ChatState) -> ChatState:
    inspiration = wise_kb.get_character_inspiration()
    is_collaboration = state.get('collaboration_mode', False)
    discussion_log = state.get('discussion_log', [])
    
    # ✅ Build discussion context ONLY if collaboration mode
    discussion_context = ""
    if is_collaboration and discussion_log:
        discussion_context = "\n🗣️ AGENT DISCUSSION INSIGHTS:\n"
        for log in discussion_log:
            if log['agent'] != 'wise_mentor':  # Other agents' insights
                discussion_context += f"- {log['agent']}: {log['content']}\n"
        discussion_context += "\nBased on this discussion, provide your response to the user.\n"
    
    # Get inter-agent context
    inter_agent_context = knowledge_manager.get_inter_agent_context('wise_mentor')
    
    # Build the complete prompt
    prompt_text = create_domain_prompt(
        'wise_mentor',
        f"{inspiration['name']} from {inspiration['movie']}",
        inspiration['dialogue'],
        state['user_input'],
        state.get('original_query', state['user_input']),
        state.get('conversation', []),
        is_collaboration
    )
    
    # ✅ Add discussion context ONLY if collaboration mode
    if is_collaboration:
        prompt_text += discussion_context
    
    # Add reasoning and citation instruction to prompt
    prompt_text += f"""
    
IMPORTANT: After your response, include:
1. 🎬 Citation to your character inspiration with specific dialogue reference
2. 💭 Brief explanation of your reasoning process
3. 🔗 Attribution to other agents if using their insights
"""
    
    # Add inter-agent context to prompt
    prompt_text += f"\n{inter_agent_context}"
    
    # Build conversation messages with the COMPLETE prompt
    messages = build_conversation_messages(state, prompt_text)
    
    # Single LLM call with full context
    response = llm.invoke(messages)
    
    # Generate citation
    citation_system = CitationSystem()
    citation = citation_system.generate_citation(inspiration, "Character-inspired wisdom and perspective approach")

    # ✅ Only do inter-agent communication in collaboration mode
    agent_responses = ""
    agent_communication = ""
    
    if is_collaboration:
        agent_responses = respond_to_agent_conversations('wise_mentor', state)
        agent_communication = create_inter_agent_conversation('wise_mentor', state)
    
    # Build full response
    full_response = response.content + agent_responses + agent_communication + citation    
    
    new_message = {
        'role': 'wise_mentor',
        'content': full_response,
        'inspiration': inspiration['name'],
        'movie': inspiration['movie'],
        'domain': 'life_wisdom',
        'citation': inspiration['full_citation']
    }
    
    print(f"\n🧠 Wise Mentor (inspired by {inspiration['name']}) [LIFE WISDOM DOMAIN]:")
    print(f"{full_response}")
    
    # Share knowledge for collaboration ONLY in collaboration mode
    if is_collaboration:
        knowledge_manager.share_knowledge('wise_mentor', 'therapist', 
                                        f"Growth perspective: this challenge can strengthen resilience", 'wisdom_insight')
        knowledge_manager.share_knowledge('wise_mentor', 'intelligent_expert', 
                                        f"Big picture: career advancement matters more than past relationships", 'wisdom_insight')
        print("🔗 Shared 2 wisdom insights")

    print("-" * 60)
    
    # Update agents_completed list
    agents_completed = state.get('agents_completed', [])
    if 'wise_mentor' not in agents_completed:
        agents_completed.append('wise_mentor')
    
    return {
        **state,
        'messages': [new_message],
        'agents_completed': agents_completed
    }

def next_speaker(state: ChatState) -> str:
    turn = state.get('turn', 1)
    target_agent = state.get('target_agent', '')
    collaboration_mode = state.get('collaboration_mode', False)
    collaboration_step = state.get('collaboration_step', 1)
    last_speaker = state.get('last_speaker', '')
    agents_completed = state.get('agents_completed', [])
    
    # ✅ Handle collaboration sequence
    if collaboration_mode:
        # Define the collaboration sequence
        collaboration_sequence = ['therapist', 'intelligent_expert', 'wise_mentor']
        
        # Find next agent that hasn't completed yet
        for agent in collaboration_sequence:
            if agent not in agents_completed:
                if agent == 'therapist':
                    print(f"🤝 COLLABORATION STEP 1: Therapist responds")
                elif agent == 'intelligent_expert':
                    print(f"🤝 COLLABORATION STEP 2: Expert responds")
                elif agent == 'wise_mentor':
                    print(f"🤝 COLLABORATION STEP 3: Wise Mentor responds")
                return agent
        
        # All agents have completed - reset collaboration mode
        print(f"🤝 COLLABORATION COMPLETE")
        collaboration_summary = knowledge_manager.get_collaboration_summary()
        print(f"🤝 COLLABORATION: {collaboration_summary}")
        return 'end_conversation'
    
    # Check if we need to process a new user input
    conversation = state.get('conversation', [])
    user_messages = [msg for msg in conversation if msg['role'] == 'user']
    
    if user_messages:
        latest_user_msg = user_messages[-1]['content']
        
        # Check if this is a new user input (not processed yet)
        if last_speaker == 'user' or turn == 1:
            # Check for explicit agent targeting
            if latest_user_msg.startswith('therapist '):
                print(f"🎯 EXPLICIT: User chose Therapist - single response")
                return 'therapist'
            elif latest_user_msg.startswith('expert '):
                print(f"🎯 EXPLICIT: User chose Expert - single response")
                return 'intelligent_expert'
            elif latest_user_msg.startswith('wise ') or latest_user_msg.startswith('mentor '):
                print(f"🎯 EXPLICIT: User chose Wise Mentor - single response")
                return 'wise_mentor'
            else:
                # No explicit targeting - check if collaboration is needed
                domain_analysis = domain_classifier.analyze_query_domain(latest_user_msg)
                print(f"📊 Domain scores - Therapist: {domain_analysis['therapist_relevance']}, Expert: {domain_analysis['expert_relevance']}, Wise: {domain_analysis['wise_relevance']}")
                print(f"🎯 Analysis: {domain_analysis['reasoning']}")
                
                if domain_analysis['collaboration_needed']:
                    # Enable collaboration mode and start with therapist
                    print(f"🤝 COLLABORATION MODE: All 3 agents will respond")
                    return 'therapist'
                else:
                    # Single agent based on domain analysis
                    new_agent = domain_analysis['primary_domain']
                    print(f"🎯 DOMAIN-BASED: {new_agent} will respond (single agent)")
                    return new_agent
    
    # If last speaker was an agent or no new user input, wait for user input
    return 'end_conversation'
def update_conversation(state: ChatState, agent_type: str) -> ChatState:
    conversation = state.get('conversation', [])
    
    if state.get('messages'):
        latest_message = state['messages'][-1]
        conversation.append(latest_message)
    
    # Update collaboration step if in collaboration mode
    collaboration_step = state.get('collaboration_step', 1)
    collaboration_mode = state.get('collaboration_mode', False)
    agents_completed = state.get('agents_completed', [])
    
    if collaboration_mode:
        collaboration_step += 1
    
    # If we just completed a collaboration, reset the mode
    if collaboration_mode and len(agents_completed) >= 3:
        collaboration_mode = False
        collaboration_step = 1
        agents_completed = []
        print("🔄 Collaboration mode reset")
    
    return {
        **state,
        'conversation': conversation,
        'turn': state.get('turn', 1) + 1,
        'last_speaker': agent_type,
        'collaboration_mode': collaboration_mode,
        'collaboration_step': collaboration_step,
        'agents_completed': agents_completed
    }
def therapist_turn(state: ChatState) -> ChatState:
    result = therapist_agent(state)
    return update_conversation(result, 'therapist')

def wise_mentor_turn(state: ChatState) -> ChatState:
    result = wise_mentor_agent(state)
    return update_conversation(result, 'wise_mentor')

def intelligent_expert_turn(state: ChatState) -> ChatState:
    result = intelligent_expert_agent(state)
    return update_conversation(result, 'intelligent_expert')

# ================================================================================
# BUILD WORKFLOW - COLLABORATION FIX
# ================================================================================

workflow = StateGraph(ChatState)

workflow.add_node("start_conversation", start_conversation)
workflow.add_node("therapist", therapist_turn)
workflow.add_node("wise_mentor", wise_mentor_turn)
workflow.add_node("intelligent_expert", intelligent_expert_turn)

workflow.set_entry_point("start_conversation")

workflow.add_conditional_edges(
    "start_conversation",
    next_speaker,
    {
        "therapist": "therapist",
        "wise_mentor": "wise_mentor",
        "intelligent_expert": "intelligent_expert",
        "end_conversation": END
    }
)

workflow.add_conditional_edges(
    "therapist",
    next_speaker,
    {
        "therapist": "therapist",
        "wise_mentor": "wise_mentor", 
        "intelligent_expert": "intelligent_expert",
        "end_conversation": END
    }
)

workflow.add_conditional_edges(
    "wise_mentor",
    next_speaker,
    {
        "therapist": "therapist",
        "wise_mentor": "wise_mentor",
        "intelligent_expert": "intelligent_expert", 
        "end_conversation": END
    }
)

workflow.add_conditional_edges(
    "intelligent_expert",
    next_speaker,
    {
        "therapist": "therapist",
        "wise_mentor": "wise_mentor",
        "intelligent_expert": "intelligent_expert",
        "end_conversation": END
    }
)

app = workflow.compile()
print("✅ Collaboration sequence workflow created!")

# ================================================================================
# SESSION CLASS
# ================================================================================

class CollaborationFixedSession:
    def __init__(self):
        self.conversation = []
        self.original_problem = ""
        global knowledge_manager
        knowledge_manager = CollaborativeKnowledgeManager()
    
    def chat(self, message):
        print(f"\n{'='*80}")
        print(f"You: {message}")
        print(f"{'='*80}")
        
        if not self.original_problem:
            self.original_problem = message
        
        initial_state = {
            'messages': [],
            'user_input': message,
            'conversation': self.conversation.copy(),
            'turn': 1,
            'last_speaker': 'user',
            'target_agent': '',
            'original_query': self.original_problem,
            'shared_knowledge': {},
            'agent_questions': [],
            'collaboration_mode': False,
            'collaboration_step': 1,
            'inter_agent_dialogue': [],
            'agents_completed': []
        }
        
        try:
            result = app.invoke(initial_state)
            
            # Update session
            self.conversation = result.get('conversation', self.conversation)
            
            print(f"\n🎯 Response complete!")
            print("\n💬 Continue conversation:")
            print("   'therapist [question]' - for emotional support")
            print("   'expert [question]' - for practical solutions") 
            print("   'wise [question]' - for life wisdom")
            print("   Or ask any question for automatic domain detection")
            print("-" * 80)
            
        except Exception as e:
            print(f"❌ Error: {e}")


def interactive_chat():
    print("\n🎭 STARTING COLLABORATION CHAT")
    print("Type 'quit' or 'exit' to stop")
    print("="*80)
    
    session = CollaborationFixedSession()
    
    print("\n💭 How collaboration works:")
    print("   🎯 Complex problems trigger all 3 agents")
    print("   🔄 Sequence: Therapist → Expert → Wise Mentor")
    print("   🔗 Agents share knowledge with each other")
    print("   💬 Explicit targeting still works: 'therapist/expert/wise [question]'")
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ['quit', 'exit', '']:
            print("\n👋 Thanks for testing the collaboration system!")
            break
        
        session.chat(user_input)

if __name__ == "__main__":
    interactive_chat()