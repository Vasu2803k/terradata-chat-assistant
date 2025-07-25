import time
import os
import joblib
import streamlit as st
import requests
from dotenv import load_dotenv
from pathlib import Path

# --- ChatGPT-like CSS with improved layout ---
CHATGPT_DARK_CSS = """
<style>
/* Main layout */
body, .stApp {
    background-color: #343541 !important;
    color: #ececf1 !important;
}

/* Sidebar styling */
[data-testid="stSidebar"], .stSidebar {
    background: #202123 !important;
    width: 260px !important;
}
.stSidebar .stButton {
    width: 100%;
}
.stSidebar .stButton>button {
    width: 100%;
    text-align: left;
    font-size: 13px !important;
    padding: 6px 12px !important;
    margin: 2px 0 !important;
    background: #444654 !important;
    color: #ececf1 !important;
    border-radius: 6px;
    border: none;
    font-weight: 500;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.stSidebar .stButton>button:hover {
    background: #565869 !important;
}

/* Main content area - wider layout */
.main .block-container {
    max-width: 95% !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* Chat bubbles */
.user-bubble {
    background: #2a2b32;
    color: #ececf1;
    border-radius: 8px;
    padding: 12px 20px;
    margin: 10px 0;
    text-align: right;
    align-self: flex-end;
    max-width: 70%;
    word-wrap: break-word;
    margin-left: auto;
}
.assistant-bubble {
    background: #444654;
    color: #ececf1;
    border-radius: 8px;
    padding: 12px 20px;
    margin: 10px 0;
    text-align: left;
    align-self: flex-start;
    max-width: 70%;
    word-wrap: break-word;
}

/* Fixed title area */
.chat-header {
    position: sticky;
    top: 0;
    z-index: 1000;
    background-color: #343541;
    padding: 20px 0 10px 0;
    margin-bottom: 20px;
}

/* Chat area - full width and height */
.fixed-chat-area {
    height: calc(100vh - 280px);
    overflow-y: auto;
    padding: 20px;
    border: 1px solid #444654;
    border-radius: 8px;
    background-color: #353740;
    width: 100%;
}

/* Scrollbar styling */
.fixed-chat-area::-webkit-scrollbar {
    width: 8px;
}
.fixed-chat-area::-webkit-scrollbar-track {
    background: #2a2b32;
}
.fixed-chat-area::-webkit-scrollbar-thumb {
    background: #565869;
    border-radius: 4px;
}

/* Expander styling */
.stExpander {
    background: #353740 !important;
    color: #ececf1 !important;
    border-radius: 8px !important;
    margin-top: 8px !important;
}
.stExpander summary {
    font-size: 14px !important;
}

/* Input area styling */
.stTextInput>div>div>input {
    background: #40414f !important;
    color: #ececf1 !important;
    border: 1px solid #565869 !important;
    border-radius: 8px !important;
    padding: 12px !important;
    font-size: 16px !important;
}

/* Hide Streamlit default elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Sidebar text styling */
.stSidebar h1 {
    font-size: 18px !important;
    margin-bottom: 10px !important;
}
.stSidebar p, .stSidebar div {
    font-size: 13px !important;
}
</style>
"""

# --- Ensure data directory exists ---
# Set project root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / 'frontend' / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- Load past chats ---
try:
    past_chats: dict = joblib.load(str(DATA_DIR / 'past_chats_list'))
except:
    past_chats = {}

API_URL = "http://localhost:8000/api/v1/chat/"  # Adjust if backend runs elsewhere

# --- Page config ---
st.set_page_config(
    page_title="Terradata Chat Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Apply CSS ---
st.markdown(CHATGPT_DARK_CSS, unsafe_allow_html=True)

# --- Sidebar: LangSmith Dashboard and Past Chats ---
with st.sidebar:
    st.markdown('### LangSmith Dashboard')
    st.markdown('[Open Dashboard →](https://smith.langchain.com/)')
    st.markdown('---')
    st.markdown('### Past Chats')
    if st.button('+ New Chat'):
        new_chat_id = f'{time.time()}'
        st.session_state.chat_id = new_chat_id
        st.session_state.chat_title = f'ChatSession-{new_chat_id}'
        st.session_state.messages = []
        st.session_state.reasoning = []
    
    # Display past chats with compact styling
    for chat_id, chat_title in past_chats.items():
        # Truncate long titles
        display_title = chat_title[:30] + '...' if len(chat_title) > 30 else chat_title
        if st.button(display_title, key=f"chat_{chat_id}"):
            st.session_state.chat_id = chat_id
            st.session_state.chat_title = chat_title
    
    if 'chat_id' not in st.session_state:
        st.session_state.chat_id = f'{time.time()}'
        st.session_state.chat_title = f'ChatSession-{st.session_state.chat_id}'

# --- Main Chat Area ---
# Fixed Title Bar
st.markdown('<div class="chat-header">', unsafe_allow_html=True)
st.title("Terradata Chat Assistant")
st.markdown('</div>', unsafe_allow_html=True)

# --- Initialize session state ---
if 'chat_id' not in st.session_state:
    st.session_state.chat_id = f'{time.time()}'
    st.session_state.chat_title = f'ChatSession-{st.session_state.chat_id}'

# --- Load chat history for current session ---
try:
    st.session_state.messages = joblib.load(str(DATA_DIR / f'{st.session_state.chat_id}-st_messages'))
    st.session_state.reasoning = joblib.load(str(DATA_DIR / f'{st.session_state.chat_id}-reasoning'))
except:
    st.session_state.messages = []
    st.session_state.reasoning = []

# Initialize streaming state
if 'is_streaming' not in st.session_state:
    st.session_state.is_streaming = False
if 'current_response' not in st.session_state:
    st.session_state.current_response = ""
if 'current_reasoning' not in st.session_state:
    st.session_state.current_reasoning = []

# --- Create the chat container ---
st.markdown('<div class="fixed-chat-area" id="chat-area">', unsafe_allow_html=True)

# Render all existing messages
message_container = st.container()
with message_container:
    for idx, message in enumerate(st.session_state.messages):
        if message['role'] == 'user':
            st.markdown(f'<div class="user-bubble">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-bubble">{message["content"]}</div>', unsafe_allow_html=True)
            # Show reasoning steps for assistant messages
            if idx < len(st.session_state.reasoning) and st.session_state.reasoning[idx]:
                with st.expander("Reasoning Steps", expanded=False):
                    for step in st.session_state.reasoning[idx]:
                        if step.get("type") == "reasoning":
                            st.write(f"**Step:** {step.get('step','')}")
                            for k, v in step.items():
                                if k not in ("type", "step") and v is not None:
                                    st.write(f"- **{k.replace('_',' ').capitalize()}**: {v}")
                        elif step.get("type") == "error":
                            st.error(step.get("error"))
    
    # Show current streaming response
    if st.session_state.is_streaming:
        if st.session_state.current_response:
            st.markdown(f'<div class="assistant-bubble">{st.session_state.current_response}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-bubble">Thinking...</div>', unsafe_allow_html=True)
        
        # Show live reasoning if available
        if st.session_state.current_reasoning:
            with st.expander("Reasoning Steps", expanded=False):
                for step in st.session_state.current_reasoning:
                    if step.get("type") == "reasoning":
                        st.write(f"**Step:** {step.get('step','')}")
                        for k, v in step.items():
                            if k not in ("type", "step") and v is not None:
                                st.write(f"- **{k.replace('_',' ').capitalize()}**: {v}")
                    elif step.get("type") == "error":
                        st.error(step.get("error"))

st.markdown('</div>', unsafe_allow_html=True)

# Auto-scroll to bottom script
st.markdown("""
<script>
    var chatArea = document.getElementById('chat-area');
    if(chatArea) {
        chatArea.scrollTop = chatArea.scrollHeight;
    }
</script>
""", unsafe_allow_html=True)

# --- Chat input at the bottom ---
def send_message():
    prompt = st.session_state.get('input', '')
    if not prompt.strip():
        return
        
    # Save this as a chat for later
    if st.session_state.chat_id not in past_chats:
        past_chats[st.session_state.chat_id] = st.session_state.chat_title
        joblib.dump(past_chats, str(DATA_DIR / 'past_chats_list'))
    
    # Add user message immediately
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    
    # Initialize streaming state
    st.session_state.is_streaming = True
    st.session_state.current_response = ""
    st.session_state.current_reasoning = []
    
    # Clear input immediately
    st.session_state['input'] = ''
    
    # Force rerun to show user message immediately
    st.rerun()

def process_streaming_response():
    """Handle the streaming response in the background"""
    if not st.session_state.is_streaming:
        return
    
    # Get the last user message
    last_user_message = None
    for msg in reversed(st.session_state.messages):
        if msg['role'] == 'user':
            last_user_message = msg['content']
            break
    
    if not last_user_message:
        st.session_state.is_streaming = False
        return
    
    reasoning_steps = []
    full_response = ''
    
    try:
        with requests.post(API_URL, json={"user_id": "user", "message": last_user_message}, stream=True, timeout=60) as resp:
            if resp.status_code == 200:
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = line.decode("utf-8")
                        import json
                        chunk = json.loads(chunk)
                    except Exception:
                        continue
                        
                    if chunk.get("type") == "reasoning":
                        reasoning_steps.append(chunk)
                        st.session_state.current_reasoning = reasoning_steps
                        
                    elif chunk.get("type") == "final":
                        full_response = chunk.get("response", "(No response)")
                        st.session_state.current_response = full_response
                        break
            else:
                full_response = f"Error: {resp.text}"
                st.session_state.current_response = full_response
                
    except Exception as e:
        full_response = f"Error: {str(e)}"
        st.session_state.current_response = full_response
    
    # Finalize the response
    st.session_state.messages.append({'role': 'ai', 'content': full_response})
    st.session_state.reasoning.append(reasoning_steps)
    
    # Reset streaming state
    st.session_state.is_streaming = False
    st.session_state.current_response = ""
    st.session_state.current_reasoning = []
    
    # Save to file
    joblib.dump(st.session_state.messages, str(DATA_DIR / f'{st.session_state.chat_id}-st_messages'))
    joblib.dump(st.session_state.reasoning, str(DATA_DIR / f'{st.session_state.chat_id}-reasoning'))
    
    # Final rerun to show completed state
    st.rerun()

# Process streaming if active
if st.session_state.is_streaming and st.session_state.current_response == "":
    process_streaming_response()

# Chat input with improved styling
col1, col2 = st.columns([10, 1])
with col1:
    st.text_input('Type your message...', key='input', on_change=send_message, label_visibility="collapsed")