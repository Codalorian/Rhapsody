import streamlit as st
import subprocess
from ollama import chat, ResponseError
import os
import json
from datetime import datetime
import re
from PIL import Image

# Import the ModelFetcher and Transformers descriptions
from inference.textgeneration.model_fetcher import ModelFetcher, MODEL_DESCRIPTIONS_TRANSFORMERS 

# Import the TextGenerator for Transformers API
from inference.textgeneration.streamer import TextGenerator

# --- Placeholder for easygui (if you don't have it installed) ---
try:
    import easygui
except ImportError:
    st.warning("`easygui` not found. Install it (`pip install easygui`) for donation message.")
    class MockEasyGUI: # Mock class to prevent errors if easygui is not installed
        def msgbox(self, message):
            st.info(f"Donation Info (easygui not installed):\n{message}")
    easygui = MockEasyGUI()

# File to store conversations
CONVERSATIONS_FILE = "conversations.json"
SETTINGS_FILE = "settings.json" # New file for persistent settings

# Model Descriptions for Ollama (your existing list)
MODEL_DESCRIPTIONS_OLLAMA = {
    "orca-mini": "A small, efficient model optimized for conversational tasks and responses.",
    "smollm": "A lightweight language model designed for minimal memory usage.",
    "smollm2": "An improved version of smollm with better accuracy and efficiency.",
    "tinydolphin": "A compact and fast model, ideal for small-scale conversations and code generation.",
    "dolphin-phi": "A model optimized for simple reasoning and code-based tasks.",
    "mistral-small": "A small variant of Mistral with high efficiency for quick and clean text output.",
    "stablelm2": "An optimized StableLM model for smooth and stable performance on conversational tasks.",
    "stable-code": "A model optimized for and understanding code snippets.",
    "wizard-vicuna-uncensored": "A conversational model focused on uncensored, creative output.",
    "openchat": "OpenChat is ideal for dynamic, multi-turn conversations with quick responses.",
    "aya:8b": "A fast and lightweight model with creative generation and comprehension capabilities.",
    "codeqwen": "CodeQwen is specifically designed for coding tasks and technical problem-solving.",
    "qwen2-math:7b": "Optimized for math and technical conversations, with accurate problem-solving skills.",
    "deepseek-llm:7b": "A general-purpose conversational model for small tasks and clear responses.",
    "neural-chat": "A lightweight chat model fine-tuned for human-like responses and small tasks.",
    "nous-hermes:7b": "A conversational assistant optimized for quick and accurate language outputs."
}

# --- Streamlit Session State Initialization (MOVED TO TOP) ---
def initialize_session_state():
    # Load settings and conversations if not already in session_state
    if 'settings' not in st.session_state:
        st.session_state['settings'] = load_settings()
    if 'conversations' not in st.session_state:
        st.session_state['conversations'] = load_conversations()
    if 'current_conversation' not in st.session_state:
        st.session_state['current_conversation'] = None
    if 'system_prompt' not in st.session_state:
        st.session_state['system_prompt'] = "You are a helpful and friendly AI assistant."


# --- Settings Management ---
def load_settings():
    """Load settings from a JSON file."""
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as file:
            try:
                settings = json.load(file)
                # Ensure 'generation_params' always exists with defaults if not loaded
                if 'generation_params' not in settings:
                    settings['generation_params'] = {
                        "max_new_tokens": 100,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "top_k": 50,
                        "repetition_penalty": 1.0,
                        "do_sample": True,
                        "num_beams": 1
                    }
                return settings
            except json.JSONDecodeError:
                pass # Fall through to return default settings
    return {
        "last_selected_api": "Ollama", # Default API
        "last_selected_model": { # Default model for each API type
            "Ollama": list(MODEL_DESCRIPTIONS_OLLAMA.keys())[0] if MODEL_DESCRIPTIONS_OLLAMA else "orca-mini",
            "Transformers": "gpt2",
            "llama.cpp": "llama2", # Placeholder for llama.cpp, will be a file path later
        },
        # Default generation parameters
        "generation_params": {
            "max_new_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.0,
            "do_sample": True,
            "num_beams": 1
        }
    }

def save_settings():
    """Save current settings to a JSON file."""
    with open(SETTINGS_FILE, "w") as file:
        json.dump(st.session_state['settings'], file, indent=4)

# --- Conversation Management (your existing code) ---
def load_conversations():
    """Load conversations from a JSON file."""
    if os.path.exists(CONVERSATIONS_FILE):
        with open(CONVERSATIONS_FILE, "r") as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                return {}
    return {}


def save_conversations():
    """Save current conversations to a JSON file."""
    with open(CONVERSATIONS_FILE, "w") as file:
        json.dump(st.session_state['conversations'], file, indent=4)

# --- Ollama Model Management (your existing code) ---
def is_model_installed(model_name):
    """Checks if a model is installed locally for Ollama."""
    try:
        installed_models = subprocess.check_output(["ollama", "list"], universal_newlines=True)
        return model_name in installed_models
    except subprocess.CalledProcessError:
        return False


def install_model_with_progress(model_name):
    """Installs an Ollama model with progress tracking."""
    st.info(f"Installing model '{model_name}'. Please wait...")
    progress_bar = st.progress(0)

    process = subprocess.Popen(
        ["ollama", "pull", model_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8"
    )

    for line in process.stdout:
        line = line.strip()
        st.write(line)
        match = re.search(r'(\d+)%', line)
        if match:
            progress = int(match.group(1))
            progress_bar.progress(progress / 100)

    process.wait()
    if process.returncode == 0:
        st.success(f"Model '{model_name}' successfully installed!")
    else:
        st.error(f"Failed to install model '{model_name}'. Please try again.")

# --- Ollama Generation (your existing code) ---
def generate_ollama_response_with_streaming(user_input, model, messages, placeholder, generation_params):
    """Generates streaming AI response considering full conversation history using Ollama."""
    # Check if the model is installed
    if not is_model_installed(model):
        install_model_with_progress(model)

    system_prompt = st.session_state['system_prompt']
    # Ollama chat expects a list of messages directly, not a combined string
    ollama_messages = [{'role': 'system', 'content': system_prompt}] + messages + [{'role': 'user', 'content': user_input}]
    
    try:
        response = chat(
            model=model,
            messages=ollama_messages,
            stream=True,
            # Pass generation parameters to Ollama (if supported by ollama.chat and the model)
            options={
                "temperature": generation_params.get("temperature"),
                "top_p": generation_params.get("top_p"),
                "top_k": generation_params.get("top_k"),
                "num_predict": generation_params.get("max_new_tokens"), # Ollama uses num_predict for max_tokens
                "num_gqa": generation_params.get("num_beams"), # Approximate num_beams for GQA (not exact mapping)
                "repeat_penalty": generation_params.get("repetition_penalty"),
            }
        )
    except ResponseError as e:
        st.error(f"Ollama Model '{model}' not found or service unavailable. Error: {e}")
        return "Error: Ollama model not found or service not running."

    full_response = ""
    for chunk in response:
        if 'message' in chunk and 'content' in chunk['message']:
            text = chunk['message']['content']
            full_response += text
            placeholder.markdown(full_response + "▌") # Live update with cursor
    placeholder.markdown(full_response) # Final update without cursor
    return full_response

# --- Cached Transformers TextGenerator Instance ---
@st.cache_resource
def get_transformer_generator(model_name: str) -> TextGenerator:
    """Returns a cached TextGenerator instance for the given model name."""
    print(f"Initializing/Retrieving TextGenerator for: {model_name}")
    return TextGenerator(model_name)

# --- Transformers Generation ---
def generate_transformers_response_with_streaming(user_input, model_id, messages, placeholder, generation_params):
    """Generates streaming AI response using Transformers."""
    try:
        # Get the cached TextGenerator instance
        generator = get_transformer_generator(model_id)

        # Prepare prompt (simple concatenation for now, can be improved for chat models)
        system_prompt = st.session_state['system_prompt']
        # Transformers models usually take a single string prompt or a specific chat format
        # For general text-generation models, simple concatenation is common.
        # For actual chat models (like Llama-2, Mistral finetunes), you'd need to format messages
        # according to their `chat_template`.
        prompt_text = f"System: {system_prompt}\n" + "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + f"\nUser: {user_input}\nAssistant:"
        
        full_response = ""
        # Call the generate_stream method from TextGenerator
        for token in generator.generate_stream(
            prompt=prompt_text,
            max_tokens=generation_params.get("max_new_tokens"),
            temperature=generation_params.get("temperature"),
            top_p=generation_params.get("top_p"),
            top_k=generation_params.get("top_k"),
            repetition_penalty=generation_params.get("repetition_penalty"),
            do_sample=generation_params.get("do_sample"),
            num_beams=generation_params.get("num_beams")
        ):
            full_response += token
            placeholder.markdown(full_response + "▌") # Live update with cursor
        
        placeholder.markdown(full_response) # Final update without cursor
        return full_response

    except Exception as e:
        st.error(f"Error generating with Transformers model '{model_id}'. Please check console for details. Error: {e}")
        return f"Error: Failed to generate with Transformers ({e})"


def generate_conversation_title(messages, model):
    """Generates a title for the conversation based on the initial exchange."""
    # Temporarily use Ollama for title generation regardless of selected API, 
    # as it's directly callable.
    try:
        # Use the default Ollama model for title generation if it's available
        title_gen_model = st.session_state['settings']['last_selected_model']['Ollama']
        prompt = "Summarize this conversation in 3-8 words for use as a title:\n\n"
        conversation_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        response = chat(model=title_gen_model, messages=[{'role': 'user', 'content': prompt + conversation_text}], stream=False)
        return response['message']['content'].strip()
    except Exception as e:
        st.warning(f"Could not generate title (Ollama not available or model '{title_gen_model}' not installed): {e}. Using timestamp.")
        return datetime.now().strftime("Conversation %Y-%m-%d %H:%M:%S")


def delete_conversation(conv_id):
    """Delete a conversation"""
    if conv_id in st.session_state['conversations']:
        del st.session_state['conversations'][conv_id]
        if st.session_state['current_conversation'] == conv_id:
            st.session_state['current_conversation'] = None
        save_conversations()


def format_timestamp(timestamp_str):
    """Format timestamp for display"""
    try:
        # Try to parse the timestamp
        dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        # Format it nicely
        today = datetime.now().date()
        if dt.date() == today:
            return f"Today {dt.strftime('%I:%M %p')}"
        elif dt.date() == today - datetime.timedelta(days=1):
            return f"Yesterday {dt.strftime('%I:%M %p')}"
        else:
            return dt.strftime("%b %d, %Y")
    except:
        return timestamp_str
        
def start_new_conversation():
    """Creates a new conversation with timestamp, model, and API info."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    default_title = f"New Conversation {timestamp}"
    current_api = st.session_state['settings']['last_selected_api']
    current_model_id = st.session_state['settings']['last_selected_model'][current_api]
    conv_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    st.session_state['conversations'][conv_id] = {
        'title': default_title,
        'api': current_api,
        'model': current_model_id,
        'messages': [],
        'timestamp': timestamp
    }
    st.session_state['current_conversation'] = conv_id
    save_conversations()
    
def main():
    st.set_page_config(layout="wide", page_title="RYFAI Chat Interface") # Removed 'theme' arg
    
    # Add custom CSS for styling
    st.markdown(
        """
        <style>
        img {
            max-width: 100%;
            width: 400px; /* Set a fixed width for images */
            height: auto;
            margin: 0 auto;
            display: block;
        }
        .model-display {
            position: fixed;
            top: 10px;
            left: 10px;
            background-color: var(--secondary-background-color); /* Use Streamlit theme variable */
            color: var(--text-color); /* Use Streamlit theme variable */
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 14px;
            font-weight: bold;
            box-shadow: 0px 0px 5px rgba(0, 0, 0, 0.2);
            z-index: 1000; /* Ensure it's above other elements */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Initialize session state variables (NOW AT THE CORRECT POSITION)
    initialize_session_state()

    # --- TOP OF THE APP: API and Model Selection ---
    api_col, model_select_col = st.columns([0.3, 0.7]) 

    with api_col:
        st.header("🧠 API selection")
        initial_api_index = ['Transformers', 'Ollama', 'llama.cpp'].index(st.session_state['settings']['last_selected_api'])
        select_api = st.selectbox(
            "Select an API to use",
            ['Transformers', 'Ollama', 'llama.cpp'],
            index=initial_api_index,
            key="api_selector" 
        )
        if st.session_state['settings']['last_selected_api'] != select_api:
            st.session_state['settings']['last_selected_api'] = select_api
            save_settings()


    # --- Model Selection Logic based on API ---
    selected_model = None
    model_description = "No description available."
    model_fetcher = ModelFetcher() # Initialize ModelFetcher once

    with model_select_col:
        st.header("🧠 Model Selection")

        if select_api == "Transformers":
            # Fetch models (cached by ModelFetcher)
            organized_models = model_fetcher.fetch_text_generation_models()
            
            last_hf_model = st.session_state['settings']['last_selected_model']['Transformers']

            companies = sorted(organized_models.keys())
            
            initial_company_idx = 0
            if last_hf_model: # Try to pre-select company based on last model
                for i, comp in enumerate(companies):
                    for size_dict in organized_models[comp].values():
                        if last_hf_model in size_dict:
                            initial_company_idx = i
                            break
                    if initial_company_idx == i and last_hf_model in organized_models[comp].get(list(organized_models[comp].keys())[0], []): # Check if found in any size
                        break
            
            selected_company = st.selectbox(
                "Company/Creator",
                companies,
                index=initial_company_idx,
                key="transformer_company_selector"
            )

            sizes = sorted(organized_models.get(selected_company, {}).keys())
            
            initial_size_idx = 0
            if selected_company in organized_models: # Try to pre-select size based on last model
                for i, size_name in enumerate(sizes):
                    if last_hf_model in organized_models[selected_company][size_name]:
                        initial_size_idx = i
                        break
            
            selected_size = None
            if sizes:
                selected_size = st.selectbox(
                    "Model Size", 
                    sizes, 
                    index=initial_size_idx, 
                    key="transformer_size_selector"
                )

            if selected_size:
                models_in_size = organized_models[selected_company][selected_size]
                
                initial_model_idx = 0
                if last_hf_model in models_in_size:
                    initial_model_idx = models_in_size.index(last_hf_model)
                elif models_in_size:
                    initial_model_idx = 0
                else:
                    models_in_size = ["No models found for this size/company"]

                selected_model_hf = st.selectbox(
                    "Model",
                    models_in_size,
                    index=initial_model_idx,
                    format_func=lambda x: x.split("/")[-1] if "/" in x else x,
                    key="transformer_model_id_selector"
                )
                
                if selected_model_hf == "No models found for this size/company":
                    selected_model = None
                    model_description = "Please select a valid model."
                else:
                    selected_model = selected_model_hf
                    model_description = model_fetcher.get_model_description(selected_model, selected_company, selected_size)
                    
            else:
                st.warning("No model sizes available for this company.")
                selected_model = None
                model_description = "Please select a company with available models."

        elif select_api == "Ollama":
            model_options = list(MODEL_DESCRIPTIONS_OLLAMA.keys())
            last_ollama_model = st.session_state['settings']['last_selected_model']['Ollama']
            initial_ollama_index = 0
            if last_ollama_model in model_options:
                initial_ollama_index = model_options.index(last_ollama_model)

            selected_model = st.selectbox(
                "Choose a model", 
                model_options, 
                key="ollama_model_selector",
                index=initial_ollama_index
            )
            model_description = MODEL_DESCRIPTIONS_OLLAMA.get(selected_model, 'No description available')

        elif select_api == "llama.cpp":
            st.markdown("**You must load a custom GGUF/GGML model from the Hugging Face Hub.**")
            st.info("Direct model selection and loading for `llama.cpp` is under development.")
            selected_model = "llama.cpp_custom_model" 
            model_description = "Using a custom model with `llama.cpp`."
        
        st.markdown(f"**Model Description:** {model_description}")
        
        if selected_model and st.session_state['settings']['last_selected_model'][select_api] != selected_model:
            st.session_state['settings']['last_selected_model'][select_api] = selected_model
            save_settings()


    # --- Main Title (now below the selectors) ---
    st.title("💬 Rhapsody AI chat")

    # --- Sidebar (with generation parameters at the top) ---
        # --- Sidebar (with generation parameters at the top) ---
    with st.sidebar:
        # GENERATION PARAMETERS (MOVED TO TOP OF SIDEBAR)
        st.header("⚙️ Generation Settings")
        
        st.session_state['settings']['generation_params']['max_new_tokens'] = st.slider(
            "Max tokens", 10, 500, st.session_state['settings']['generation_params']['max_new_tokens'], key="max_tokens_slider"
        )
        st.session_state['settings']['generation_params']['temperature'] = st.slider(
            "Temperature", 0.1, 2.0, st.session_state['settings']['generation_params']['temperature'], key="temperature_slider"
        )
        
        with st.expander("Advanced Generation Options"):
            st.session_state['settings']['generation_params']['top_p'] = st.slider(
                "Top-p", 0.0, 1.0, st.session_state['settings']['generation_params']['top_p'], key="top_p_slider"
            )
            st.session_state['settings']['generation_params']['top_k'] = st.slider(
                "Top-k", 0, 200, st.session_state['settings']['generation_params']['top_k'], key="top_k_slider"
            )
            st.session_state['settings']['generation_params']['repetition_penalty'] = st.slider(
                "Repetition penalty", 0.5, 2.0, st.session_state['settings']['generation_params']['repetition_penalty'], key="repetition_penalty_slider"
            )
            st.session_state['settings']['generation_params']['num_beams'] = st.slider(
                "Number of Beams", 1, 10, st.session_state['settings']['generation_params']['num_beams'], key="num_beams_slider"
            )
            st.session_state['settings']['generation_params']['do_sample'] = st.checkbox(
                "Do Sampling (if temperature > 0.0)", value=st.session_state['settings']['generation_params']['do_sample'], key="do_sample_checkbox"
            )
        
        save_settings() # Save updated generation parameters
        
        st.divider() # Visual separator
        
        # CONVERSATIONS SECTION - NEW CHATGPT STYLE
        st.header("🗂️ Conversations")
        
        # New conversation button
        if st.button("➕ New Conversation", use_container_width=True, type="primary"):
            start_new_conversation()
        
        # Display conversations as a list
        conversations = st.session_state['conversations']
        
        # Sort conversations by timestamp (newest first)
        sorted_convs = sorted(conversations.items(), 
                            key=lambda x: x[1].get('timestamp', ''), 
                            reverse=True)
        
        # Display each conversation as a tile
        for conv_id, conv_data in sorted_convs:
            # Get conversation info
            title = conv_data.get('title', 'Untitled')
            if len(title) > 30:
                title = title[:27] + "..."
            
            model = conv_data.get('model', 'Unknown')
            if '/' in model:
                model = model.split('/')[-1]  # Show only model name, not full path
            
            api = conv_data.get('api', 'Unknown')
            timestamp = conv_data.get('timestamp', '')
            formatted_time = format_timestamp(timestamp)
            
            # Create a container for each conversation
            with st.container():
                col1, col2 = st.columns([5, 1])
                
                with col1:
                    # Make the conversation clickable
                    if st.button(
                        f"**{title}**\n\n{api}: {model} • {formatted_time}", 
                        key=f"conv_{conv_id}",
                        use_container_width=True,
                        disabled=(st.session_state['current_conversation'] == conv_id)
                    ):
                        st.session_state['current_conversation'] = conv_id
                        # Update the current model/API to match the conversation
                        st.session_state['settings']['last_selected_api'] = api
                        st.session_state['settings']['last_selected_model'][api] = conv_data.get('model', '')
                        save_settings()
                
                with col2:
                    # Delete button
                    if st.button("🗑️", key=f"del_{conv_id}", help="Delete conversation"):
                        delete_conversation(conv_id)
                        st.rerun()  # or st.experimental_rerun() for older versions
        
        st.divider() # Visual separator # Visual separator
        
        # SYSTEM PROMPT SECTION
        st.subheader("📝 System Prompt")
        st.session_state['system_prompt'] = st.text_area(
            "Edit the system prompt to control AI behavior:",
            value=st.session_state['system_prompt'],
            height=150,
            key="system_prompt_editor" 
        )
        
        # DONATE BUTTON
        if st.button("Donate to the Dev!"):
            easygui.msgbox("Thank you so much for donating to RYFAI! It keeps projects like RYFAI alive!. The developer currently accepts Bitcoin, Monero, and Litecoin donations!\n\n"
            "Bitcoin: bc1qjnrvt3d8ms69zusvh244v5h2hya9yhqzsemtc2\n"
            "Litecoin: LRJRFiUWkQ1ZL1ZDDnaZ4D2VwtjxMtCe2E\n"
            "Monero: 494oHEbuekCRA8hcWyo81DLPsvy435neSdxJ33m9c4hf5UtJUARrq6f2vU3APWDosFW147pHDv2WK4fVWnWcemHK4d4Ene4"
            )

    # Display current model in the top left (fixed position CSS takes care of this)
    current_model_display = selected_model if selected_model else "No model selected"
    st.markdown(
        f'<div class="model-display">API: {select_api} | Model: {current_model_display}</div>',
        unsafe_allow_html=True,
    )
    
    # Main Chat Interface
    if st.session_state['current_conversation']:
        # Pass parameters to display_chat_ui()
        display_chat_ui(selected_model, select_api, st.session_state['settings']['generation_params'])
    else:
        st.info("Start a new conversation or select one from the sidebar.")


# Modified display_chat_ui to accept selected_model, selected_api, and generation_params
def display_chat_ui(selected_model, selected_api, generation_params):
    """Displays a ChatGPT-like chat interface for the current conversation."""
    if not st.session_state['current_conversation']:
        st.info("Start a new conversation or select one from the sidebar.")
        return

    # Get the current conversation object
    conv_id = st.session_state['current_conversation']
    conversation = st.session_state['conversations'][conv_id]
    messages = conversation['messages']
    model_for_generation = selected_model
    api_for_generation = selected_api

    # Display conversation title and model at the top
    st.subheader(conversation.get('title', 'Untitled Conversation'))
    st.caption(f"Model: {conversation.get('model', 'Unknown')} | API: {conversation.get('api', 'Unknown')}")

    # Display all previous messages
    for msg in messages:
        with st.chat_message("user" if msg['role'] == 'user' else "assistant"):
            st.markdown(msg['content'])

    # Chat input
    user_input = st.chat_input("Type your message here...")
    if user_input:
        # Append user message
        messages.append({'role': 'user', 'content': user_input})
        save_conversations()
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate AI Response based on selected API
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            if api_for_generation == "Ollama":
                full_response = generate_ollama_response_with_streaming(
                    user_input, model_for_generation, messages, response_placeholder, generation_params
                )
            elif api_for_generation == "Transformers":
                full_response = generate_transformers_response_with_streaming(
                    user_input, model_for_generation, messages, response_placeholder, generation_params
                )
            elif api_for_generation == "llama.cpp":
                st.error("llama.cpp API is selected, but generation backend is not yet implemented.")
                full_response = "Error: llama.cpp backend not implemented."

            # Save assistant message
            messages.append({'role': 'assistant', 'content': full_response})
            # Save model and API used for this conversation
            conversation['model'] = model_for_generation
            conversation['api'] = api_for_generation
            save_conversations()

            # Auto-generate title after the first user input and assistant response
            if len(messages) == 2:
                title_gen_model = st.session_state['settings']['last_selected_model']['Ollama']
                conversation_title = generate_conversation_title(messages, title_gen_model)
                conversation['title'] = conversation_title
                save_conversations()


if __name__ == "__main__":
    main()


