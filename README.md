# JARVIS AI Assistant

JARVIS is an advanced AI assistant that combines voice and text interfaces, task and reminder management, and intelligent conversation using state-of-the-art LLMs (Google Gemini, Groq Llama3) and vector search for context. It features a modern web UI and can also run in continuous voice mode.

## Features

- **Conversational AI**: Chat with JARVIS using natural language (text or voice).
- **Task Management**: Add, list, and update tasks with priorities and due dates.
- **Reminders**: Set and receive reminders.
- **Contextual Memory**: Vector search for previous conversations and context.
- **Web Interface**: Modern, interactive web UI with voice support.
- **Voice Mode**: Wake-word detection and continuous listening (desktop only).
- **Agentic Reasoning**: Uses LangChain agents for advanced task handling.

## Architecture

JARVIS is built with a modular, agentic architecture that enables flexible, context-aware, and extensible AI assistance:

- **LLM Backbone**: Uses Google Gemini and Groq Llama3 models for natural language understanding and generation.
- **LangChain Agents & Tools**: The assistant leverages [LangChain](https://python.langchain.com/) agents, which can invoke a set of "tools" (functions) for structured operations such as task management, reminders, and context retrieval. This enables the AI to reason and act, not just chat.
- **Vector Database (FAISS)**: All conversations and knowledge snippets are embedded using Sentence Transformers and stored in a FAISS vector database. This allows for fast semantic search and retrieval of relevant context for every user query.
- **Knowledge Base**: The assistant builds a knowledge base from user interactions, storing conversation history, tasks, reminders, and extracted user preferences/topics. This knowledge base is used to personalize responses and provide continuity.
- **Web & Voice Interface**: The backend is a Flask server exposing REST APIs, while the frontend is a modern HTML/JS UI supporting both text and voice input/output.
- **Background Services**: Reminders and scheduled tasks are managed in background threads, ensuring timely alerts and persistent memory.

**High-Level Flow:**

1. **User Input** (text/voice) → 
2. **Frontend** (Web UI) → 
3. **Flask Backend** → 
4. **Agent Decision** (LLM + LangChain tools) → 
5. **Tool Execution** (task/reminder/search/context) → 
6. **LLM Response Generation** (with context from vector DB & knowledge base) → 
7. **Frontend Output** (text/voice) → 
8. **Data Storage** (Excel, Pickle, FAISS)

## Requirements

- Python 3.8+
- Node.js (optional, for frontend development)
- API keys for [Google Gemini](https://aistudio.google.com/apikey) and [Groq](https://console.groq.com/keys)

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/gamkers/Project_Jarvis.git
   cd Project_Jarvis
   ```

2. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set API Keys**

   - Get your Gemini API key here: [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)
   - Get your Groq API key here: [https://console.groq.com/keys](https://console.groq.com/keys)

   Export your API keys as environment variables:

   ```bash
   export GEMINI_API_KEY='your_gemini_api_key'
   export GROQ_API_KEY='your_groq_api_key'
   ```

   Or set them in your shell profile (`.bashrc`, `.zshrc`, etc).

4. **Run the Backend**

   ```bash
   python app.py
   ```

   The Flask server will start at [http://localhost:4000](http://localhost:4000).

5. **Access the Web Interface**

   Open your browser and go to [http://localhost:4000](http://localhost:4000).

## Usage

- **Web UI**: Use the chat box to interact with JARVIS. You can type or use the microphone button for voice input.
- **Voice Mode**: (Desktop only) Run `jarvis_ai.py` directly for continuous wake-word listening.
- **APIs**: The backend exposes REST endpoints for conversation, tasks, reminders, and search.

## Project Structure

```
Jarvis_Project/
├── app.py                # Flask backend server
├── jarvis_ai.py          # Main Jarvis AI logic (agents, tools, vector DB, etc.)
├── requirements.txt      # Python dependencies
├── static/               # Frontend static files (HTML, CSS, JS)
├── jarvis_data/          # Data storage (created at runtime)
└── README.md
```

## Notes

- For best voice experience, use Chrome or Edge browsers.
- Data is stored in `jarvis_data/` as Excel and pickle files.
- Make sure your API keys are valid and have sufficient quota.

## License

MIT License

