# Project_Jarvis-
For this Educational purpose only

## AIM :

"To design and develop a multi-functional AI Assistant named J.A.R.V.I.S that can interact with humans via voice commands and perform system-level, web, and smart automation tasks with intelligence, personalization, and real-time responsiveness."

## Tech Stack (Tools & Technologies):

| Module                  | Tech                                            |
| ----------------------- | ----------------------------------------------- |
| Voice Recognition       | `speech_recognition`, `pyttsx3`, `gTTS`         |
| NLP (Processing speech) | `OpenAI GPT API`, `transformers`, `ChatterBot`  |
| Web Automation          | `Selenium`, `requests`, `BeautifulSoup`         |
| GUI                     | `Tkinter`, `PyQt5`, `Kivy`                      |
| System Control          | `os`, `subprocess`, `pyautogui`, `psutil`       |
| Memory / Brain          | `SQLite`, `JSON`, `pickle`                      |
| API Integration         | `Weather`, `WolframAlpha`, `Spotify`, `YouTube` |

## Sample Architecture:

JARVIS is built with a modular, agentic architecture that enables flexible, context-aware, and extensible AI assistance:

LLM Backbone: Uses Google Gemini and Groq Llama3 models for natural language understanding and generation.
LangChain Agents & Tools: The assistant leverages LangChain agents, which can invoke a set of "tools" (functions) for structured operations such as task management, reminders, and context retrieval. This enables the AI to reason and act, not just chat.
Vector Database (FAISS): All conversations and knowledge snippets are embedded using Sentence Transformers and stored in a FAISS vector database. This allows for fast semantic search and retrieval of relevant context for every user query.
Knowledge Base: The assistant builds a knowledge base from user interactions, storing conversation history, tasks, reminders, and extracted user preferences/topics. This knowledge base is used to personalize responses and provide continuity.
Web & Voice Interface: The backend is a Flask server exposing REST APIs, while the frontend is a modern HTML/JS UI supporting both text and voice input/output.
Background Services: Reminders and scheduled tasks are managed in background threads, ensuring timely alerts and persistent memory.



## Requirements

Python 3.8+
Node.js (optional, for frontend development)
API keys for Google Gemini and Groq

## Setup

### 1) Clone the repository :

```
https://github.com/thiruking/Project_Jarvis-.git
cd Project_Jarvis
```
### 2) Install Python dependencies :

```
pip3 install -r requirements.txt  #for windows
pip install -r requirements.txt  #for macos
```
### 3) Set API Keys:

Get your Gemini API key here: https://aistudio.google.com/apikey
Get your Groq API key here: https://console.groq.com/keys
Export your API keys as environment variables:
```
export GEMINI_API_KEY='your_gemini_api_key'
export GROQ_API_KEY='your_groq_api_key'
```
Or set them in your shell profile (.bashrc, .zshrc, etc).

### 4) Run the Backend

```
python app.py

```
The Flask server will start at http://localhost:4000.

### 5) Access the Web Interface

Open your browser and go to http://localhost:4000.

### 6) Project Structure

```
Jarvis_Project/
├── app.py                # Flask backend server
├── jarvis_ai.py          # Main Jarvis AI logic (agents, tools, vector DB, etc.)
├── requirements.txt      # Python dependencies
├── static/               # Frontend static files (HTML, CSS, JS)
├── jarvis_data/          # Data storage (created at runtime)
└── README.md
```
### NOTES :

For best voice experience, use Chrome or Edge browsers.
Data is stored in jarvis_data/ as Excel and pickle files.
Make sure your API keys are valid and have sufficient quota.
