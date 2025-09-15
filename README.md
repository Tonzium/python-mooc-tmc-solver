# MOOC Python Exercise Solver

A **Streamlit web application** that lets you browse programming tasks from the [Helsinki University MOOC courses](https://ohjelmointi-25.mooc.fi/), scrape their content, and use AI agents to **automatically solve** and **analyze** student code.  
The app is designed to speed up learning by providing insights, feedback, and ready-to-run code examples.

---

## ✨ Features
- 📚 **Exercise Browser** – Navigate all ~250+ MOOC programming tasks with a clean UI.  
- 🔎 **Scraper Integration** – Automatically fetch task text, starter code, and hints.  
- 🤖 **AI Solver** – Generate Python solutions with local LLMs (e.g. Ollama models).  
- 🧪 **Code Analyzer** – Compare student submissions against reference solutions.  
- 💡 **Learning Insights** – Highlights what the exercise is trying to teach.  
- 🖥️ **Streamlit UI** – Responsive, card-based layout with multi-page navigation.  

---

## 📂 Project Structure

source/

├── app.py # Main entrypoint (Streamlit app)  
├── ai/  
│ ├── agents.py # Multi-agent system for solving tasks  
│ └── analyzer.py # Code analysis logic  
├── scraping/  
│ └── mooc_scraper.py # Selenium scraper for MOOC exercises  
├── pages/  
│ └── exercise.py # Exercise detail page  
├── utils/  
│ └── cache.py # Cache helpers for offline use  
└── requirements.txt # Python dependencies  


## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://gitlab.com/yourusername/mooc-python-solver.git
cd python-mooc-tmc-solver
```

### 2. Compose Docker
Start services (app + Ollama)
```bash
docker compose up -d --build
```

### 2. Pull the model
Docker container needs model to run it locally
```bash
docker compose exec ollama ollama pull deepseek-r1:latest
```

### 3. Open the app

Streamlit UI at: http://localhost:8501

## Author

Developed by Toni Kiuru (Tonzium).
Inspired by the MOOC.fi Python course
