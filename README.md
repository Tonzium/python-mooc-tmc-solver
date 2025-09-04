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
cd mooc-python-solver
```

### 2. Install dependencies
Use a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate   # on Linux / macOS
.venv\Scripts\activate      # on Windows

pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run source/app.py
```

The app will open at http://localhost:8501

## Author

Developed by Toni Kiuru (Tonzium).
Inspired by the MOOC.fi Python course
