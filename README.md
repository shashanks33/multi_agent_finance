# Multi-Agent Finance Dashboard

A Streamlit-based web application showcasing a multi-agent AI finance system built with LangChain and LangGraph. You can fetch user-specific news, run sentiment analysis, pull and cache financial data, and generate personalized investment recommendations.

## 📦 Repository Structure

```
multi_agent_finance/
├── finance-env/            # Python virtual environment
├── src/
│   ├── chains/            # LangChain agent definitions
│   ├── graph/             # LangGraph pipeline wiring
│   ├── utils/             # Helpers (storage, settings)
│   ├── streamlit_app.py   # Streamlit UI entrypoint
│   ├── orchestrator.py    # (optional) CLI orchestrator
│   └── __init__.py        # Package marker
├── .env                   # Environment variables (e.g., OPENAI_API_KEY)
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## 🚀 Quickstart

1. **Clone the repo**

   ```bash
   git clone https://github.com/shashanks33/multi_agent_finance.git
   cd multi_agent_finance
   ```

2. **Create & activate a virtual environment**

   ```bash
   python3 -m venv finance-env
   source finance-env/bin/activate      # macOS/Linux
   .\finance-env\Scripts\activate    # Windows PowerShell
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Copy `.env.example` to `.env` and fill in your API keys:

   ```ini
   OPENAI_API_KEY=sk-...
   ```

5. **Run the Streamlit app**

   ```bash
   streamlit run src/streamlit_app.py
   ```

   Then navigate to `http://localhost:8501` in your browser.

## 🛠 Development

* **Chains**: Define individual agents in `src/chains/`.
* **Graph**: Wire agents into a DAG in `src/graph/pipeline_graph.py`.
* **Utils**: Database connections, caching, settings in `src/utils/`.
* **Orchestrator**: A CLI runner at `src/orchestrator.py`.

## 📄 Contributing

1. Fork the repo and create a feature branch.
2. Commit your changes with clear messages.
3. Open a Pull Request against `main`.