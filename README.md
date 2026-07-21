#  Paper Digest Agent

An AI pipeline that reads an arXiv research paper and produces a structured summary — method, key results, limitations, and a relevance score — plus concrete follow-up experiment ideas from a domain-expert AI reviewer.

Built with **LangChain**, **LangGraph**, and **Pydantic AI**, running entirely on **Google Gemini's free tier**. No paid API keys, no billing, ever.

**🔗 Live demo:** *[add your Streamlit Cloud URL here]*

---

## How it works

```
arXiv paper ID
      │
      ▼
LangChain          →  downloads the PDF, splits it into clean text chunks
      │
      ▼
LangGraph          →  extract → summarize → critique loop
                       (retries once if the summary isn't grounded in the source)
      │
      ▼
Pydantic AI         →  Agent 1 turns the text into a validated, structured
  (2 agents)            PaperSummary (title, method, results, limitations,
                         relevance score)
                       Agent 2 plays a domain-expert reviewer and proposes
                         2–3 concrete follow-up experiments
      │
      ▼
Streamlit UI        →  renders the summary + follow-up ideas
```

## Features

- 📄 **Just paste an arXiv ID or URL** — no manual PDF upload needed
- 🔁 **Self-correcting summaries** — a built-in critique step catches ungrounded claims and retries
- ✅ **Type-safe, structured output** — no fragile JSON parsing, powered by Pydantic AI
- 🤖 **Two-agent design** — a summarizer and a domain-expert reviewer that reasons over the first agent's output
- 💸 **Completely free to run** — Google Gemini's free tier + free Streamlit hosting

## Tech stack

`Python` · `LangChain` · `LangGraph` · `Pydantic AI` · `Pydantic` · `Google Gemini API` · `Streamlit`

## Getting started

### 1. Clone and install
```bash
git clone https://github.com/sthanasriharsha/paper-digest-agent
cd paper-digest-agent
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 2. Get a free Gemini API key
Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey), sign in with any Google account, and create a key. No credit card required.

### 3. Set your key
```bash
export GOOGLE_API_KEY=AIza...        # macOS/Linux
$env:GOOGLE_API_KEY="AIza..."        # Windows PowerShell
```

### 4. Run it

Command line:
```bash
python core.py 2511.22940
```

Web app:
```bash
streamlit run app.py
```

## Project structure

```
paper-digest-agent/
├── core.py              # LangChain + LangGraph + Pydantic AI pipeline
├── app.py                # Streamlit UI
├── requirements.txt
└── README.md
```

## Deployment

This app is deployed for free on [Streamlit Community Cloud](https://share.streamlit.io), auto-redeploying on every push to `main`. The API key is stored in Streamlit's Secrets manager and never committed to this repo.

## License

MIT — free to use, modify, and build on.
