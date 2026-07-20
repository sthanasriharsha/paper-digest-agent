# Paper Digest Agent — 100% Free Version (Google Gemini)

A research-paper summarizer chaining **LangChain**, **LangGraph**, **Pydantic AI**, and **CrewAI** — running entirely on Google Gemini's free tier, 

```
arXiv PDF --> [LangChain: load + chunk] --> [LangGraph: extract -> summarize -> critique loop]
          --> [Pydantic AI: structured PaperSummary] --> [CrewAI: domain-expert follow-up ideas]
```

## Why Gemini over Groq/OpenAI

| Piece | Cost | Why |
|---|---|---|
| LangChain, LangGraph, Pydantic AI, CrewAI | Free | Open-source Python libraries |
| **LLM (Gemini 2.0 Flash)** | Free | 1,500 requests/day free tier, generous context window, sign in with any Google account |
| Hosting (Streamlit Community Cloud) | Free | Free forever for public apps |
| arXiv PDFs | Free | Public |

No separate account creation needed if you already use Gmail/Google — just visit AI Studio and generate a key.

## 1. Get your free Gemini key (2 min)

1. Go to https://aistudio.google.com/apikey
2. Sign in with your Google account
3. Click "Create API Key" → select or create a Google Cloud project (no billing needs to be enabled for the free tier)
4. Copy the key (starts with `AIza...`)

## 2. Local setup (5 min)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
export GEMINI_API_KEY=AIza...   # Windows: set GEMINI_API_KEY=AIza...
```

## 3. Test from the command line

```bash
python core.py 2511.22940
```

Downloads that One-to-All Animation paper and prints a structured JSON summary. Confirms everything works before touching the UI.

## 4. Run the web app locally

```bash
streamlit run app.py
```
