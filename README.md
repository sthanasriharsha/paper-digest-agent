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

Opens at `http://localhost:8501`. Paste your Gemini key in the sidebar.

## 5. Get a free live link (10 min)

### Push to GitHub (free)
```bash
git init
git add .
git commit -m "Paper Digest Agent"
gh repo create paper-digest-agent --public --source=. --push
```
(No GitHub CLI? Create a repo manually on github.com and `git push` to it.)

### Deploy on Streamlit Community Cloud (free)
1. Go to https://share.streamlit.io, sign in with GitHub
2. Click "New app" → select your repo → branch `main` → file `app.py`
3. Under Advanced settings → Secrets, optionally add:
   ```
   GEMINI_API_KEY = "AIza..."
   ```
   (Optional — if skipped, each visitor pastes their own free key, so you're never on the hook for anyone else's usage.)
4. Click Deploy

You'll get a public URL like `https://paper-digest-agent-yourname.streamlit.app` in about 2 minutes. Zero cost, no billing ever attached.

## 6. Rate limits to know about

Gemini 2.0 Flash free tier (check current numbers at https://ai.google.dev/gemini-api/docs/rate-limits):
- 15 requests/minute, 1,500 requests/day, 1M tokens/minute
- Comfortably enough for personal use or a portfolio demo link

If you hit a rate limit, requests get rejected temporarily — the free tier never auto-charges you; it simply throttles or errors until the window resets.

## 7. Fully offline alternative (zero API calls at all)

If you'd rather not depend on any external API, swap Gemini for Ollama running locally:

```bash
# Install Ollama: https://ollama.com/download
ollama pull llama3.1:8b
```

In `core.py`, replace the Gemini model setup with:
```python
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIModel(
    "llama3.1:8b",
    provider=OpenAIProvider(base_url="http://localhost:11434/v1"),
)
```
For CrewAI: `LLM(model="ollama/llama3.1:8b", base_url="http://localhost:11434")`.

Needs a machine with enough RAM (8B model needs ~8GB+). Your lab machine (RTX 4070 SUPER, 12GB VRAM) handles this easily — but note it won't be a public link, since it depends on your local machine running.

## 8. What each framework does (for your resume)

| Framework | Role |
|---|---|
| LangChain | `PyPDFLoader` + `RecursiveCharacterTextSplitter` for ingestion |
| LangGraph | 3-node stateful graph (`extract → summarize → critique`) with a conditional retry edge |
| Pydantic AI | Type-safe, validated LLM output (`PaperSummary` schema) |
| CrewAI | Domain-expert agent proposing follow-up experiments relevant to your ISL research |
