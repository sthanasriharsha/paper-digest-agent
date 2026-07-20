# 📄 Paper Digest Agent

An AI pipeline that takes an arXiv research paper and produces a structured summary — method, key results, limitations, relevance score — plus concrete, actionable follow-up experiment ideas from a domain-expert AI reviewer.

Built to explore GenAI/Agentic-AI patterns (LangChain, LangGraph, multi-agent design, structured outputs) while producing a real, usable tool for ongoing ISL (Indian Sign Language) avatar research.

**🔗 Live app:** *(add your Streamlit Cloud URL here)*
**📦 Repo:** [github.com/sthanasriharsha/paper-digest-agent](https://github.com/sthanasriharsha/paper-digest-agent)

---

## ✨ What It Does

Give it an arXiv paper ID (e.g. `2511.22940`) and your current research context (e.g. *"working on ISL avatar generation using pose-guided diffusion"*), and it will:

1. **Download** the paper PDF directly from arXiv
2. **Extract and chunk** the text
3. **Summarize** it into a validated, structured schema — title, method, key results, limitations, and a 1–10 relevance score with reasoning
4. **Self-check** — verify the summary's claimed "key results" actually appear (in some form) in the source text; retry once if not
5. **Review** — pass the structured summary to a second AI agent that role-plays a sign-language-avatar research expert, proposing 2–3 concrete follow-up experiments specific to your actual toolchain (ComfyUI, WanVideo, pose-guided diffusion)

Output is rendered in a Streamlit web app, with raw JSON also available.

---

## 🏗️ Architecture

```
arXiv paper ID
      │
      ▼
┌─────────────────┐
│   LangChain      │  PyPDFLoader downloads/reads the PDF
│   (ingestion)    │  RecursiveCharacterTextSplitter chunks it into
└────────┬─────────┘  overlapping ~1500-char segments, reassembled
         │             into a clean text blob (~12k chars, to control
         │             token usage on the free tier)
         ▼
┌─────────────────────────────────────────────┐
│              LangGraph                        │
│         (stateful control flow)               │
│                                                │
│   ┌─────────┐    ┌───────────┐   ┌──────────┐│
│   │ extract │───▶│ summarize │──▶│ critique ││
│   └─────────┘    └───────────┘   └────┬─────┘│
│                        ▲                │      │
│                        └── retry ───────┘      │
│                       (max 1 retry)             │
│                      or end (pass to next stage)│
└──────────────────────┬────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────┐
│              Pydantic AI (x2 agents)          │
│                                                │
│  Agent 1 — Summarizer                         │
│    output_type=PaperSummary (a Pydantic       │
│    BaseModel) → the LLM's raw response is     │
│    validated and coerced into this exact      │
│    schema before the code ever sees it        │
│                                                │
│  Agent 2 — Domain Expert Reviewer             │
│    Takes the validated PaperSummary +         │
│    the user's research context, returns       │
│    2-3 concrete follow-up ideas as plain text │
└──────────────────────┬────────────────────────┘
                        │
                        ▼
                 Streamlit UI
          (renders structured sections,
           raw JSON expandable at the bottom)
```

Both AI agents call **Google Gemini's `gemini-2.5-flash`** model (free tier) via `pydantic-ai`'s `google:` provider.

---

## 🛠️ Tech Stack

| Layer | Tool | What it's doing here |
|---|---|---|
| Ingestion | **LangChain** (`langchain-community`, `langchain-text-splitters`) | `PyPDFLoader` to fetch/parse the PDF; `RecursiveCharacterTextSplitter` to chunk long documents into overlapping segments so no single API call exceeds context limits |
| Orchestration / control flow | **LangGraph** | A `StateGraph` with 3 nodes (`extract`, `summarize`, `critique`) and one conditional edge implementing a genuine retry loop — not a linear chain. State is a `TypedDict` tracking `paper_text`, `summary`, `critique_passed`, `attempts` |
| Structured output / agents | **Pydantic AI** | Two agents, both instantiated against `google:gemini-2.5-flash`. Agent 1 forces output into a `PaperSummary(BaseModel)` — no manual JSON parsing, no regex extraction; the framework validates types (e.g. `relevance_score: int` constrained `1–10` via `Field(ge=1, le=10)`) before returning |
| Data modeling / validation | **Pydantic** | The `PaperSummary` schema itself — `title`, `method`, `key_results: List[str]`, `limitations`, `relevance_score`, `relevance_notes` |
| LLM provider | **Google Gemini API** (`gemini-2.5-flash`) | Free tier: 1,500 requests/day, 15 req/min. Chosen over OpenAI specifically to keep the whole project at zero cost |
| Web UI | **Streamlit** | Single-page form (arXiv ID + research context) → spinner states during each pipeline stage → rendered summary sections → collapsible raw JSON |
| Deployment | **Streamlit Community Cloud** | Free hosting, auto-redeploys on every GitHub push, secrets managed via their dashboard (not committed to the repo) |
| Networking | **requests** | Direct HTTP fetch of the arXiv PDF from `arxiv.org/pdf/{id}.pdf` |

### On CrewAI

The project originally used **CrewAI** for the domain-expert review step (`Agent`, `Task`, `Crew` classes). During deployment, CrewAI's `chromadb` dependency turned out to have an **unresolved upstream bug**: `chromadb`'s `pydantic.v1` compatibility shim crashes on class construction under Python 3.14 (confirmed via [chroma-core/chroma#6546](https://github.com/chroma-core/chroma/issues/6546), and multiple other projects hitting the same wall — a real ecosystem-wide incompatibility as of mid-2026). Streamlit Cloud's build environment also didn't honor a `runtime.txt` Python-version pin.

**Decision made:** rather than fight a platform Python version I don't control, I replaced the CrewAI reviewer with a second Pydantic AI agent running the identical prompting strategy (same "Sign Language Avatar Research Expert" persona, same task). Functionally identical output, zero fragile dependencies — a cleaner architecture that removes an entire extra framework and its dependency surface for no loss of functionality.

---
## 🚀 Local Setup

```bash
git clone https://github.com/sthanasriharsha/paper-digest-agent
cd paper-digest-agent
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
```

Get a free API key: https://aistudio.google.com/apikey (Google account, no billing needed for free tier)

Set it permanently via Windows Environment Variables (`GOOGLE_API_KEY`), or per-session:

```powershell
$env:GOOGLE_API_KEY="AIzaSy..."
```

```bash
# macOS/Linux
export GOOGLE_API_KEY="AIzaSy..."
```

**Test the core pipeline (no UI):**

```bash
python core.py 2511.22940
```

**Run the web app:**

```bash
streamlit run app.py
```

---

## ☁️ Deployment

Hosted free on **Streamlit Community Cloud**, auto-redeploys on every push to `main`. The API key is stored in Streamlit's Secrets manager (Settings → Secrets → `GOOGLE_API_KEY = "..."`), never committed to the repo — the app reads it silently via `st.secrets` / `os.environ`, with no key input shown to end users.

---

## 📌 Tech Stack (short form)

> **Tech Stack:** Python, LangChain, LangGraph, Pydantic AI, Pydantic, Google Gemini API, Streamlit, Git/GitHub, Streamlit Cloud

**Longer form:**

> **Tech Stack:** Python · LangChain (document ingestion, text splitting) · LangGraph (stateful multi-step agent orchestration) · Pydantic AI (structured LLM outputs, multi-agent design) · Google Gemini API · Streamlit · REST APIs · Git/GitHub · Streamlit Community Cloud (CI/CD-style auto-deploy)

---

## 🗺️ Roadmap / Future Work

- Swap the heuristic groundedness check for an LLM-based grader or embedding similarity
- Add caching to avoid re-summarizing the same paper
- Support batch processing of multiple papers
- Move off the free tier to a paid model for higher throughput
- Add proper error handling/retries with exponential backoff on Gemini API calls
- Explore self-hosting via Ollama for zero marginal cost at scale

---

## 📄 License

*(add your license here, e.g. MIT)*
