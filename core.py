"""
Paper Digest Agent — core.py
Combines LangChain (ingestion), LangGraph (control flow),
Pydantic AI (structured output), and CrewAI (multi-agent review)
into one pipeline that summarizes a research paper (PDF or arXiv URL).

100% free to run: uses Google Gemini's free API (gemini-2.5-flash) instead of OpenAI.
Get a free key at https://aistudio.google.com/apikey — just needs a Google account.

Requires: GEMINI_API_KEY environment variable.
"""

import os
import re
import tempfile
import requests
from typing import TypedDict, Optional, List

from pydantic import BaseModel, Field

# --- LangChain: document loading + splitting -------------------------------
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- LangGraph: cyclic control flow -----------------------------------------
from langgraph.graph import StateGraph, END

# --- Pydantic AI: structured, validated LLM output --------------------------
from pydantic_ai import Agent as PydanticAIAgent

# --- CrewAI: multi-agent review layer ---------------------------------------
from crewai import Agent, Task, Crew, LLM

GEMINI_MODEL_NAME = "gemini-2.5-flash"  # currently on the free tier (2.0-flash is not, as of mid-2026)
# pydantic-ai's GoogleModel reads GOOGLE_API_KEY; bridge from GEMINI_API_KEY if that's what's set
if os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
PYDANTIC_AI_MODEL_STRING = f"google:{GEMINI_MODEL_NAME}"


# =============================================================================
# 1. PYDANTIC AI — output schema
# =============================================================================
class PaperSummary(BaseModel):
    title: str = Field(description="Paper title")
    method: str = Field(description="One paragraph describing the core method")
    key_results: List[str] = Field(description="3-5 bullet point key results")
    limitations: str = Field(description="Stated or implied limitations")
    relevance_score: int = Field(
        description="1-10 relevance to sign-language avatar / ISL research", ge=1, le=10
    )
    relevance_notes: str = Field(description="Why it scores this way")


summarizer_agent = PydanticAIAgent(
    PYDANTIC_AI_MODEL_STRING,
    output_type=PaperSummary,
    system_prompt=(
        "You are a research assistant specializing in computer vision and "
        "sign-language avatar generation. Summarize the given paper text "
        "into the required structured format. Be precise and avoid inventing "
        "results not present in the text."
    ),
)


# =============================================================================
# 2. LANGCHAIN — ingestion helpers
# =============================================================================
def download_arxiv_pdf(arxiv_id_or_url: str, out_path: str = None) -> str:
    """Accepts either an arXiv ID (e.g. 2511.22940) or a full URL."""
    if out_path is None:
        out_path = os.path.join(tempfile.gettempdir(), "paper.pdf")

    if arxiv_id_or_url.startswith("http"):
        pdf_url = arxiv_id_or_url.replace("/abs/", "/pdf/")
        if not pdf_url.endswith(".pdf"):
            pdf_url += ".pdf"
    else:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id_or_url}.pdf"

    resp = requests.get(pdf_url, timeout=30)
    resp.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(resp.content)
    return out_path


def load_and_chunk(pdf_path: str) -> str:
    """Loads a PDF and returns cleaned, concatenated text (first ~12k chars)."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    chunks = splitter.split_documents(pages)

    full_text = "\n".join(c.page_content for c in chunks)
    full_text = re.sub(r"\s+", " ", full_text)
    # Keep it within a reasonable context budget for the demo
    return full_text[:12000]


# =============================================================================
# 3. LANGGRAPH — extract -> summarize -> critique loop
# =============================================================================
class GraphState(TypedDict):
    paper_text: str
    summary: Optional[PaperSummary]
    critique_passed: bool
    attempts: int


def node_extract(state: GraphState) -> GraphState:
    # In this simple pipeline extraction already happened in load_and_chunk,
    # so this node just validates we have text to work with.
    if not state["paper_text"] or len(state["paper_text"]) < 200:
        raise ValueError("Paper text too short — extraction failed.")
    return state


def node_summarize(state: GraphState) -> GraphState:
    result = summarizer_agent.run_sync(
        f"Summarize this paper text:\n\n{state['paper_text']}"
    )
    state["summary"] = result.output
    state["attempts"] += 1
    return state


def node_critique(state: GraphState) -> GraphState:
    summary = state["summary"]
    # Lightweight grounding check: do key_results terms actually appear
    # (loosely) in the source text? Cheap heuristic to avoid an extra LLM call.
    text_lower = state["paper_text"].lower()
    grounded_hits = sum(
        1 for r in summary.key_results if any(w.lower() in text_lower for w in r.split()[:4])
    )
    state["critique_passed"] = grounded_hits >= max(1, len(summary.key_results) // 2)
    return state


def route_after_critique(state: GraphState) -> str:
    if state["critique_passed"] or state["attempts"] >= 2:
        return "end"
    return "retry"


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("extract", node_extract)
    graph.add_node("summarize", node_summarize)
    graph.add_node("critique", node_critique)

    graph.set_entry_point("extract")
    graph.add_edge("extract", "summarize")
    graph.add_edge("summarize", "critique")
    graph.add_conditional_edges(
        "critique", route_after_critique, {"retry": "summarize", "end": END}
    )
    return graph.compile()


# =============================================================================
# 4. CREWAI — domain-expert review layer
# =============================================================================
def run_crew_review(summary: PaperSummary, research_context: str) -> str:
    llm = LLM(model=f"gemini/{GEMINI_MODEL_NAME}")

    domain_expert = Agent(
        role="Sign Language Avatar Research Expert",
        goal="Evaluate papers for relevance to ISL avatar generation research",
        backstory=(
            "You specialize in skeleton-to-avatar animation, pose-guided video "
            "generation, and Indian Sign Language research. You know models like "
            "UniAnimate-DiT, AnimateAnyone, and WanVideo-based pipelines."
        ),
        llm=llm,
        verbose=False,
    )

    review_task = Task(
        description=(
            f"Given this paper summary:\n{summary.model_dump_json(indent=2)}\n\n"
            f"And this researcher's current work:\n{research_context}\n\n"
            "Write 2-3 concrete follow-up experiments or techniques the "
            "researcher could try, based on this paper. Be specific and actionable."
        ),
        expected_output="A short list of 2-3 concrete, actionable follow-up ideas.",
        agent=domain_expert,
    )

    crew = Crew(agents=[domain_expert], tasks=[review_task], verbose=False)
    result = crew.kickoff()
    return str(result)


# =============================================================================
# 5. FULL PIPELINE
# =============================================================================
def summarize_paper(pdf_path: str, research_context: str = "") -> dict:
    paper_text = load_and_chunk(pdf_path)

    app = build_graph()
    final_state = app.invoke(
        {"paper_text": paper_text, "summary": None, "critique_passed": False, "attempts": 0}
    )

    summary: PaperSummary = final_state["summary"]
    follow_ups = run_crew_review(summary, research_context or "General AI/ML research.")

    return {
        "summary": summary.model_dump(),
        "grounded": final_state["critique_passed"],
        "attempts": final_state["attempts"],
        "follow_up_ideas": follow_ups,
    }


if __name__ == "__main__":
    # Quick CLI test: python core.py 2511.22940
    import sys
    import json

    arxiv_id = sys.argv[1] if len(sys.argv) > 1 else "2511.22940"
    path = download_arxiv_pdf(arxiv_id)
    output = summarize_paper(
        path,
        research_context="Working on ISL avatar video generation using pose-guided diffusion models.",
    )
    print(json.dumps(output, indent=2))