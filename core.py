"""
Paper Digest Agent
-------------------
Takes an arXiv paper and produces:
  1. A structured summary (method, key results, limitations, relevance score)
  2. A short list of concrete follow-up research ideas

Pipeline:
  LangChain   -> downloads the PDF and splits it into clean text chunks
  LangGraph   -> runs a summarize -> critique -> (retry if needed) loop
  Pydantic AI -> two agents: one writes the summary, one reviews it and
                 suggests follow-up experiments

Runs on Google Gemini's free tier, so it costs nothing to use.
Set your API key first:  export GOOGLE_API_KEY=your_key_here
"""

import os
import re
import json
import sys
import tempfile
from typing import TypedDict, Optional, List

import requests
from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from pydantic_ai import Agent


MODEL = "google:gemini-2.5-flash"

# pydantic-ai looks for GOOGLE_API_KEY specifically, so if the user only set
# GEMINI_API_KEY, copy it over.
if os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]


# ---------------------------------------------------------------------------
# Output schema — this is what the summarizer agent is required to return.
# Pydantic validates the structure automatically, so there's no manual
# JSON parsing anywhere in this file.
# ---------------------------------------------------------------------------
class PaperSummary(BaseModel):
    title: str
    method: str = Field(description="A short paragraph describing the core method")
    key_results: List[str] = Field(description="3-5 key findings")
    limitations: str
    relevance_score: int = Field(ge=1, le=10, description="Relevance to sign-language avatar research")
    relevance_notes: str = Field(description="Why it scored the way it did")


summarizer = Agent(
    MODEL,
    output_type=PaperSummary,
    system_prompt=(
        "You are a research assistant specializing in computer vision and "
        "sign-language avatar generation. Summarize the paper text into the "
        "required format. Only report what's actually in the text — don't "
        "invent results."
    ),
)

reviewer = Agent(
    MODEL,
    output_type=str,
    system_prompt=(
        "You are a sign-language avatar research expert, familiar with "
        "pose-guided video generation models like UniAnimate-DiT, "
        "AnimateAnyone, and WanVideo. Given a paper summary and a "
        "researcher's current project, suggest 2-3 specific, actionable "
        "follow-up experiments. Reference real tools and techniques, not "
        "generic advice."
    ),
)


# ---------------------------------------------------------------------------
# Step 1: download and read the paper
# ---------------------------------------------------------------------------
def download_arxiv_pdf(arxiv_id_or_url: str) -> str:
    """Downloads a paper's PDF given either an arXiv ID or a full URL."""
    if arxiv_id_or_url.startswith("http"):
        pdf_url = arxiv_id_or_url.replace("/abs/", "/pdf/")
        if not pdf_url.endswith(".pdf"):
            pdf_url += ".pdf"
    else:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id_or_url}.pdf"

    response = requests.get(pdf_url, timeout=30)
    response.raise_for_status()

    out_path = os.path.join(tempfile.gettempdir(), "paper.pdf")
    with open(out_path, "wb") as f:
        f.write(response.content)
    return out_path


def extract_text(pdf_path: str) -> str:
    """Loads a PDF and returns cleaned text, trimmed to fit comfortably
    within the model's context window."""
    pages = PyPDFLoader(pdf_path).load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    chunks = splitter.split_documents(pages)

    text = " ".join(chunk.page_content for chunk in chunks)
    text = re.sub(r"\s+", " ", text)
    return text[:12000]


# ---------------------------------------------------------------------------
# Step 2: summarize -> critique -> retry (LangGraph)
# ---------------------------------------------------------------------------
class PipelineState(TypedDict):
    paper_text: str
    summary: Optional[PaperSummary]
    is_grounded: bool
    attempts: int


def summarize_step(state: PipelineState) -> PipelineState:
    result = summarizer.run_sync(f"Summarize this paper:\n\n{state['paper_text']}")
    state["summary"] = result.output
    state["attempts"] += 1
    return state


def critique_step(state: PipelineState) -> PipelineState:
    """Sanity-checks the summary against the source text: do the claimed
    key results actually show up in the paper, at least loosely? This is a
    cheap heuristic rather than another LLM call, to keep things fast and
    within the free-tier request limits."""
    text = state["paper_text"].lower()
    summary = state["summary"]

    hits = sum(
        1 for result in summary.key_results
        if any(word.lower() in text for word in result.split()[:4])
    )
    state["is_grounded"] = hits >= max(1, len(summary.key_results) // 2)
    return state


def decide_next_step(state: PipelineState) -> str:
    if state["is_grounded"] or state["attempts"] >= 2:
        return "done"
    return "retry"


def build_pipeline():
    graph = StateGraph(PipelineState)
    graph.add_node("summarize", summarize_step)
    graph.add_node("critique", critique_step)

    graph.set_entry_point("summarize")
    graph.add_edge("summarize", "critique")
    graph.add_conditional_edges(
        "critique", decide_next_step, {"retry": "summarize", "done": END}
    )
    return graph.compile()


# ---------------------------------------------------------------------------
# Step 3: get follow-up ideas from the reviewer agent
# ---------------------------------------------------------------------------
def get_follow_up_ideas(summary: PaperSummary, research_context: str) -> str:
    prompt = (
        f"Paper summary:\n{summary.model_dump_json(indent=2)}\n\n"
        f"Researcher's current project:\n{research_context}\n\n"
        "Suggest 2-3 concrete follow-up experiments based on this paper."
    )
    result = reviewer.run_sync(prompt)
    return result.output


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def summarize_paper(pdf_path: str, research_context: str = "") -> dict:
    paper_text = extract_text(pdf_path)

    pipeline = build_pipeline()
    final_state = pipeline.invoke(
        {"paper_text": paper_text, "summary": None, "is_grounded": False, "attempts": 0}
    )

    summary = final_state["summary"]
    follow_ups = get_follow_up_ideas(summary, research_context or "General AI/ML research.")

    return {
        "summary": summary.model_dump(),
        "grounded": final_state["is_grounded"],
        "attempts": final_state["attempts"],
        "follow_up_ideas": follow_ups,
    }


if __name__ == "__main__":
    arxiv_id = sys.argv[1] if len(sys.argv) > 1 else "2511.22940"
    pdf_path = download_arxiv_pdf(arxiv_id)
    result = summarize_paper(
        pdf_path,
        research_context="Working on ISL avatar video generation using pose-guided diffusion models.",
    )
    print(json.dumps(result, indent=2))
