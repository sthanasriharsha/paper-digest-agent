"""
Streamlit UI for Paper Digest Agent.
Run locally: streamlit run app.py
Deploy free at: https://share.streamlit.io (Streamlit Community Cloud)
"""

import os
import json
import streamlit as st

st.set_page_config(page_title="Paper Digest Agent", page_icon="🧠", layout="centered")

st.title("🧠 Paper Digest Agent")
st.caption(
    "LangChain + LangGraph + Pydantic AI + CrewAI — summarizes a research paper and "
    "suggests follow-ups for your ISL avatar research. 100% free (runs on Google Gemini's free tier)."
)

with st.sidebar:
    st.header("Setup")
    st.markdown("Get a free key: [aistudio.google.com/apikey](https://aistudio.google.com/apikey)")
    default_key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
    api_key = st.text_input(
        "Gemini API Key", type="password", key="gemini_key_input", value=default_key
    )
    if default_key:
        st.caption("✓ Pre-filled from your environment variable")

# --- Everything the user needs to fill in lives inside one form, so all values
#     are captured together the moment "Summarize Paper" is clicked. This avoids
#     the widget-timing bug where a just-pasted value isn't registered yet. ---
with st.form("digest_form"):
    arxiv_input = st.text_input(
        "arXiv ID or URL", placeholder="e.g. 2511.22940 or https://arxiv.org/abs/2511.22940"
    )
    research_context = st.text_area(
        "Your current research context (optional)",
        value="Working on ISL avatar video generation using pose-guided diffusion (WanVideo / ComfyUI).",
    )
    submitted = st.form_submit_button("Summarize Paper", type="primary")

if submitted:
    key_value = (api_key or "").strip()
    arxiv_value = (arxiv_input or "").strip()

    if not key_value:
        st.error("Please enter your free Gemini API key in the sidebar, then click Summarize Paper again.")
    elif not arxiv_value:
        st.error("Please enter an arXiv ID or URL.")
    else:
        # Set env vars BEFORE importing core, so the pydantic-ai / CrewAI clients
        # inside core.py are constructed with the correct key.
        os.environ["GEMINI_API_KEY"] = key_value
        os.environ["GOOGLE_API_KEY"] = key_value

        from core import download_arxiv_pdf, summarize_paper

        with st.spinner("Downloading paper..."):
            pdf_path = download_arxiv_pdf(arxiv_value)

        with st.spinner("Extracting → summarizing → critiquing (LangGraph loop)..."):
            result = summarize_paper(pdf_path, research_context)

        summary = result["summary"]

        st.success(f"Done in {result['attempts']} pass(es). Grounded: {result['grounded']}")

        st.subheader(summary["title"])
        st.markdown(f"**Relevance score:** {summary['relevance_score']}/10")
        st.markdown(f"**Why:** {summary['relevance_notes']}")

        st.markdown("### Method")
        st.write(summary["method"])

        st.markdown("### Key Results")
        for r in summary["key_results"]:
            st.markdown(f"- {r}")

        st.markdown("### Limitations")
        st.write(summary["limitations"])

        st.markdown("### 🔧 Follow-up Ideas (CrewAI domain expert)")
        st.write(result["follow_up_ideas"])

        with st.expander("Raw JSON output"):
            st.code(json.dumps(result, indent=2), language="json")
