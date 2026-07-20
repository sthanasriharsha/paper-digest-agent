"""
Streamlit UI for Paper Digest Agent.
Run locally: streamlit run app.py
Deploy free at: https://share.streamlit.io (Streamlit Community Cloud)

API key is read silently from Streamlit secrets or environment variables —
no key input field is shown to the user.
"""

import os
import json
import streamlit as st

st.set_page_config(page_title="Paper Digest Agent", page_icon="🧠", layout="centered")

st.title("🧠 Paper Digest Agent")
st.caption(
    "LangChain + LangGraph + Pydantic AI — summarizes a research paper and "
    "suggests follow-ups for your ISL avatar research."
)


def _get_secret(name):
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


API_KEY = (
    os.environ.get("GEMINI_API_KEY", "")
    or os.environ.get("GOOGLE_API_KEY", "")
    or _get_secret("GEMINI_API_KEY")
    or _get_secret("GOOGLE_API_KEY")
)

if API_KEY:
    os.environ["GEMINI_API_KEY"] = API_KEY
    os.environ["GOOGLE_API_KEY"] = API_KEY

with st.form("digest_form"):
    arxiv_input = st.text_input(
        "arXiv ID or URL", placeholder="e.g. 2511.22940 or https://arxiv.org/abs/2511.22940"
    )
    research_context = st.text_area(
        "Your current research context (optional)",
        value="Your description",
    )
    submitted = st.form_submit_button("Summarize Paper", type="primary")

if submitted:
    arxiv_value = (arxiv_input or "").strip()

    if not API_KEY:
        st.error(
            "No Gemini API key configured. Add GEMINI_API_KEY under "
            "Settings → Secrets in Streamlit Cloud, or set it as an environment "
            "variable if running locally."
        )
    elif not arxiv_value:
        st.error("Please enter an arXiv ID or URL.")
    else:
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

        st.markdown("### 🔧 Follow-up Ideas (domain-expert agent)")
        st.write(result["follow_up_ideas"])

        with st.expander("Raw JSON output"):
            st.code(json.dumps(result, indent=2), language="json")
