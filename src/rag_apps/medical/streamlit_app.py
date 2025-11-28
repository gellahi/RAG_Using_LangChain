from __future__ import annotations

import streamlit as st

from rag_apps.common.logging_utils import get_logger
from .pipeline import build_pipeline


LOGGER = get_logger(__name__)


@st.cache_resource(show_spinner=False)
def load_pipeline():
    LOGGER.info("Loading medical RAG pipeline inside Streamlit")
    return build_pipeline()


def render_sources(sources: list[dict]) -> None:
    if not sources:
        st.info("No citations available.")
        return
    for source in sources:
        st.markdown(
            f"- **{source.get('sample_name', 'Unknown')}** | {source.get('specialty', 'N/A')}\n"
            f"  - Keywords: {source.get('keywords', 'N/A')}"
        )


def main() -> None:
    st.set_page_config(page_title="Medical RAG QA", layout="wide")
    st.title("🩺 Medical RAG QA System")
    st.write("Ask grounded medical questions answered with evidence from the mtsamples corpus.")

    pipeline = load_pipeline()

    with st.form("medical-form", clear_on_submit=False):
        question = st.text_area("Question", height=120, placeholder="What risks were described before Lap-Band surgery?")
        submitted = st.form_submit_button("Generate Answer")

    if submitted and question:
        with st.spinner("Retrieving supporting context and generating answer..."):
            result = pipeline.answer(question)
        st.subheader("Answer")
        st.write(result["answer"])
        st.subheader("Citations")
        render_sources(result["sources"])
    elif submitted:
        st.warning("Please enter a question before submitting.")

    st.caption("Vector store must be built first via `python -m rag_apps.medical.build_vector_store`. ")


if __name__ == "__main__":
    main()
