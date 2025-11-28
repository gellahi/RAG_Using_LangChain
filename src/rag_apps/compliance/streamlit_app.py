from __future__ import annotations

import streamlit as st

from rag_apps.common.logging_utils import get_logger
from .agent import ComplianceAgent, build_agent


LOGGER = get_logger(__name__)


@st.cache_resource(show_spinner=False)
def load_agent() -> ComplianceAgent:
    LOGGER.info("Loading compliance agent inside Streamlit")
    return build_agent()


def render_results(results: list[dict]) -> None:
    if not results:
        st.info("No rules evaluated with the current filters.")
        return
    for row in results:
        color = "🟢" if row["verdict"].lower().startswith("compliant") else "🔴"
        with st.expander(f"{color} {row['rule_id']} • {row['category']} ({row['severity']})"):
            st.write(f"**Verdict:** {row['verdict']}")
            st.write(f"**Evidence:** {row['evidence']}")
            st.write(f"**Remediation:** {row['remediation'] or 'N/A'}")
            if row.get("sources"):
                st.caption("Sources: " + "; ".join(src.get("doc_name", "unknown") for src in row["sources"]))


def main() -> None:
    st.set_page_config(page_title="Compliance Checker", layout="wide")
    st.title("📋 Policy Compliance Checker")
    st.write("Evaluate CUAD contracts against custom policy rules with LangChain + Gemini.")

    agent = load_agent()
    severities = sorted({rule.severity for rule in agent.rules})
    categories = sorted({rule.category for rule in agent.rules})

    with st.form("compliance-form"):
        question = st.text_area(
            "Compliance Question",
            value="Do these agreements meet our security and privacy obligations?",
            height=120,
        )
        selected_severities = st.multiselect("Severities", options=severities, default=severities)
        selected_categories = st.multiselect("Categories", options=categories, default=categories)
        submitted = st.form_submit_button("Run Assessment")

    if submitted:
        active_rules = [
            rule
            for rule in agent.rules
            if rule.severity in selected_severities and rule.category in selected_categories
        ]
        if not question.strip():
            st.warning("Please enter a compliance question.")
            return
        if not active_rules:
            st.warning("No rules match the current filters.")
            return
        with st.spinner(f"Evaluating {len(active_rules)} rules..."):
            results = [agent.assess_rule(rule, question) for rule in active_rules]
        render_results(results)

    st.caption("Build the vector store first via `python -m rag_apps.compliance.build_vector_store`. ")


if __name__ == "__main__":
    main()
