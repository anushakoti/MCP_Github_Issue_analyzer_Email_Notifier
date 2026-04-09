"""
GitHub Issue Analyzer & Email Notifier — Streamlit Web UI.

Run with:
    streamlit run app.py
"""

import asyncio
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GitHub Issue Analyzer",
    page_icon="🔍",
    layout="wide",
)

# ── Header ────────────────────────────────────────────────────────────────────

st.title("🔍 GitHub Issue Analyzer & Email Notifier")
st.markdown(
    "An agentic AI system powered by **LangGraph** + **AWS Bedrock** + **MCP**. "
    "Enter a GitHub repository and optionally request an email summary."
)
st.divider()

# ── Sidebar — Config ──────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")

    repo_input = st.text_input(
        "Repository (owner/repo)",
        placeholder="e.g. anushakoti/RAG_HealthCare",
    )

    send_email = st.checkbox("📧 Send email notification", value=True)

    recipient_email = st.text_input(
        "Recipient email",
        value=os.environ.get("EMAIL_RECIPIENT", ""),
        disabled=not send_email,
    )

    st.divider()
    st.markdown("**Tech stack**")
    st.markdown("- LangGraph multi-agent workflow")
    st.markdown("- AWS Bedrock (Claude 3.5 Sonnet)")
    st.markdown("- GitHub MCP via Composio")
    st.markdown("- Gmail MCP via Composio")

# ── Main area ─────────────────────────────────────────────────────────────────

col1, col2 = st.columns([2, 1])

with col1:
    custom_prompt = st.text_area(
        "Custom instruction (optional)",
        placeholder="e.g. Focus only on bug issues with no assignee",
        height=80,
    )

analyze_btn = st.button("🚀 Analyze Issues", type="primary", use_container_width=False)

# ── Analysis Logic ────────────────────────────────────────────────────────────

if analyze_btn:
    if not repo_input or "/" not in repo_input:
        st.error("Please enter a valid repository in the format `owner/repo`.")
    else:
        # Build the natural language prompt
        prompt_parts = [f"Analyze all open issues in {repo_input}"]
        if custom_prompt.strip():
            prompt_parts.append(f". {custom_prompt.strip()}")
        if send_email:
            prompt_parts.append(f". Then email a full summary report to {recipient_email}")
        else:
            prompt_parts.append(". Do not send an email, just show me the analysis.")

        full_prompt = "".join(prompt_parts)

        # Lazy import to avoid circular issues at module load
        from graph.graph_builder import build_graph

        initial_state = {
            "messages": [HumanMessage(content=full_prompt)],
            "issues_report": "",
            "next_agent": "",
            "email_sent": False,
        }

        with st.spinner("⏳ Fetching and analyzing GitHub issues via MCP..."):
            try:
                graph = build_graph()
                result = asyncio.run(graph.ainvoke(initial_state))
                success = True
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")
                success = False

        if success:
            st.success("✅ Analysis complete!")

            # Display the report
            st.subheader(f"📊 Issues Report — `{repo_input}`")
            report = result.get("issues_report", "")
            st.markdown(report)

            # Email status
            if result.get("email_sent"):
                st.info(f"📧 Report emailed to **{recipient_email}**")

            # Download button
            st.download_button(
                label="⬇️ Download Report",
                data=report,
                file_name=f"issues_report_{repo_input.replace('/', '_')}.md",
                mime="text/markdown",
            )

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption("GitHub Issue Analyzer & Email Notifier · Built with LangGraph + MCP")
