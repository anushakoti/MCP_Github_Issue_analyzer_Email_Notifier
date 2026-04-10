# 🔍 GitHub Issue Analyzer & Email Notifier

An agentic AI system that uses **MCP (Model Context Protocol)** to connect directly to GitHub, analyze repository issues with an LLM, and automatically send email notifications with an AI-generated summary report — all orchestrated via a **LangGraph multi-agent workflow**.

---

## What It Does

1. **Fetches GitHub issues** — connects to the GitHub MCP server to retrieve open issues from any repository
2. **Analyzes issues with AI** — an LLM agent (AWS Bedrock Claude 3.5 Sonnet) classifies, summarizes, and prioritizes the issues
3. **Generates a structured report** — severity classification, recurring patterns, top blockers, and recommendations
4. **Sends email notification** — an email agent dispatches the report to configured recipients via Gmail MCP
5. **Multi-agent routing** — LangGraph orchestrates the handoff between the GitHub analyzer agent and email notifier agent

---

## Architecture

```
User Input (repo name / query)
          │
          ▼
   LangGraph Workflow
          │
    ┌─────┴──────────────┐
    │                    │
GitHub Analyzer      Email Notifier
    Agent               Agent
    │                    │
GitHub MCP           Gmail MCP
(Composio)           (Composio)
    │                    │
    └─────┬──────────────┘
          │
    AWS Bedrock
    (Claude 3.5 Sonnet)
          │
          ▼
  Issues Report + Email Sent
```

### LangGraph State

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    issues_report: str    # AI-generated analysis
    next_agent: str       # routing: "send_email" | "end"
    email_sent: bool      # True once email dispatched
```

### Folder Structure

```
MCP_Github_Issue_analyzer_Email_Notifier/
├── agents/
│   ├── github_issue_analyzer_agent.py   # ReAct agent: fetches + analyzes issues
│   └── email_notifier_agent.py          # ReAct agent: sends report via Gmail MCP
├── graph/
│   └── graph_builder.py                 # LangGraph workflow, state, routing
├── tools/
│   ├── github_tools.py                  # GitHub MCP client wrapper
│   └── email_tools.py                   # Gmail MCP client wrapper
├── app.py                               # Streamlit web UI
├── cli.py                               # Interactive CLI
└── requirements.txt
```

---

## Tech Stack

| Component | Purpose |
|---|---|
| **LangGraph** | Multi-agent workflow orchestration with stateful routing |
| **LangChain AWS** | AWS Bedrock integration (Claude 3.5 Sonnet) |
| **langchain-mcp-adapters** | Connects agents to GitHub and Gmail MCP servers |
| **Composio MCP** | Hosted MCP servers for GitHub and Gmail |
| **AWS Bedrock / boto3** | LLM inference backend |
| **Streamlit** | Web UI layer |
| **python-dotenv** | Environment variable management |

---

## Getting Started

### Prerequisites

- Python 3.11+
- AWS account with Bedrock access enabled (Claude 3.5 Sonnet)
- [Composio](https://composio.dev) account with GitHub and Gmail integrations connected

### Installation

```bash
git clone https://github.com/anushakoti/MCP_Github_Issue_analyzer_Email_Notifier.git
cd MCP_Github_Issue_analyzer_Email_Notifier
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file in the project root:

```env
# AWS Bedrock
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1

# Composio (GitHub + Gmail MCP servers)
COMPOSIO_API_KEY=your_composio_api_key

# Email recipient
NOTIFICATION_EMAIL=recipient@example.com
```

---

## Running the App

### CLI (Terminal)

```bash
python cli.py
```

**Example interaction:**

```
============================================================
  🔍 GitHub Issue Analyzer & Email Notifier
============================================================
  Powered by: LangGraph + AWS Bedrock + MCP

You: Analyze open issues in anushakoti/RAG_HealthCare and email me a summary

⏳ Analyzing repository issues...

Agent:
## GitHub Issues Analysis Report

### Summary
- Total open issues: 12  |  Date range: Jan 2024 – Apr 2025

### Severity Classification
- 🔴 High Priority: #4 NullPointerException in query pipeline, #7 Auth token expiry bug
- 🟡 Medium Priority: #2 Slow embedding generation, #9 Missing retry logic
- 🟢 Low Priority: #1 Update README, #11 Add typing hints

...

📧 Email notification dispatched successfully.
```

### Streamlit Web UI

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

Features:
- Enter any `owner/repo` to analyze
- Toggle email notification on/off
- Download the report as a Markdown file

---

## How the Agents Work

### 1. GitHub Analyzer Agent (`agents/github_issue_analyzer_agent.py`)

- Built with `create_react_agent` (LangGraph prebuilt)
- Uses `langchain-mcp-adapters` to connect to the GitHub MCP server via Composio
- Autonomously calls MCP tools (`list_issues`, `get_issue`, `list_issue_comments`) as needed
- LLM produces a structured report with severity, patterns, and recommendations

### 2. Email Notifier Agent (`agents/email_notifier_agent.py`)

- Built with `create_react_agent` + Gmail MCP tools
- Receives the `issues_report` string from LangGraph state
- Composes and dispatches a formatted HTML email via the `send_email` MCP tool
- Sets `email_sent = True` in graph state on success

### 3. LangGraph Workflow (`graph/graph_builder.py`)

- `START → github_analyzer_node` — always runs first
- `github_analyzer_node → should_send_email()` — conditional edge
  - If user prompt contains "email / notify / send" → routes to `email_notifier_node`
  - Otherwise → routes directly to `END`
- `email_notifier_node → END`

---

## Key Design Decisions

**Why LangGraph?**
LangGraph's typed state and conditional edges make agent routing explicit and debuggable — unlike simple LangChain chains where routing logic is buried in prompts.

**Why MCP?**
MCP (Model Context Protocol) provides a standardized way for LLMs to interact with external services. Using `langchain-mcp-adapters`, any MCP-compatible server becomes a set of LangChain tools the agent can call autonomously.

**Why ReAct agents?**
Each agent uses a Reason-Act loop, letting the LLM decide which tools to call and in what order — e.g., the GitHub agent may call `list_issues` then `get_issue` for each result without hardcoded orchestration.
