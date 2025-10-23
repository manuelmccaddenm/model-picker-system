Coding Rules & Standards

This document defines the technical standards for the project, ensuring consistency and velocity.

1. Core Stack

Language: Python 3.10+

Core Libraries:

pandas: For all data manipulation and EDA.

scikit-learn: For the core "model zoo" (LinearRegression, RandomForestClassifier, etc.).

prophet, statsmodels: For the time-series models.

Orchestrator/UI: streamlit (for a simple, data-first UI to interact with the Data Agent and see the Eval Agent's progress).

2. Agent Framework (Based on agent-sdk-doc.md)

SDK: LangGraph (Pending approval from decisions.md).

Structure:

Each agent (Data Agent, Model Agent, Eval Agent) will be a node or a series of nodes in the main graph.

The "Rapid Loop" of the Eval Agent (evaluate, mutate, iterate) will be a cycle within the graph.

The "Slow Loop" (meta-learning) will be the final node in the graph, writing to memory.json.

3. Prompt Engineering (Based on prompting-for-agents.md)

Format: All prompts will be role-based (System messages) and follow a "Role, Tools, Schema" (RTS) pattern.

Location: Prompts will not be hardcoded in Python files. They will be stored as .md or .txt files in a prompts/ directory and loaded at runtime.

Output: All agents must output structured data.

Agent-to-agent communication must be via structured JSON.

Agents interacting with tools must use Pydantic BaseModel schemas for inputs and outputs. This is non-negotiable for stability.

4. Tool Development

All tools (e.g., perform_eda, train_eval_model) will be defined in src/tools/.

Tools must be strongly typed using Pydantic BaseModel for their arguments (args_schema).

Tools must have clear docstrings, as the agent will use these to decide how and when to call the tool.

5. State Management

The central state of our application will be a TypedDict (or Pydantic model) managed by LangGraph.

This state will include:

business_context: dict (Output of Data Agent)

j_definition: dict (Output of Model Agent)

run_history: List[dict] (The run_history.jsonl log, appended to by Eval Agent)

final_recommendation: dict

6. Logging & Artifacts

run_history.jsonl: This is the most critical artifact. The Eval Agent must log a structured JSON line for every single model run (including J score, model params, and its reasoning for the mutation).

memory.json: This is our persistent meta-learning database.

logging: Use the standard Python logging module for console-based debug messages.

7. Code Style & Quality

Formatter: black

Linter: flake8

Type Hints: All function definitions must include type hints.