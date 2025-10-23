Project: Self-Improving AI System (Hackathon)

1. High-Level Objective

To build a multi-agent system that manages the end-to-end machine learning lifecycle. The system is designed to learn from every project, improving both its problem-definition process and its model selection strategies for future tasks.

2. Core Architecture: 3-Agent System

For the hackathon, the system will operate as a 3-agent core, with a main script or simple UI handling the orchestration and user interaction directly.

Workflow Overview

Input: User provides a dataset and an initial prompt.

The Data Agent is activated. It interacts directly with the user to define the business problem and performs an EDA.

Once the BusinessContext is finalized, the Model Agent is activated. It consults memory.json and the BusinessContext to propose a baseline (3 models) and a custom Loss Function (J).

The Eval Agent is activated and runs the Rapid Loop (evaluate, mutate, iterate) until J converges.

The final model is presented directly to the user for User Confirmation.

The Eval Agent runs the Slow Loop (self-reflection) based on the user's confirmation and updates memory.json with new learnings.

The Agents

1. Data Agent (The Business & Data Analyst)

Role: To deeply understand the problem and the data.

Tasks:

Receives the initial dataset and prompt.

Analyzes the initial prompt. Generates clarifying questions for the user (e.g., "What is more important: interpretability or predictive accuracy?").

Iterates on questions with the user until the business problem is perfectly understood.

Uses tools to perform a full Exploratory Data Analysis (EDA): understands target variables, features, distributions, correlations, etc.

Output: A structured BusinessContext.json artifact detailing the task, business rules, and key data insights.

2. Model Agent (The Initial Strategist)

Role: To set the technical strategy for solving the problem.

Tasks:

Receives the BusinessContext.json after the Data Agent is complete.

Consults memory.json (both theory and experiences) to find relevant knowledge.

Defines the custom Loss Function (J) that algorithmically represents the business goal (e.g., J = 0.7*interpretability_score + 0.2*accuracy_score + 0.1*cost_score).

Proposes an initial baseline of 3 models that are well-suited to the problem context.

Output: The J_definition and a list of 3 model_specs for Run 1.

3. Eval Agent (The Optimizer & Sage)

Role: Manages the solution loop (optimization) and the meta-learning loop (reflection).

Tasks (Rapid Loop - Optimization):

Evaluate: Receives model proposals, executes them, and calculates their J score.

Mutate: Analyzes the results and proposes mutations (e.g., hyperparameter tuning, trying new model families based on failure modes).

Iterate: Repeats this evaluation/mutation cycle until J falls below a defined threshold.

Log: Is extremely verbose. It logs its reasoning, decisions, and results at every step (e.g., to run_history.jsonl).

Propose: Submits the final_recommended_model for user review.

Tasks (Slow Loop - Meta-Learning):

Activate: Triggered after receiving User Confirmation (accept or reject).

Reflect: Based on the user's feedback, the agent reviews its own verbose run_history.jsonl logs.

Learn: It extracts key, generalizable learnings (e.g., "For this type of high-cardinality data, RandomForest failed, but GradientBoosting with specific preprocessing succeeded. This is a new pattern.").

Update: It writes these new learnings into the experiences section of memory.json.

3. Future Work (Next Steps)

Introduce a Triage Agent: As a "next step," we will implement a Triage Agent to act as a central orchestrator. This agent will manage all user interaction and agent handoffs, allowing the specialized agents to focus purely on their tasks without requiring direct user-facing logic.