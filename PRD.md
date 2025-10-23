# Model Picker System - PRD

## 1. Objective
Build a 3-agent system (Data, Model, Eval) that manages the ML lifecycle end-to-end and learns from experience to improve future decisions.

## 2. Users and Scope
- Primary user: Data Scientist / ML Engineer during exploration and baseline selection.
- Scope: Tabular datasets (classification, regression); conceptual evaluation loop (no training code), meta-learning via memory updates.

## 3. High-Level Architecture
- Data Agent: Understands the problem and performs EDA; outputs BusinessContext.
- Model Agent: Uses BusinessContext + Memory to define J and propose 3 baseline model_specs.
- Eval Agent: Rapid Loop (evaluate, mutate, iterate) → final_recommendation; Slow Loop → learnings and suggested memory updates.
- Orchestrator: `src/loop.py` sequences agents and writes artifacts to `logs/` and `memory/`.

## 4. Memory Schema (v1.0)
Root fields:
- schema_version: "1.0"
- updated_at: ISO timestamp
- teoria: [ModelTheory]
- experiencias: [Experience]
- lecciones_analista: [AnalystLesson]

ModelTheory additions (to current structure):
- prerequisites: {scaling_required: bool, handles_missing: "native|impute|no", categorical_handling: "onehot|ordinal|native", data_size_min: int|null}
- typical_preprocessing: [string]
- typical_failure_modes: [string]

Experience structure:
- id (uuid), timestamp, task
- dataset_fingerprint: {rows, cols, target, column_types, notes}
- business_context: {goal, constraints[], tradeoffs{interpretabilidad, precision_predictiva, costo_computacional, robustez_outliers}}
- J: {definition, weights{...}}
- candidates: [{model_id, rationale, assumed_prereqs[]}]
- selected_model: {model_id, hyperparams{}, preprocessing[]}
- results: {metrics{}, J_score, calibration_notes}
- failure_modes_observed[]
- feature_importance: {method, top[{feature, importance}]}|null
- run_metadata: {runtime_seconds, compute_cost, env{python, libs{}}}
- takeaways[]

AnalystLesson structure:
- id (uuid), question, why, trigger_conditions{task?, patterns?, thresholds?}, priority (1-3), last_updated

## 5. Memory Tool
Module: `src/memory_tool.py`
- ensure_schema(path) -> Dict: initializes schema_version/updated_at and required lists.
- build_indexes(mem) -> {by_task, by_tag}
- get_theory(mem, task?, tags?, limit?) -> [ModelTheory]
- rank_candidates_by_J(models, J_weights) -> [(model_id, score, model)]
- get_experiences(mem, task?, k=5) -> [Experience]
- add_experience(path, experience) -> id
- add_lesson(path, lesson) -> id

## 6. Agent SDK Integration
- Agents defined with OpenAI Agents SDK (`agents.Agent`, `Runner.run_sync`).
- Data Agent uses `CodeInterpreterTool` to inspect CSVs.
- Pydantic output models enforce structured outputs.

## 7. Orchestration & Artifacts
- Inputs: dataset path, user prompt
- Outputs:
  - logs/BusinessContext.json
  - logs/run_history.jsonl
  - memory/memory.json (updated with experiences/learnings)

## 8. Future Work
- Add guardrails and validation for Agent outputs.
- Implement retrieval scoring for experiences (schema similarity).
- Extend to time series agents/tools.
- Introduce Triage Agent for multi-agent routing.
