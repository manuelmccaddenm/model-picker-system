from typing import Any, Dict, List
from pydantic import BaseModel
import agents as sdk
from agents import Runner, CodeInterpreterTool


class BusinessContext(BaseModel):
    task: str | None
    target: str | None
    business_rules: List[str]
    clarifying_questions: List[str]
    key_insights: List[str]


class ModelSpec(BaseModel):
    model_id: str
    rationale: str
    implementation: Dict[str, Any] | None = None  # library-aware implementation spec


class ModelProposals(BaseModel):
    J_definition: str
    model_specs: List[ModelSpec]


class Iteration(BaseModel):
    evaluation: str
    mutations: List[str]


class RapidLoopResult(BaseModel):
    iterations: List[Iteration]
    final_recommendation: ModelSpec


class SlowLoopResult(BaseModel):
    learnings: List[str]
    suggested_memory_updates: Dict[str, Any]


DATA_INSTRUCTIONS = (
    "You are the Data Agent. Your goal is (1) to understand the problem and (2) to profile the dataset with the Code Interpreter. "
    "You must return ONLY valid JSON that matches the BusinessContext schema. "
    "\n\n"
    "CODE INTERPRETER CAPABILITIES:\n"
    "- It can read CSV/Parquet files when the user provides a path or bytes. Load a sample (e.g., 1,000 rows) if the file is large.\n"
    "- Compute: number of rows/columns, dtypes, null counts, number of distinct values, and example values per column.\n"
    "- Detect datetime columns and try to infer frequency with pandas.infer_freq.\n"
    "- Detect a panel pattern if there are reasonable (id_col, time_col) pairs (e.g., columns that look like entity IDs and time per entity).\n"
    "- Determine the target type when present: 'numeric' if dtype is continuous numeric (float/int with high cardinality), 'categorical' if few categories.\n"
    "- Flag warnings (e.g., >30% nulls in a key column, high categorical cardinality, potential target leakage due to future timestamps, etc.).\n"
    "\n"
    "INITIAL QUESTIONNAIRE (minimal):\n"
    "- objective ∈ {'_classify','_regress'} and primary_metric ∈ {'_accuracy','_rmse','_loss'}.\n"
    "- Inference rule: if the target is 'numeric' => objective='_regress', primary_metric='_rmse'. "
    "If it is 'categorical' => objective='_classify', primary_metric='_accuracy'. "
    "If you cannot infer, leave those fields empty and add a single clear question in 'clarifying_questions'.\n"
    "- Ask questions only when the information is critical and impossible to infer (file path, target name, or when the target is not present in the data).\n"
    "\n"
    "FLOW:\n"
    "1) Extract from the user's instruction: task, target, business_rules, objective, primary_metric (if explicitly provided). "
    "2) Use the Code Interpreter to profile the dataset (if the user provided a path/file). Build 'dataset_profile'.\n"
    "3) With the profile, fill 'dataset_inferences':\n"
    "   - dataset_kind: 'time_series' if there is a valid time column and a single series; 'panel' if there are (id_col, time_col) with multiple series; 'tabular' otherwise.\n"
    "   - timeseries: {time_col, inferred_frequency, n_series}\n"
    "   - panel: {entity_id_col, time_col}\n"
    "   - task_suggestion: {'inferred_target_type','suggested_objective','suggested_primary_metric','class_imbalance_flag'}\n"
    "   - data_warnings: list of strings describing detected issues.\n"
    "4) Set 'objective' and 'primary_metric' as follows: if the user provided them, honor them; otherwise use 'task_suggestion'. "
    "If you cannot suggest, leave them empty and add a single question in 'clarifying_questions'.\n"
    "5) Populate 'key_insights' with 3–6 short, actionable bullets about the dataset (e.g., 'target is binary with 12% positives; potential imbalance').\n"
    "\n"
    "OUTPUT: Return ONLY valid JSON following the BusinessContext schema "
    "with keys: task, target, business_rules, objective, primary_metric, clarifying_questions, "
    "dataset_profile, dataset_inferences, key_insights."
)

MODEL_INSTRUCTIONS = (
    "Role: You are the Model Agent. You design the initial technical plan based on the BusinessContext and the Memory.\n"
    "Primary Goal: Define a business-aligned loss J and propose exactly 3 strong, diverse baseline model candidates.\n\n"
    "Inputs you will receive in the user message:\n"
    "- BUSINESS_CONTEXT: includes task, target, rules, clarifying questions, key insights (schema/EDA).\n"
    "- MEMORY: includes teoria (model catalog with tradeoffs) and experiencias (past outcomes).\n\n"
    "Grounding & Constraints:\n"
    "- Prefer models that exist in MEMORY.teoria for the given task; use their tradeoff_scores and tags.\n"
    "- Ensure candidates are diverse in family (e.g., linear vs ensemble vs probabilistic) when possible.\n"
    "- Respect obvious prerequisites (e.g., scaling_required) and business rules from BUSINESS_CONTEXT.\n"
    "- Do not propose more than 3 models. Do not invent non-existent model names.\n"
    "- Avoid redundancy (no duplicates).\n\n"
    "How to design J (Loss):\n"
    "- Translate business tradeoffs (interpretability, precision_predictiva, costo_computacional, robustez_outliers if applicable) into an explicit weighted formula.\n"
    "- Express J as a single-line formula string (e.g., 'J = 0.6*interpretabilidad + 0.3*precision_predictiva + 0.1*costo_computacional').\n"
    "- Weights should sum approximately to 1. If tradeoffs are missing, choose sensible defaults and justify briefly in rationale text.\n\n"
    "Implementation specification (must align with Coding Rules):\n"
    "- Use the Python stack: pandas for data handling; scikit-learn for tabular models; prophet or statsmodels for time series.\n"
    "- For EACH model_spec, provide an 'implementation' object with fields:\n"
    "  {\n"
    "    'library': 'scikit-learn' | 'prophet' | 'statsmodels',\n"
    "    'imports': [strings],\n"
    "    'preprocessing': [steps],\n"
    "    'estimator': { 'class': string, 'params': { ... } },\n"
    "    'fit_predict': 'pseudo-code block showing X/y split, fit, predict',\n"
    "    'metrics': [ 'accuracy' | 'f1' | 'rmse' | 'mae' | 'mape' ]\n"
    "  }\n"
    "- Keep code as concise pseudo-code compatible with the stated libraries; no full scripts.\n\n"
    "Candidate selection rubric (apply before output):\n"
    "1) Alignment with BUSINESS_CONTEXT task/constraints.\n"
    "2) Coverage across tradeoffs in J (don’t pick three similar models).\n"
    "3) Leverage MEMORY.experiencias patterns when relevant (e.g., which families worked under similar data conditions).\n\n"
    "Output format (STRICT): return ONLY valid JSON with this schema:\n"
    "{\n"
    "  \"J_definition\": string,\n"
    "  \"model_specs\": [\n"
    "    { \"model_id\": string, \"rationale\": string, \"implementation\": { ... } },\n"
    "    { \"model_id\": string, \"rationale\": string, \"implementation\": { ... } },\n"
    "    { \"model_id\": string, \"rationale\": string, \"implementation\": { ... } }\n"
    "  ]\n"
    "}\n\n"
    "Quality bar:\n"
    "- Each rationale must be 1-3 sentences, referencing J tradeoffs and key constraints (interpretability, scaling, missing data).\n"
    "- Each implementation must pick an appropriate library and list realistic preprocessing + estimator choices.\n"
    "- If MEMORY lacks suitable options, fall back to common baselines for the task but state that in the rationale.\n"
)


EVAL_INSTRUCTIONS = (
    "You are the Eval Agent. Manage the Rapid Loop conceptually (evaluate, mutate, iterate) and the Slow Loop (reflection). "
    "For Rapid Loop, return ONLY JSON: {iterations:[{evaluation, mutations}], final_recommendation:{model_id, rationale}}. "
    "For Slow Loop, return ONLY JSON: {learnings:[], suggested_memory_updates:{}}."
)


def build_data_agent() -> sdk.Agent:
    return sdk.Agent(
        name="DataAgent",
        instructions=DATA_INSTRUCTIONS,
        tools=[CodeInterpreterTool()],
        output_type=BusinessContext,
    )


def build_model_agent() -> sdk.Agent:
    return sdk.Agent(
        name="ModelAgent",
        instructions=MODEL_INSTRUCTIONS,
        tools=[],
        output_type=ModelProposals,
    )


def build_eval_agent() -> sdk.Agent:
    return sdk.Agent(
        name="EvalAgent",
        instructions=EVAL_INSTRUCTIONS,
        tools=[],
        # We'll pass which output is expected at each step by changing the prompt
    )
