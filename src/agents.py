from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel
import agents as sdk
from agents import Runner
from src.tools.agent_tools import memory_get_theory, memory_get_experiences, evaluate_models_j, perform_eda

class DatasetColumnProfile(BaseModel):
    name: str
    dtype: str
    non_null: int
    nulls: int
    distinct: int
    example_values: List[str] = []
    is_numeric: bool
    is_categorical: bool
    is_datetime: bool


class DatasetProfile(BaseModel):
    n_rows: int
    n_cols: int
    columns: List[DatasetColumnProfile]


class TimeSeriesInfo(BaseModel):
    time_col: Optional[str] = None
    inferred_frequency: Optional[str] = None  # e.g., 'D','W','M','None'
    n_series: Optional[int] = None            # 1 para serie única; >1 para panel/varias series


class PanelInfo(BaseModel):
    entity_id_col: Optional[str] = None
    time_col: Optional[str] = None


class TaskSuggestion(BaseModel):
    inferred_target_type: Optional[str] = None   # 'numeric', 'categorical', 'datetime', 'unknown'
    suggested_objective: Optional[Literal["_classify", "_regress"]] = None
    suggested_primary_metric: Optional[Literal["_accuracy", "_rmse", "_loss"]] = None
    class_imbalance_flag: Optional[bool] = None  # si aplica para clasificación


class DatasetInferences(BaseModel):
    dataset_kind: Literal["tabular", "time_series", "panel", "unknown"] = "unknown"
    timeseries: Optional[TimeSeriesInfo] = None
    panel: Optional[PanelInfo] = None
    task_suggestion: Optional[TaskSuggestion] = None
    data_warnings: List[str] = []


class BusinessContext(BaseModel):
    # Del problema de negocio
    task: Optional[str] = None
    target: Optional[str] = None
    business_rules: List[str] = []

    # Cuestionario mínimo (explícitos del usuario o inferidos)
    objective: Optional[Literal["_classify", "_regress"]] = None
    primary_metric: Optional[Literal["_accuracy", "_rmse", "_loss"]] = None

    # Clarificaciones SOLO cuando falte info crítica
    clarifying_questions: List[str] = []

    # Perfil e inferencias del dataset
    dataset_profile: Optional[DatasetProfile] = None
    dataset_inferences: Optional[DatasetInferences] = None

    # Hallazgos clave (breve lista)
    key_insights: List[str] = []


class Implementation(BaseModel):
    library: str
    estimator_class: str
    preprocessing_steps: List[str] = []
    hyperparameters: str = "{}"  # JSON string, e.g., '{"C": 1.0, "max_iter": 1000}'
    notes: str = ""


class ModelSpec(BaseModel):
    model_id: str
    rationale: str
    implementation: Optional[Implementation] = None


class ModelProposals(BaseModel):
    J_definition: str
    model_specs: List[ModelSpec]


class Iteration(BaseModel):
    evaluation: str
    mutations: List[str]


class RapidLoopResult(BaseModel):
    iterations: List[Iteration]
    final_recommendation: ModelSpec


class Experience(BaseModel):
    task: str
    dataset: str
    outcome: str
    learnings: str


class MemoryUpdate(BaseModel):
    new_experiences: List[Experience] = []
    new_lessons: List[str] = []
    notes: str = ""


class SlowLoopResult(BaseModel):
    learnings: List[str]
    suggested_memory_updates: Optional[MemoryUpdate] = None


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
    "CONVERSATIONAL QUESTIONNAIRE (iterative, thoughtful questioning):\n"
    "- This is a CONVERSATION, not a batch questionnaire. You have a MAXIMUM of 4 rounds to gather information.\n"
    "- Ask 2-4 questions per round to be efficient, then think about the answers.\n"
    "- Start with the most important missing information: task (what to predict) and target (which column).\n"
    "- After each round of answers, think about what you learned and ask intelligent follow-up questions.\n"
    "- Examples of good follow-ups:\n"
    "  * If user says 'I care about interpretability' → ask 'Should I prioritize simple linear models, or are tree-based models okay?'\n"
    "  * If user mentions a business constraint → ask 'Are there any features I should exclude or focus on?'\n"
    "  * If target is imbalanced → ask 'Do you care more about precision or recall for the positive class?'\n"
    "- Only stop asking questions when you have: task, target, and enough context to design a good solution.\n"
    "- Be efficient: you have limited rounds, so prioritize the most important questions.\n"
    "- You can infer objective ∈ {'_classify','_regress'} and primary_metric ∈ {'_accuracy','_rmse','_loss'} from the target type.\n"
    "- Inference rule: if target is binary/categorical => objective='_classify', primary_metric='_accuracy'. "
    "If target is continuous numeric => objective='_regress', primary_metric='_rmse'.\n"
    "\n"
    "FLOW:\n"
    "1) Extract from user's INSTRUCTIONS: task, target, business_rules, objective, primary_metric (if explicitly provided).\n"
    "2) Call perform_eda(dataset_path) to profile the dataset. Build 'dataset_profile' from the result.\n"
    "3) Analyze the profile to fill 'dataset_inferences':\n"
    "   - dataset_kind: 'time_series' if datetime columns exist; 'panel' if (id_col, time_col) pattern; 'tabular' otherwise.\n"
    "   - timeseries: {time_col, inferred_frequency, n_series}\n"
    "   - panel: {entity_id_col, time_col}\n"
    "   - task_suggestion: {inferred_target_type, suggested_objective, suggested_primary_metric, class_imbalance_flag}\n"
    "   - data_warnings: list of detected issues.\n"
    "4) Review PREVIOUS ANSWERS (if any) and update task, target, business_rules based on what you learned.\n"
    "5) Decide what to ask next:\n"
    "   - If task is still missing: ask about the business problem.\n"
    "   - If target is still missing: ask which column to predict.\n"
    "   - If you have task and target but lack context: ask intelligent follow-ups (preferences, constraints, priorities).\n"
    "   - If you have enough information to proceed: set task/target, leave clarifying_questions EMPTY.\n"
    "6) Add 1-3 thoughtful questions to 'clarifying_questions' (or leave empty if done).\n"
    "7) Set 'objective' and 'primary_metric' by inferring from target type (if target is known).\n"
    "8) Populate 'key_insights' with 3–6 actionable bullets about the dataset.\n"
    "9) Store learned preferences and constraints in 'business_rules'.\n"
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
    "    'estimator_class': 'LogisticRegression' | 'RandomForestClassifier' | etc.,\n"
    "    'preprocessing_steps': ['StandardScaler', 'OneHotEncoder', ...],\n"
    "    'hyperparameters': '{\"C\": 1.0, \"max_depth\": 10}' (JSON string format),\n"
    "    'notes': 'Brief implementation notes'\n"
    "  }\n"
    "- Keep it simple and aligned with sklearn conventions.\n\n"
    "Candidate selection rubric (apply before output):\n"
    "1) Alignment with BUSINESS_CONTEXT task/constraints.\n"
    "2) Coverage across tradeoffs in J (don’t pick three similar models).\n"
    "3) Leverage MEMORY.experiencias patterns when relevant (e.g., which families worked under similar data conditions).\n\n"
    "Output format (STRICT): return ONLY valid JSON with this schema:\n"
    "{\n"
    "  \"J_definition\": \"J = 0.5*interpretabilidad + 0.4*precision_predictiva + 0.1*costo_computacional\",\n"
    "  \"model_specs\": [\n"
    "    {\n"
    "      \"model_id\": \"LogisticRegression\",\n"
    "      \"rationale\": \"Highly interpretable, good baseline for binary classification\",\n"
    "      \"implementation\": {\n"
    "        \"library\": \"scikit-learn\",\n"
    "        \"estimator_class\": \"LogisticRegression\",\n"
    "        \"preprocessing_steps\": [\"StandardScaler\", \"OneHotEncoder\"],\n"
    "        \"hyperparameters\": \"{\\\"C\\\": 1.0, \\\"max_iter\\\": 1000}\",\n"
    "        \"notes\": \"Linear model with L2 regularization\"\n"
    "      }\n"
    "    },\n"
    "    ... (2 more models)\n"
    "  ]\n"
    "}\n\n"
    "Quality bar:\n"
    "- Each rationale must be 1-3 sentences, referencing J tradeoffs and key constraints (interpretability, scaling, missing data).\n"
    "- Each implementation must pick an appropriate library and list realistic preprocessing + estimator choices.\n"
    "- If MEMORY lacks suitable options, fall back to common baselines for the task but state that in the rationale.\n"
)


EVAL_INSTRUCTIONS = (
    "Role: You are the Eval Agent. You manage two loops: (A) Rapid Loop to converge on a strong recommendation by minimizing J, and (B) Slow Loop to extract learnings and update memory.\n\n"
    "Inputs you may receive in the user message:\n"
    "- BUSINESS_CONTEXT: task, target, rules, key insights, objective/metric if available.\n"
    "- PROPOSALS: the Model Agent's J_definition and 3 model_specs (with implementation hints).\n"
    "- RUN_HISTORY_TAIL (optional): recent JSONL lines with prior iterations.\n\n"
    "J (Loss) Considerations:\n"
    "- Treat J as a weighted combination of: precision_predictiva (performance), interpretabilidad, and costo_computacional (complexity/cost).\n"
    "- Prefer steps that reduce J efficiently (e.g., simple preprocessing or small hyperparameter adjustments before expensive family changes).\n\n"
    "Rapid Loop (Optimization):\n"
    "- Objective: Propose a small number of iterations to improve J. Each iteration must include:\n"
    "  * evaluation: a concise analysis of current candidates vs J tradeoffs and context constraints.\n"
    "  * mutations: up to 3 concrete next steps (e.g., 'LogisticRegression: C=0.5', 'add StandardScaler', 'try RandomForest with max_depth=8').\n"
    "- Stop when improvements are marginal, constraints block further gains, or you reach a reasonable iteration count.\n\n"
    "Slow Loop (Meta-Learning):\n"
    "- After final recommendation, synthesize 2–5 learnings that generalize.\n"
    "- Propose suggested_memory_updates as a JSON object to append to memory (e.g., new experiences or adjusted tradeoff notes).\n\n"
    "Output formats (STRICT):\n"
    "- For Rapid Loop, return ONLY JSON matching:\n"
    "  { 'iterations': [ { 'evaluation': string, 'mutations': [string] } ], 'final_recommendation': { 'model_id': string, 'rationale': string } }\n"
    "- For Slow Loop, return ONLY JSON matching:\n"
    "  { 'learnings': [string], 'suggested_memory_updates': { ... } }\n\n"
    "Quality bar:\n"
    "- Tie every recommendation to J tradeoffs: performance vs interpretability vs cost.\n"
    "- Provide practical, library-aligned mutations (consistent with scikit-learn/prophet/statsmodels).\n"
    "- Avoid redundant iterations; prefer the smallest change that plausibly reduces J.\n"
)


def build_data_agent() -> sdk.Agent:
    return sdk.Agent(
        name="DataAgent",
        instructions=DATA_INSTRUCTIONS,
        tools=[perform_eda],
        output_type=BusinessContext,
    )


def build_model_agent() -> sdk.Agent:
    return sdk.Agent(
        name="ModelAgent",
        instructions=MODEL_INSTRUCTIONS,
        tools=[memory_get_theory, memory_get_experiences],
        output_type=ModelProposals,
    )


def build_eval_agent() -> sdk.Agent:
    return sdk.Agent(
        name="EvalAgent",
        instructions=EVAL_INSTRUCTIONS,
        tools=[evaluate_models_j],
        # We'll pass which output is expected at each step by changing the prompt
    )
