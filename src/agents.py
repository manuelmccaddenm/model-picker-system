from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel
import agents as sdk
from agents import Runner, CodeInterpreterTool


# =========================
# Pydantic output contracts
# =========================

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


class ModelSpec(BaseModel):
    model_id: str
    rationale: str


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


# =========================
# Agent instructions
# =========================

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
    "You are the Model Agent. Using the BusinessContext and Memory, define a custom J (loss) and propose 3 baseline model specs. "
    "Honor BusinessContext.objective and BusinessContext.primary_metric if present. "
    "Return ONLY JSON: {J_definition, model_specs:[{model_id, rationale}]}."
)

EVAL_INSTRUCTIONS = (
    "You are the Eval Agent. Manage the Rapid Loop conceptually (evaluate, mutate, iterate) and the Slow Loop (reflection). "
    "For Rapid Loop, return ONLY JSON: {iterations:[{evaluation, mutations}], final_recommendation:{model_id, rationale}}. "
    "For Slow Loop, return ONLY JSON: {learnings:[], suggested_memory_updates:{}}."
)


# =========================
# Builders
# =========================

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
