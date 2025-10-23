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
    "You are the Data Agent. You help understand the business problem and the data. "
    "You have access to a code interpreter tool for inspecting CSV contents. "
    "Return ONLY valid JSON for BusinessContext with keys: task, target, business_rules, clarifying_questions, key_insights."
)

MODEL_INSTRUCTIONS = (
    "You are the Model Agent. Using the BusinessContext and Memory, define a custom J (loss) and propose 3 baseline model specs. "
    "Return ONLY JSON: {J_definition, model_specs:[{model_id, rationale}]}."
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
