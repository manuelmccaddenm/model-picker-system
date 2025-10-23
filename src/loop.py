from pathlib import Path
import json
from agents import Runner
from .config import DATA_DIR, LOGS_DIR, MEMORY_DIR
from .utils import write_json, append_jsonl, read_json
from .agents import (
    build_data_agent,
    build_model_agent,
    build_eval_agent,
    BusinessContext,
    ModelProposals,
    RapidLoopResult,
    SlowLoopResult,
)


def run_project(dataset_filename: str, user_prompt: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = str(DATA_DIR / dataset_filename)
    business_context_path = LOGS_DIR / "BusinessContext.json"
    run_history_path = LOGS_DIR / "run_history.jsonl"
    memory_path = MEMORY_DIR / "memory.json"

    data_agent = build_data_agent()
    data_prompt = (
        f"USER_PROMPT: {user_prompt}\n"
        f"DATASET_CSV_PATH: {dataset_path}\n"
        "Use the code interpreter tool to load and inspect the CSV as needed."
    )
    bc_result = Runner.run_sync(data_agent, data_prompt)
    business_context = bc_result.final_output_as(BusinessContext)
    write_json(business_context_path, business_context.model_dump())
    append_jsonl(run_history_path, {"phase": "business_context", **business_context.model_dump()})

    memory = read_json(memory_path) or {"theory": [], "experiences": []}
    model_agent = build_model_agent()
    model_prompt = (
        "BUSINESS_CONTEXT:\n" + json.dumps(business_context.model_dump(), ensure_ascii=False) + "\n\n" +
        "MEMORY:\n" + json.dumps(memory, ensure_ascii=False)
    )
    mp_result = Runner.run_sync(model_agent, model_prompt)
    proposals = mp_result.final_output_as(ModelProposals)
    append_jsonl(run_history_path, {"phase": "baseline_proposals", **proposals.model_dump()})

    eval_agent = build_eval_agent()
    rapid_prompt = (
        "BUSINESS_CONTEXT:\n" + json.dumps(business_context.model_dump(), ensure_ascii=False) + "\n\n" +
        "PROPOSALS:\n" + json.dumps(proposals.model_dump(), ensure_ascii=False) + "\n\n" +
        "Return ONLY JSON: {iterations:[{evaluation, mutations}], final_recommendation:{model_id, rationale}}"
    )
    rapid_result = Runner.run_sync(eval_agent, rapid_prompt, output_type=RapidLoopResult)
    rapid = rapid_result.final_output_as(RapidLoopResult)
    append_jsonl(run_history_path, {"phase": "rapid_loop", **rapid.model_dump()})
    final_rec = rapid.final_recommendation
    print("Final recommendation:", json.dumps(final_rec.model_dump()))

    with open(run_history_path, "r", encoding="utf-8") as f:
        logs_tail = "".join(f.readlines()[-200:])
    slow_prompt = (
        "BUSINESS_CONTEXT:\n" + json.dumps(business_context.model_dump(), ensure_ascii=False) + "\n\n" +
        "LOGS_TAIL:\n" + logs_tail + "\n\n" +
        "Return ONLY JSON: {learnings:[], suggested_memory_updates:{}}"
    )
    slow_result = Runner.run_sync(eval_agent, slow_prompt, output_type=SlowLoopResult)
    slow = slow_result.final_output_as(SlowLoopResult)
    append_jsonl(run_history_path, {"phase": "slow_loop", **slow.model_dump()})

    memory.setdefault("experiences", []).append({
        "task": business_context.task,
        "dataset": dataset_filename,
        "final_recommendation": final_rec.model_dump(),
        "J_definition": proposals.J_definition,
        "learnings": slow.learnings,
    })
    suggested = slow.suggested_memory_updates or {}
    if suggested:
        memory.update(suggested)
    write_json(memory_path, memory)
