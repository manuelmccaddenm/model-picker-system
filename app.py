from __future__ import annotations

import json
from pathlib import Path
import argparse
from agents import Runner
from src.agents import (
    build_data_agent,
    build_model_agent,
    build_eval_agent,
    BusinessContext,
    ModelProposals,
    RapidLoopResult,
    SlowLoopResult,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agent-driven orchestration")
    parser.add_argument("dataset", help="Path to CSV dataset")
    parser.add_argument("--prompt", dest="prompt", default="", help="User prompt for Data Agent (if empty, agent will ask)")
    parser.add_argument("--hpo_budget", dest="hpo_budget", type=int, default=4, help="HPO trials per candidate (for Eval tool)")
    parser.add_argument("--w_perf", dest="w_perf", type=float, default=0.6, help="Weight for performance in J")
    parser.add_argument("--w_interp", dest="w_interp", type=float, default=0.3, help="Weight for interpretability in J")
    parser.add_argument("--w_cost", dest="w_cost", type=float, default=0.1, help="Weight for computational cost in J")
    args = parser.parse_args()

    dataset_path = str(Path(args.dataset).resolve())
    
    # If no prompt given, use a minimal one that forces the agent to ask
    if not args.prompt.strip():
        args.prompt = "I have a dataset. Help me understand what I can do with it."

    # 1) Data Agent loop: ask until we have task and target
    data_agent = build_data_agent()
    user_context = {
        "DATASET_CSV_PATH": dataset_path,
        "INSTRUCTIONS": args.prompt,
    }
    
    print(f"\n{'='*60}")
    print("DATA AGENT: Understanding the problem")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_path}")
    print(f"Initial prompt: {args.prompt}\n")
    
    MAX_DATA_ITERATIONS = 4
    iteration = 0
    while iteration < MAX_DATA_ITERATIONS:
        iteration += 1
        print(f"\n--- Data Agent Iteration {iteration}/{MAX_DATA_ITERATIONS} ---")
        
        msg = (
            f"You are analyzing a dataset for a machine learning project.\n\n"
            f"DATASET_CSV_PATH: {user_context['DATASET_CSV_PATH']}\n"
            f"INSTRUCTIONS: {user_context['INSTRUCTIONS']}\n"
        )
        if "ANSWERS" in user_context:
            msg += f"\nPREVIOUS ANSWERS:\n{json.dumps(user_context['ANSWERS'], indent=2)}\n"
        
        msg += (
            f"\nREMINDER: You must call perform_eda('{dataset_path}') to profile the data. "
            f"If task or target are unclear, add clarifying_questions."
        )
        
        bc_result = Runner.run_sync(data_agent, msg)
        bc = bc_result.final_output_as(BusinessContext)
        
        # Check if we have clarifying questions to ask
        if bc.clarifying_questions:
            print("\n🤔 Data Agent needs clarification:")
            for q in bc.clarifying_questions:
                ans = input(f"\nQ: {q}\nYour answer: ")
                user_context.setdefault("ANSWERS", []).append({"question": q, "answer": ans})
            continue
        
        # No more questions - check if we have task and target
        if bc.task and bc.target:
            print(f"\n✓ Task identified: {bc.task}")
            print(f"✓ Target identified: {bc.target}")
            break
        
        # If agent didn't provide questions and still missing fields, fail
        print(f"\n⚠️  Data Agent returned incomplete context:")
        print(f"   task: {bc.task or 'MISSING'}")
        print(f"   target: {bc.target or 'MISSING'}")
        print(f"   clarifying_questions: {bc.clarifying_questions or 'NONE'}")
        raise RuntimeError("Data Agent could not determine task/target and provided no clarifying questions.")
    
    # Check if we hit max iterations without completing
    if iteration >= MAX_DATA_ITERATIONS and (not bc.task or not bc.target):
        print(f"\n⚠️  Reached maximum iterations ({MAX_DATA_ITERATIONS}) without complete context:")
        print(f"   task: {bc.task or 'MISSING'}")
        print(f"   target: {bc.target or 'MISSING'}")
        raise RuntimeError(f"Data Agent could not complete after {MAX_DATA_ITERATIONS} iterations. Please provide more specific initial context.")

    print("\n" + "="*60)
    print("DATA AGENT COMPLETE")
    print("="*60)
    print("BusinessContext:", json.dumps(bc.model_dump(), ensure_ascii=False, indent=2))

    # 2) Model Agent: propose baseline using memory tools
    model_agent = build_model_agent()
    model_msg = (
        "BUSINESS_CONTEXT:\n" + json.dumps(bc.model_dump(), ensure_ascii=False) + "\n"
        "Use tools if needed."
    )
    mp_result = Runner.run_sync(model_agent, model_msg)
    mp = mp_result.final_output_as(ModelProposals)
    print("ModelProposals:", json.dumps(mp.model_dump(), ensure_ascii=False, indent=2))

    # 3) Eval Agent: run rapid loop using evaluator tool
    eval_agent = build_eval_agent()
    eval_msg = (
        "BUSINESS_CONTEXT:\n" + json.dumps(bc.model_dump(), ensure_ascii=False) + "\n"
        "PROPOSALS:\n" + json.dumps(mp.model_dump(), ensure_ascii=False) + "\n"
        f"EVAL_ARGS: dataset_path={dataset_path}, target={bc.target}, task={bc.task}, weights={{'precision_predictiva':{args.w_perf}, 'interpretabilidad':{args.w_interp}, 'costo_computacional':{args.w_cost}}}, hpo_budget={args.hpo_budget}"
    )
    rapid_result = Runner.run_sync(eval_agent, eval_msg, output_type=RapidLoopResult)
    rapid = rapid_result.final_output_as(RapidLoopResult)
    print("RapidLoop:", json.dumps(rapid.model_dump(), ensure_ascii=False, indent=2))

    # 4) Slow loop
    slow_msg = (
        "BUSINESS_CONTEXT:\n" + json.dumps(bc.model_dump(), ensure_ascii=False) + "\n"
        "LOGS_TAIL:\n(omitted in MVP)"
    )
    slow_result = Runner.run_sync(eval_agent, slow_msg, output_type=SlowLoopResult)
    slow = slow_result.final_output_as(SlowLoopResult)
    print("SlowLoop:", json.dumps(slow.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
