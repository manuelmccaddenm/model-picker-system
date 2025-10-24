from __future__ import annotations

import json
from pathlib import Path
import argparse
from datetime import datetime
import pandas as pd
from agents import Runner
from src.agents import (
    build_data_agent,
    build_model_agent,
    build_eval_agent_rapid,
    build_eval_agent_slow,
    BusinessContext,
    ModelProposals,
    RapidLoopResult,
    SlowLoopResult,
)
from src.config import LOGS_DIR


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
    
    # Setup logging
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_agent_log = LOGS_DIR / f"data_agent_{run_id}.jsonl"
    model_agent_log = LOGS_DIR / f"model_agent_{run_id}.jsonl"
    eval_agent_log = LOGS_DIR / f"eval_agent_{run_id}.jsonl"
    
    def log_agent_interaction(log_file: Path, iteration: int, input_msg: str, output: dict, agent_result, agent_name: str):
        """Log agent input/output/reasoning to JSONL file with full verbosity"""
        # Extract comprehensive reasoning from agent result
        reasoning = {
            "messages": [],
            "tool_calls_summary": [],
            "thinking_steps": []
        }
        
        # Extract all messages from the agent execution
        if hasattr(agent_result, 'all_messages'):
            for msg in agent_result.all_messages():
                msg_data = {
                    "role": getattr(msg, 'role', 'unknown'),
                    "content": None,
                    "tool_calls": [],
                    "tool_call_id": None,
                    "name": None
                }
                
                # Extract content (can be string or list of content blocks)
                if hasattr(msg, 'content'):
                    content = msg.content
                    if isinstance(content, str):
                        msg_data["content"] = content
                    elif isinstance(content, list):
                        # Handle content blocks (text, tool_use, etc.)
                        msg_data["content"] = []
                        for block in content:
                            if hasattr(block, 'text'):
                                msg_data["content"].append({"type": "text", "text": block.text})
                            elif hasattr(block, 'type'):
                                msg_data["content"].append({"type": block.type, "data": str(block)})
                            else:
                                msg_data["content"].append({"type": "unknown", "data": str(block)})
                    else:
                        msg_data["content"] = str(content)
                
                # Extract tool calls
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_call_data = {
                            "id": getattr(tc, 'id', None),
                            "type": getattr(tc, 'type', 'function'),
                            "function": {
                                "name": tc.function.name if hasattr(tc, 'function') else "unknown",
                                "arguments": tc.function.arguments if hasattr(tc, 'function') else "{}"
                            }
                        }
                        msg_data["tool_calls"].append(tool_call_data)
                        
                        # Add to summary
                        reasoning["tool_calls_summary"].append({
                            "tool": tool_call_data["function"]["name"],
                            "arguments": tool_call_data["function"]["arguments"]
                        })
                
                # Extract tool call ID (for tool response messages)
                if hasattr(msg, 'tool_call_id'):
                    msg_data["tool_call_id"] = msg.tool_call_id
                
                # Extract name (for tool messages)
                if hasattr(msg, 'name'):
                    msg_data["name"] = msg.name
                
                reasoning["messages"].append(msg_data)
                
                # Extract thinking steps (assistant messages are the reasoning)
                if msg_data["role"] == "assistant" and msg_data["content"]:
                    if isinstance(msg_data["content"], str):
                        reasoning["thinking_steps"].append(msg_data["content"])
                    elif isinstance(msg_data["content"], list):
                        for block in msg_data["content"]:
                            if isinstance(block, dict) and block.get("type") == "text":
                                reasoning["thinking_steps"].append(block["text"])
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "iteration": iteration,
            "input": input_msg,
            "reasoning": reasoning,
            "output": output,
            "summary": {
                "num_messages": len(reasoning["messages"]),
                "num_tool_calls": len(reasoning["tool_calls_summary"]),
                "num_thinking_steps": len(reasoning["thinking_steps"])
            }
        }
        
        with log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False, indent=2) + "\n")
        
        print(f"📝 Logged {agent_name} interaction to {log_file.name}")
        print(f"   Messages: {log_entry['summary']['num_messages']}, "
              f"Tool calls: {log_entry['summary']['num_tool_calls']}, "
              f"Thinking steps: {log_entry['summary']['num_thinking_steps']}")

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
        
        # Log this interaction (including reasoning)
        log_agent_interaction(
            data_agent_log,
            iteration,
            msg,
            bc.model_dump(),
            bc_result,
            "DataAgent"
        )
        
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
    print("\n" + "="*60)
    print("MODEL AGENT: Proposing baseline models")
    print("="*60)
    model_agent = build_model_agent()
    model_msg = (
        "BUSINESS_CONTEXT:\n" + json.dumps(bc.model_dump(), ensure_ascii=False) + "\n"
        "Use tools if needed."
    )
    mp_result = Runner.run_sync(model_agent, model_msg)
    mp = mp_result.final_output_as(ModelProposals)
    
    # Log Model Agent interaction (including reasoning)
    log_agent_interaction(
        model_agent_log,
        1,
        model_msg,
        mp.model_dump(),
        mp_result,
        "ModelAgent"
    )
    
    print(f"\n✓ Model Agent proposed {len(mp.model_specs)} models")
    print(f"✓ J definition: {mp.J_definition}")
    print("ModelProposals:", json.dumps(mp.model_dump(), ensure_ascii=False, indent=2))

    # 3) Eval Agent: run rapid loop
    # First, ACTUALLY run the evaluator tool (don't let agent hallucinate)
    print("\n" + "="*60)
    print("EVAL AGENT: Running Rapid Loop (Optimization)")
    print("="*60)
    print("🔬 Running real model evaluation with cross-validation and HPO...")
    
    # Import the actual function, not the FunctionTool wrapper
    from src.tools.evaluator import evaluate_with_j_hpo_assumptions
    from src.tools.models import build_candidates_from_proposals
    from src.tools.memory import load_memory
    
    # Prepare evaluation
    df = pd.read_csv(dataset_path)
    memory_path = Path(__file__).parent / "memory" / "memory.json"
    mem = load_memory(memory_path)
    
    # Convert objective to task type for sklearn
    task_type = "classification" if bc.objective == "_classify" else "regression"
    
    plan = {
        "task": task_type,  # Use "classification" or "regression" for sklearn
        "target": bc.target,
        "weights": {
            "precision_predictiva": float(args.w_perf),
            "interpretabilidad": float(args.w_interp),
            "costo_computacional": float(args.w_cost),
        },
    }
    
    # Build candidates from Model Agent's proposals (not hardcoded defaults)
    # Determine confidence-aware settings
    confidence = getattr(mp, "confidence", "medium") or "medium"
    include_fallback = False  # omit fallback for now to surface failures clearly
    hpo_budget = int(args.hpo_budget * (2 if confidence == "low" else (1 if confidence == "medium" else 1)))

    cands = build_candidates_from_proposals(
        df=df,
        target=bc.target,
        proposals=[spec.model_dump() for spec in mp.model_specs],
        task=task_type,
        include_fallback=include_fallback
    )
    
    # Run evaluation
    eval_results = evaluate_with_j_hpo_assumptions(
        df=df,
        plan=plan,
        candidates=cands,
        memory=mem,
        hpo_budget=hpo_budget,
    )
    
    print(f"✓ Evaluation complete. Winner: {eval_results['winner']['name']} with J={eval_results['winner']['J']:.3f}")
    leaderboard_str = ', '.join([f"{r['name']}(J={r['J']:.3f})" for r in eval_results['leaderboard']])
    print(f"✓ Leaderboard: {leaderboard_str}")
    
    # Now let the agent interpret the results and propose mutations
    eval_agent_rapid = build_eval_agent_rapid()
    eval_msg = (
        "BUSINESS_CONTEXT:\n" + json.dumps(bc.model_dump(), ensure_ascii=False) + "\n"
        "PROPOSALS:\n" + json.dumps(mp.model_dump(), ensure_ascii=False) + "\n\n"
        "EVALUATION RESULTS (from evaluate_models_j tool - REAL DATA):\n"
        + json.dumps(eval_results, ensure_ascii=False, indent=2) + "\n\n"
        f"Confidence from Model Agent: {confidence}.\n"
        "Your task: Analyze these REAL results and create a RapidLoopResult with:\n"
        "1. iterations[0].evaluation: Summarize what the leaderboard shows\n"
        "2. iterations[0].mutations: Propose 2-3 ways to improve J further\n"
        "3. Decide if more iterations are needed (if J can improve >5%). If confidence is 'low', be more willing to iterate and try diverse families; if 'high', converge quickly.\n"
        "4. final_recommendation: Pick the winner and explain why\n"
    )
    rapid_result = Runner.run_sync(eval_agent_rapid, eval_msg)
    rapid = rapid_result.final_output_as(RapidLoopResult)
    
    # Log Rapid Loop (including reasoning)
    log_agent_interaction(
        eval_agent_log,
        1,
        eval_msg,
        rapid.model_dump(),
        rapid_result,
        "EvalAgent_RapidLoop"
    )
    
    print(f"\n✓ Rapid Loop completed: {len(rapid.iterations)} iterations")
    print(f"✓ Final recommendation: {rapid.final_recommendation.model_id}")
    print("RapidLoop:", json.dumps(rapid.model_dump(), ensure_ascii=False, indent=2))

    # 4) Slow loop
    print("\n" + "="*60)
    print("EVAL AGENT: Running Slow Loop (Meta-Learning)")
    print("="*60)
    eval_agent_slow = build_eval_agent_slow()
    # Read full logs from this run to inform Slow Loop learnings
    try:
        data_logs_text = data_agent_log.read_text(encoding="utf-8") if data_agent_log.exists() else ""
    except Exception:
        data_logs_text = ""
    try:
        model_logs_text = model_agent_log.read_text(encoding="utf-8") if model_agent_log.exists() else ""
    except Exception:
        model_logs_text = ""
    try:
        eval_logs_text = eval_agent_log.read_text(encoding="utf-8") if eval_agent_log.exists() else ""
    except Exception:
        eval_logs_text = ""

    slow_msg = (
        "BUSINESS_CONTEXT:\n" + json.dumps(bc.model_dump(), ensure_ascii=False) + "\n"
        "RAPID_LOOP_RESULT:\n" + json.dumps(rapid.model_dump(), ensure_ascii=False) + "\n"
        "LOGS (FULL RUN):\n"
        "--- DATA AGENT LOGS ---\n" + data_logs_text + "\n"
        "--- MODEL AGENT LOGS ---\n" + model_logs_text + "\n"
        "--- EVAL AGENT LOGS ---\n" + eval_logs_text + "\n"
    )
    slow_result = Runner.run_sync(eval_agent_slow, slow_msg)
    slow = slow_result.final_output_as(SlowLoopResult)
    
    # Log Slow Loop (including reasoning)
    log_agent_interaction(
        eval_agent_log,
        2,
        slow_msg,
        slow.model_dump(),
        slow_result,
        "EvalAgent_SlowLoop"
    )
    
    print(f"\n✓ Slow Loop completed: {len(slow.learnings)} learnings extracted")
    print("SlowLoop:", json.dumps(slow.model_dump(), ensure_ascii=False, indent=2))
    
    # 5) Update memory with new experiences and lessons
    if slow.suggested_memory_updates:
        print("\n" + "="*60)
        print("UPDATING MEMORY")
        print("="*60)
        
        from src.tools.memory import add_experience, add_lesson
        
        updates = slow.suggested_memory_updates
        
        # Add new experiences
        if updates.new_experiences:
            print(f"Adding {len(updates.new_experiences)} new experience(s)...")
            for exp in updates.new_experiences:
                exp_id = add_experience(memory_path, exp.model_dump())
                print(f"  ✓ Added experience: {exp_id}")
        
        # Add new lessons
        if updates.new_lessons:
            print(f"Adding {len(updates.new_lessons)} new lesson(s)...")
            for lesson_text in updates.new_lessons:
                lesson_id = add_lesson(memory_path, {
                    "lesson": lesson_text,
                    "source": "slow_loop",
                    "run_id": run_id,
                })
                print(f"  ✓ Added lesson: {lesson_id}")
        
        if updates.notes:
            print(f"Notes: {updates.notes}")
        
        print(f"\n✓ Memory updated successfully!")
    
    # Final summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"📁 Logs saved to:")
    print(f"   - Data Agent: {data_agent_log}")
    print(f"   - Model Agent: {model_agent_log}")
    print(f"   - Eval Agent: {eval_agent_log}")


if __name__ == "__main__":
    main()
