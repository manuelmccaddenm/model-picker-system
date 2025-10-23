#!/usr/bin/env python3
"""Test Data Agent with minimal context - should ask questions"""
import json
from pathlib import Path
from agents import Runner
from src.agents import build_data_agent, BusinessContext

dataset_path = str(Path("data/sample.csv").resolve())

data_agent = build_data_agent()

# Minimal prompt - agent should ask for more info
msg = (
    f"DATASET_CSV_PATH: {dataset_path}\n"
    f"INSTRUCTIONS: I have a dataset. Help me understand what I can do with it.\n"
)

print("=" * 70)
print("Testing Data Agent with MINIMAL CONTEXT")
print("=" * 70)
print(f"\nDataset: {dataset_path}")
print(f"Prompt: 'I have a dataset. Help me understand what I can do with it.'\n")
print("Expected: Agent should call perform_eda() and then ask clarifying questions")
print("=" * 70)

try:
    print("\nRunning agent...")
    result = Runner.run_sync(data_agent, msg)
    bc = result.final_output_as(BusinessContext)
    
    print("\n" + "=" * 70)
    print("RESULT:")
    print("=" * 70)
    
    print(f"\n✓ Task: {bc.task or '❌ MISSING'}")
    print(f"✓ Target: {bc.target or '❌ MISSING'}")
    print(f"✓ Objective: {bc.objective or '(not set)'}")
    print(f"✓ Primary Metric: {bc.primary_metric or '(not set)'}")
    
    if bc.clarifying_questions:
        print(f"\n🤔 Clarifying Questions ({len(bc.clarifying_questions)}):")
        for i, q in enumerate(bc.clarifying_questions, 1):
            print(f"   {i}. {q}")
    else:
        print("\n❌ NO CLARIFYING QUESTIONS (this is wrong!)")
    
    if bc.dataset_profile:
        print(f"\n✓ Dataset Profile: {bc.dataset_profile.n_rows} rows, {bc.dataset_profile.n_cols} cols")
    else:
        print("\n❌ No dataset profile (agent didn't call perform_eda)")
    
    if bc.key_insights:
        print(f"\n💡 Key Insights ({len(bc.key_insights)}):")
        for insight in bc.key_insights:
            print(f"   - {insight}")
    
    print("\n" + "=" * 70)
    print("FULL JSON OUTPUT:")
    print("=" * 70)
    print(json.dumps(bc.model_dump(), indent=2, ensure_ascii=False))
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

