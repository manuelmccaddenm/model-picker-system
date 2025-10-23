#!/usr/bin/env python3
"""Test the Data Agent in isolation"""
import json
from pathlib import Path
from agents import Runner
from src.agents import build_data_agent, BusinessContext

dataset_path = str(Path("data/sample.csv").resolve())

data_agent = build_data_agent()

msg = (
    f"You are analyzing a dataset for a machine learning project.\n\n"
    f"DATASET_CSV_PATH: {dataset_path}\n"
    f"INSTRUCTIONS: I want to predict customer churn from this dataset\n"
    f"\nREMINDER: You must call perform_eda('{dataset_path}') to profile the data. "
    f"If task or target are unclear, add clarifying_questions."
)

print("=" * 60)
print("Testing Data Agent")
print("=" * 60)
print(f"\nInput message:\n{msg}\n")
print("Running agent...")

try:
    result = Runner.run_sync(data_agent, msg)
    bc = result.final_output_as(BusinessContext)
    
    print("\n" + "=" * 60)
    print("Agent Response:")
    print("=" * 60)
    print(json.dumps(bc.model_dump(), indent=2, ensure_ascii=False))
    
    print("\n" + "=" * 60)
    print("Analysis:")
    print("=" * 60)
    print(f"Task: {bc.task or 'MISSING'}")
    print(f"Target: {bc.target or 'MISSING'}")
    print(f"Clarifying Questions: {bc.clarifying_questions or 'NONE'}")
    print(f"Dataset Profile: {'Present' if bc.dataset_profile else 'MISSING'}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

