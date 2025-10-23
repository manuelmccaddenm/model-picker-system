#!/usr/bin/env python3
"""Comprehensive system verification before running"""
import sys
from pathlib import Path

print("=" * 70)
print("SYSTEM VERIFICATION")
print("=" * 70)

errors = []
warnings = []

# 1. Check Python version
print("\n1. Python Version")
if sys.version_info < (3, 8):
    errors.append("Python 3.8+ required")
    print(f"   ✗ Python {sys.version_info.major}.{sys.version_info.minor} (need 3.8+)")
else:
    print(f"   ✓ Python {sys.version_info.major}.{sys.version_info.minor}")

# 2. Check required packages
print("\n2. Required Packages")
required_packages = [
    "agents",
    "openai",
    "pandas",
    "numpy",
    "sklearn",
    "dotenv",
]

for pkg in required_packages:
    try:
        __import__(pkg if pkg != "dotenv" else "dotenv")
        print(f"   ✓ {pkg}")
    except ImportError:
        errors.append(f"Missing package: {pkg}")
        print(f"   ✗ {pkg} (not installed)")

# 3. Check file structure
print("\n3. File Structure")
required_files = [
    "app.py",
    "src/agents.py",
    "src/config.py",
    "src/tools/agent_tools.py",
    "src/tools/memory.py",
    "src/tools/evaluator.py",
    "src/tools/models.py",
    "src/tools/metrics.py",
    "data/sample.csv",
    "memory/memory.json",
]

for file_path in required_files:
    if Path(file_path).exists():
        print(f"   ✓ {file_path}")
    else:
        errors.append(f"Missing file: {file_path}")
        print(f"   ✗ {file_path}")

# 4. Check imports
print("\n4. Import Verification")
try:
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
    print("   ✓ src.agents imports")
except ImportError as e:
    errors.append(f"Import error: {e}")
    print(f"   ✗ src.agents imports: {e}")

try:
    from src.tools.agent_tools import (
        perform_eda,
        memory_get_theory,
        memory_get_experiences,
        evaluate_models_j,
    )
    print("   ✓ src.tools.agent_tools imports")
except ImportError as e:
    errors.append(f"Import error: {e}")
    print(f"   ✗ src.tools.agent_tools imports: {e}")

try:
    from src.config import LOGS_DIR, MEMORY_DIR, OPENAI_MODEL
    print("   ✓ src.config imports")
except ImportError as e:
    errors.append(f"Import error: {e}")
    print(f"   ✗ src.config imports: {e}")

# 5. Check agent builders
print("\n5. Agent Builders")
try:
    from src.agents import build_data_agent, build_model_agent, build_eval_agent_rapid, build_eval_agent_slow
    
    data_agent = build_data_agent()
    print(f"   ✓ Data Agent: {data_agent.name}, {len(data_agent.tools)} tools")
    
    model_agent = build_model_agent()
    print(f"   ✓ Model Agent: {model_agent.name}, {len(model_agent.tools)} tools")
    
    eval_agent_rapid = build_eval_agent_rapid()
    print(f"   ✓ Eval Agent (Rapid): {eval_agent_rapid.name}, {len(eval_agent_rapid.tools)} tools")
    
    eval_agent_slow = build_eval_agent_slow()
    print(f"   ✓ Eval Agent (Slow): {eval_agent_slow.name}, {len(eval_agent_slow.tools)} tools")
    
except Exception as e:
    errors.append(f"Agent builder error: {e}")
    print(f"   ✗ Agent builders: {e}")

# 6. Check environment
print("\n6. Environment")
import os
from dotenv import load_dotenv
load_dotenv()

if os.getenv("OPENAI_API_KEY"):
    print("   ✓ OPENAI_API_KEY set")
else:
    warnings.append("OPENAI_API_KEY not set in .env")
    print("   ⚠ OPENAI_API_KEY not set (required for agents to work)")

# 7. Check directories
print("\n7. Directories")
from src.config import LOGS_DIR, MEMORY_DIR

for dir_path in [LOGS_DIR, MEMORY_DIR]:
    if dir_path.exists():
        print(f"   ✓ {dir_path}")
    else:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"   ✓ {dir_path} (created)")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

if errors:
    print(f"\n❌ {len(errors)} ERROR(S) FOUND:")
    for err in errors:
        print(f"   - {err}")
    print("\nPlease fix these errors before running the system.")
    sys.exit(1)

if warnings:
    print(f"\n⚠️  {len(warnings)} WARNING(S):")
    for warn in warnings:
        print(f"   - {warn}")

if not errors:
    print("\n✅ ALL CHECKS PASSED!")
    print("\nYou can now run:")
    print("  python app.py data/sample.csv")
    print("  python test_eval_agent.py")
    print("  python test_minimal_context.py")
    sys.exit(0)

