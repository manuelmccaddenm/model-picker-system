#!/usr/bin/env python3
"""Quick test to verify imports work"""
import sys
print("Python version:", sys.version)
print("Python executable:", sys.executable)

try:
    print("\n1. Testing agents SDK import...")
    import agents
    print("✓ agents imported successfully")
    print("  agents location:", agents.__file__)
    print("  agents dir:", [x for x in dir(agents) if not x.startswith('_')][:10])
except Exception as e:
    print(f"✗ Failed to import agents: {e}")
    sys.exit(1)

try:
    print("\n2. Testing src.agents import...")
    from src.agents import build_data_agent, build_model_agent, build_eval_agent
    print("✓ src.agents imported successfully")
except Exception as e:
    print(f"✗ Failed to import src.agents: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n3. Testing agent construction...")
    data_agent = build_data_agent()
    print("✓ Data agent built successfully")
    print("  Data agent tools:", [t for t in data_agent.tools])
except Exception as e:
    print(f"✗ Failed to build data agent: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")

