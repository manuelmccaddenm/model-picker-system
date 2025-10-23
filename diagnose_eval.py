#!/usr/bin/env python
"""Comprehensive diagnostic for Eval Agent issues"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*60)
print("EVAL AGENT DIAGNOSTIC")
print("="*60)

# Test 1: Check if metrics work with multi-class
print("\n1. Testing metrics with multi-class data...")
try:
    from src.tools.metrics import primary_metric
    
    # Simulate multi-class (3 classes)
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_prob = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.7, 0.2],
        [0.1, 0.2, 0.7],
        [0.9, 0.05, 0.05],
        [0.2, 0.6, 0.2],
        [0.1, 0.1, 0.8],
    ])
    
    score = primary_metric("classification", y_true, None, y_prob, name="roc_auc")
    print(f"   ✓ Multi-class ROC AUC: {score:.3f}")
    
    # Test binary
    y_true_bin = np.array([0, 1, 0, 1])
    y_prob_bin = np.array([
        [0.8, 0.2],
        [0.3, 0.7],
        [0.9, 0.1],
        [0.2, 0.8],
    ])
    
    score_bin = primary_metric("classification", y_true_bin, None, y_prob_bin, name="roc_auc")
    print(f"   ✓ Binary ROC AUC: {score_bin:.3f}")
    
except Exception as e:
    print(f"   ✗ Metrics test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Check if build_candidates works
print("\n2. Testing build_candidates...")
try:
    from src.tools.models import build_candidates
    
    df = pd.read_csv('data/iris.csv')
    cands = build_candidates(df, target='variety', task='classification')
    print(f"   ✓ Built {len(cands)} candidates")
    for c in cands:
        print(f"      - {c.name}")
except Exception as e:
    print(f"   ✗ build_candidates failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Check if a single model can be trained
print("\n3. Testing single model training...")
try:
    from sklearn.model_selection import train_test_split
    
    df = pd.read_csv('data/iris.csv')
    X = df.drop(columns=['variety'])
    y = df['variety']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    cands = build_candidates(df, target='variety', task='classification')
    model = cands[0].model
    
    print(f"   Training {cands[0].name}...")
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)
    print(f"   ✓ Training succeeded!")
    print(f"   ✓ Predictions shape: {prob.shape}")
    
    # Test metric
    score = primary_metric("classification", y_test, None, prob, name="roc_auc")
    print(f"   ✓ ROC AUC: {score:.3f}")
    
except Exception as e:
    print(f"   ✗ Training test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Check full evaluator (with timeout protection)
print("\n4. Testing full evaluator (may take 10-30 seconds)...")
try:
    from src.tools.evaluator import evaluate_with_j_hpo_assumptions
    from src.tools.memory import load_memory
    
    df = pd.read_csv('data/iris.csv')
    mem = load_memory(Path('memory/memory.json'))
    plan = {
        'task': 'classification',
        'target': 'variety',
        'weights': {
            'precision_predictiva': 0.6,
            'interpretabilidad': 0.3,
            'costo_computacional': 0.1,
        },
    }
    cands = build_candidates(df, target='variety', task='classification')
    
    print(f"   Running evaluation with {len(cands)} candidates...")
    results = evaluate_with_j_hpo_assumptions(df, plan, cands, mem, hpo_budget=1)  # Small budget
    
    print(f"   ✓ Evaluation succeeded!")
    print(f"   ✓ Winner: {results['winner']['name']} with J={results['winner']['J']:.3f}")
    print(f"   ✓ Leaderboard:")
    for r in results['leaderboard']:
        print(f"      - {r['name']}: J={r['J']:.3f}")
    
except Exception as e:
    print(f"   ✗ Evaluator test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)

