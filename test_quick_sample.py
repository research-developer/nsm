#!/usr/bin/env python3
"""Quick test to verify diversity across different problem indices."""

import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nsm.data.planning_dataset import PlanningTripleDataset

with tempfile.TemporaryDirectory() as tmpdir:
    dataset = PlanningTripleDataset(
        root=tmpdir,
        split='train',
        num_problems=24000,
        problems_per_split=True,
        seed=42
    )

    print("Sampling problems with different tier expectations:")
    print("-" * 70)

    # Test tier 0 (idx % 100 < 40)
    print("\nTier 0 examples (idx % 100 < 40):")
    for idx in [0, 1, 39, 100, 139]:
        triples = dataset.get_problem_triples(idx)
        tier = triples[0].metadata.get('tier', -1) if triples else -1

        # Count actions
        actions = [t for t in triples if t.metadata.get('type') == 'action']
        print(f"  Problem {idx:5d}: tier={tier}, {len(actions):2d} actions, "
              f"{len(triples):3d} total triples")

    # Test tier 1 (40 <= idx % 100 < 80)
    print("\nTier 1 examples (40 <= idx % 100 < 80):")
    for idx in [40, 41, 79, 140, 179]:
        triples = dataset.get_problem_triples(idx)
        tier = triples[0].metadata.get('tier', -1) if triples else -1

        # Count actions
        actions = [t for t in triples if t.metadata.get('type') == 'action']
        print(f"  Problem {idx:5d}: tier={tier}, {len(actions):2d} actions, "
              f"{len(triples):3d} total triples")

    # Test tier 2 (80 <= idx % 100 < 100)
    print("\nTier 2 examples (80 <= idx % 100 < 100):")
    for idx in [80, 81, 99, 180, 199]:
        triples = dataset.get_problem_triples(idx)
        tier = triples[0].metadata.get('tier', -1) if triples else -1

        # Count actions
        actions = [t for t in triples if t.metadata.get('type') == 'action']
        print(f"  Problem {idx:5d}: tier={tier}, {len(actions):2d} actions, "
              f"{len(triples):3d} total triples")

    print("\n" + "-" * 70)
    print("✓ Test complete")
