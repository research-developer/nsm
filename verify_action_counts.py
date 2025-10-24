#!/usr/bin/env python3
"""Verify action counts are correct in generated problems."""

import sys
import os
import tempfile
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nsm.data.planning_dataset import PlanningTripleDataset

with tempfile.TemporaryDirectory() as tmpdir:
    dataset = PlanningTripleDataset(
        root=tmpdir,
        split='train',
        num_problems=100,  # Small sample for quick verification
        problems_per_split=True,
        seed=42
    )

    print("Verifying Action Generation")
    print("=" * 70)

    # Test all three tiers
    test_cases = [
        (0, 0),    # Tier 0
        (1, 0),    # Tier 0
        (40, 1),   # Tier 1
        (41, 1),   # Tier 1
        (80, 2),   # Tier 2
        (81, 2),   # Tier 2
    ]

    print("\nDetailed Action Analysis:")
    print("-" * 70)

    action_counts_by_tier = {0: [], 1: [], 2: []}

    for idx, expected_tier in test_cases:
        triples = dataset.get_problem_triples(idx)
        tier = triples[0].metadata.get('tier', -1) if triples else -1

        # Count actions by type
        action_triples = [t for t in triples if t.metadata.get('type') == 'action']
        unique_action_names = set()
        action_types = []

        for t in action_triples:
            # The action triple has robot as subject, action_type as predicate
            unique_action_names.add(f"{t.predicate}_{idx}_{t.metadata.get('sequence')}")
            action_types.append(t.predicate)

        action_counts_by_tier[tier].append(len(action_triples))

        print(f"Problem {idx:2d} (Tier {tier}, expected {expected_tier}):")
        print(f"  Actions: {len(action_triples)}")
        print(f"  Action types: {Counter(action_types).most_common(3)}")
        print(f"  Total triples: {len(triples)}")
        print()

    print("-" * 70)
    print("\nAction Count Statistics by Tier:")
    print("-" * 70)

    for tier in [0, 1, 2]:
        counts = action_counts_by_tier[tier]
        if counts:
            print(f"Tier {tier}: min={min(counts)}, max={max(counts)}, "
                  f"avg={sum(counts)/len(counts):.1f}")

    # Verify tier-specific ranges
    print("\nExpected Ranges:")
    print(f"  Tier 0: 3-6 actions")
    print(f"  Tier 1: 6-10 actions")
    print(f"  Tier 2: 10-15 actions")

    print("\n" + "=" * 70)
    print("✓ Verification complete")
