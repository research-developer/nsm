"""
Planning Dataset Example

Demonstrates usage of PlanningTripleDataset for hierarchical
goal decomposition and action sequence reasoning.

This example shows:
1. Loading the planning dataset
2. Visualizing planning problem structure
3. Analyzing hierarchical decomposition
4. Inspecting temporal ordering constraints
5. Converting to PyG graphs for GNN processing
"""

import torch
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

from nsm.data.planning_dataset import PlanningTripleDataset
from nsm.data.graph import visualize_graph_structure


def main():
    """Run planning dataset demonstration."""
    print("=" * 70)
    print("Planning Domain Dataset - NSM Phase 1 Exploration")
    print("=" * 70)
    print()

    # 1. Create dataset
    print("1. Creating Planning Dataset...")
    print("-" * 70)

    dataset = PlanningTripleDataset(
        root="data/planning_demo",
        split='train',
        num_problems=20,
        num_locations=5,
        num_objects=8,
        seed=42
    )

    print(f"   Dataset: {dataset}")
    print(f"   Total triples: {len(dataset.triples)}")
    print(f"   Total problems: {len(dataset.problems)}")
    print()

    # 2. Display dataset statistics
    print("2. Dataset Statistics")
    print("-" * 70)

    stats = dataset.get_statistics()
    print(f"   Unique entities: {stats['num_entities']}")
    print(f"   Unique predicates: {stats['num_predicates']}")
    print(f"   Average confidence: {stats['avg_confidence']:.3f}")
    print(f"   Level distribution:")
    for level, count in sorted(stats['level_distribution'].items()):
        level_name = "Concrete (Actions/Environment)" if level == 1 else "Abstract (Goals/Capabilities)"
        print(f"     Level {level} ({level_name}): {count} triples")
    print()

    # 3. Analyze a specific planning problem
    print("3. Analyzing Planning Problem #0")
    print("-" * 70)

    problem_idx = 0
    problem_triples = dataset.get_problem_triples(problem_idx)
    print(f"   Problem has {len(problem_triples)} triples")
    print()

    # 3a. Show hierarchical structure
    print("   3a. Hierarchical Structure:")
    structure = dataset.analyze_hierarchical_structure(problem_idx)
    print(f"      Goals: {structure['num_goals']}")
    print(f"      Capabilities: {structure['num_capabilities']}")
    print(f"      Decompositions: {structure['num_decompositions']}")
    print(f"      Max decomposition depth: {structure['decomposition_depth']}")
    print()

    # 3b. Show temporal ordering
    print("   3b. Temporal Ordering:")
    ordering = dataset.analyze_temporal_ordering(problem_idx)
    print(f"      Actions: {ordering['num_actions']}")
    print(f"      Dependencies: {ordering['num_dependencies']}")
    print(f"      Valid ordering: {ordering['is_valid']}")
    print(f"      Dependency ratio: {ordering['dependency_ratio']:.2f}")
    print()

    # 4. Display sample triples by type
    print("4. Sample Triples by Category")
    print("-" * 70)

    # Categorize triples
    triples_by_type = {}
    for triple in problem_triples:
        triple_type = triple.metadata.get('type', 'unknown')
        if triple_type not in triples_by_type:
            triples_by_type[triple_type] = []
        triples_by_type[triple_type].append(triple)

    for triple_type, triples in sorted(triples_by_type.items()):
        print(f"\n   {triple_type.upper()} ({len(triples)} triples):")
        for i, triple in enumerate(triples[:3]):  # Show first 3 of each type
            level_name = "L1" if triple.level == 1 else "L2"
            print(f"      [{level_name}] ({triple.subject}, {triple.predicate}, {triple.object})")
            print(f"           confidence={triple.confidence:.3f}")
        if len(triples) > 3:
            print(f"      ... and {len(triples) - 3} more")
    print()

    # 5. Visualize goal decomposition
    print("5. Goal Decomposition Chain")
    print("-" * 70)

    # Find goal → capability → action chains
    goals = [t for t in problem_triples if t.predicate == 'achieve']
    if goals:
        goal_triple = goals[0]
        goal_name = goal_triple.object
        print(f"   Goal: {goal_name}")
        print(f"   Achieved by: {goal_triple.subject}")
        print()

        # Find what the goal requires
        requirements = [t for t in problem_triples
                       if t.subject == goal_name and t.predicate == 'requires']

        if requirements:
            print(f"   Requires:")
            for req in requirements[:5]:  # Show first 5
                req_type = "capability" if 'cap_' in req.object else "action"
                print(f"      - {req.object} ({req_type}, conf={req.confidence:.3f})")

                # If it's a capability, show what it enables
                if req_type == "capability":
                    enabled = [t for t in problem_triples
                             if t.subject == req.object and t.predicate == 'enables']
                    if enabled:
                        for e in enabled[:2]:
                            print(f"          → enables: {e.object}")
    print()

    # 6. Convert to PyG graph
    print("6. Graph Representation")
    print("-" * 70)

    graph = dataset.get_problem_graph(problem_idx)
    print(visualize_graph_structure(graph))
    print()

    # Show edge type distribution
    edge_types = graph.edge_type
    unique_types, counts = torch.unique(edge_types, return_counts=True)
    print(f"   Edge type distribution:")
    for edge_type_idx, count in zip(unique_types.tolist(), counts.tolist()):
        predicate = dataset.vocabulary.get_predicate_token(edge_type_idx)
        print(f"      {predicate}: {count} edges")
    print()

    # 7. Sample a training batch
    print("7. Training Batch Example")
    print("-" * 70)

    # Get a few samples
    batch_size = 3
    print(f"   Loading {batch_size} samples...")

    for idx in range(min(batch_size, len(dataset))):
        graph, label = dataset[idx]
        print(f"   Sample {idx}:")
        print(f"      Nodes: {graph.num_nodes}, Edges: {graph.edge_index.size(1)}")
        print(f"      Label: {label.item()} ({'valid' if label.item() == 1 else 'invalid'} sequence)")
    print()

    # 8. Visualize graph structure (if matplotlib available)
    print("8. Graph Visualization")
    print("-" * 70)

    try:
        # Create NetworkX graph for visualization
        nx_graph = to_networkx(
            graph,
            to_undirected=False,
            node_attrs=['node_level'],
            edge_attrs=['edge_type']
        )

        print(f"   Created NetworkX graph with {nx_graph.number_of_nodes()} nodes")
        print(f"   and {nx_graph.number_of_edges()} edges")

        # Optionally save visualization
        save_viz = False  # Set to True to save
        if save_viz:
            plt.figure(figsize=(12, 8))

            # Color nodes by level
            node_colors = []
            for node in nx_graph.nodes():
                level = nx_graph.nodes[node].get('node_level', 1)
                node_colors.append('lightblue' if level == 1 else 'lightcoral')

            # Draw graph
            pos = nx.spring_layout(nx_graph, k=0.5, iterations=50)
            nx.draw(
                nx_graph,
                pos,
                node_color=node_colors,
                node_size=500,
                with_labels=False,
                arrows=True,
                edge_color='gray',
                alpha=0.7
            )

            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='lightblue', label='Level 1 (Concrete)'),
                Patch(facecolor='lightcoral', label='Level 2 (Abstract)')
            ]
            plt.legend(handles=legend_elements, loc='upper right')
            plt.title(f"Planning Problem #{problem_idx} - Graph Structure")
            plt.tight_layout()

            output_path = Path("data/planning_demo/problem_graph.png")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"   Saved visualization to: {output_path}")
        else:
            print("   (Set save_viz=True to save visualization)")

    except Exception as e:
        print(f"   Visualization skipped: {e}")
    print()

    # 9. Summary
    print("9. Summary")
    print("-" * 70)
    print(f"   ✓ Generated {len(dataset.problems)} planning problems")
    print(f"   ✓ Created {len(dataset.triples)} hierarchical triples")
    print(f"   ✓ Level 1 (concrete): Actions and environmental states")
    print(f"   ✓ Level 2 (abstract): Goals and capability requirements")
    print(f"   ✓ Confidence range: {min(t.confidence for t in dataset.triples):.3f} - {max(t.confidence for t in dataset.triples):.3f}")
    print(f"   ✓ Compatible with PyG for GNN processing")
    print()

    print("=" * 70)
    print("Planning Dataset Example Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
