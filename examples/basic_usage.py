"""
Basic Usage Example for NSM Data Structures

Demonstrates:
1. Creating semantic triples
2. Building vocabularies
3. Constructing graphs
4. Using datasets
"""

import torch
from nsm.data import (
    SemanticTriple,
    TripleCollection,
    TripleVocabulary,
    GraphConstructor,
    SyntheticTripleDataset,
    visualize_graph_structure
)


def example_1_semantic_triples():
    """Example 1: Creating and manipulating semantic triples."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Semantic Triples")
    print("=" * 60)

    # Create concrete-level triples (Level 1: Actions/Environment)
    t1 = SemanticTriple(
        subject="robot",
        predicate="executes",
        object="move_left",
        confidence=0.95,
        level=1,
        metadata={"source": "planner", "step": 1}
    )

    t2 = SemanticTriple(
        subject="robot",
        predicate="executes",
        object="pick_up_object",
        confidence=0.90,
        level=1
    )

    # Create abstract-level triples (Level 2: Goals/Capabilities)
    t3 = SemanticTriple(
        subject="system",
        predicate="achieves",
        object="navigate_to_goal",
        confidence=0.85,
        level=2
    )

    print(f"\nConcrete triple: {t1}")
    print(f"Abstract triple: {t3}")
    print(f"\nIs t1 concrete? {t1.is_concrete()}")
    print(f"Is t3 abstract? {t3.is_abstract()}")

    # Work with collections
    collection = TripleCollection([t1, t2, t3])
    print(f"\nCollection: {collection}")
    print(f"Unique entities: {collection.get_unique_entities()}")
    print(f"Unique predicates: {collection.get_unique_predicates()}")

    # Filter by level
    concrete_triples = collection.filter_by_level(1)
    print(f"\nConcrete triples (Level 1): {len(concrete_triples)} triples")

    # Filter by confidence
    high_conf = collection.filter_by_confidence(min_conf=0.9)
    print(f"High confidence triples (>=0.9): {len(high_conf)} triples")


def example_2_vocabularies():
    """Example 2: Building and using vocabularies."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Vocabularies")
    print("=" * 60)

    # Create vocabulary for semantic triples
    vocab = TripleVocabulary()

    # Add triples
    triples = [
        ("robot", "executes", "move"),
        ("robot", "executes", "turn"),
        ("move", "enables", "navigation"),
        ("turn", "enables", "navigation"),
    ]

    print("\nAdding triples to vocabulary...")
    for subj, pred, obj in triples:
        subj_idx, pred_idx, obj_idx = vocab.add_triple_entities(subj, pred, obj)
        print(f"  {subj}({subj_idx}) --{pred}({pred_idx})--> {obj}({obj_idx})")

    print(f"\nVocabulary stats: {vocab}")
    print(f"  Entities: {vocab.num_entities}")
    print(f"  Predicates: {vocab.num_predicates}")

    # Lookup entities
    print("\nLookup examples:")
    print(f"  Index of 'robot': {vocab.get_entity_index('robot')}")
    print(f"  Index of 'executes': {vocab.get_predicate_index('executes')}")
    print(f"  Token at entity index 0: {vocab.get_entity_token(0)}")


def example_3_graph_construction():
    """Example 3: Constructing PyG graphs from triples."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Graph Construction")
    print("=" * 60)

    # Create sample triples representing a planning task
    triples = [
        SemanticTriple("robot", "at_location", "room_A", confidence=1.0, level=1),
        SemanticTriple("robot", "executes", "move_to_B", confidence=0.9, level=1),
        SemanticTriple("move_to_B", "changes", "location", confidence=0.95, level=1),
        SemanticTriple("robot", "achieves", "goal_B", confidence=0.8, level=2),
    ]

    print(f"\nConstructing graph from {len(triples)} triples...")

    # Initialize graph constructor
    constructor = GraphConstructor(node_feature_dim=64)

    # Construct graph
    graph = constructor.construct(triples)

    print(f"\n{visualize_graph_structure(graph)}")

    # Access graph components
    print(f"\nGraph components:")
    print(f"  Node features shape: {graph.x.shape}")
    print(f"  Edge index shape: {graph.edge_index.shape}")
    print(f"  Edge attributes shape: {graph.edge_attr.shape}")
    print(f"  Edge types: {graph.edge_type}")
    print(f"  Node levels: {graph.node_level}")

    # Show edge connectivity
    print(f"\nEdge connectivity (first 3 edges):")
    for i in range(min(3, graph.edge_index.size(1))):
        src = graph.edge_index[0, i].item()
        dst = graph.edge_index[1, i].item()
        edge_type = graph.edge_type[i].item()
        confidence = graph.edge_attr[i, 0].item()
        print(f"  Edge {i}: node {src} -> node {dst}, type={edge_type}, conf={confidence:.3f}")

    # Construct hierarchical graphs
    print("\nConstructing separate graphs for each level...")
    concrete_graph, abstract_graph = constructor.construct_hierarchical(triples)
    print(f"  Concrete graph: {concrete_graph.num_nodes} nodes, {concrete_graph.edge_index.size(1)} edges")
    print(f"  Abstract graph: {abstract_graph.num_nodes} nodes, {abstract_graph.edge_index.size(1)} edges")


def example_4_datasets():
    """Example 4: Using datasets."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Datasets")
    print("=" * 60)

    # Create synthetic dataset
    print("\nCreating synthetic dataset...")
    dataset = SyntheticTripleDataset(
        root="data/synthetic_example",
        split="train",
        num_entities=20,
        num_predicates=5,
        num_triples=50,
        num_levels=2
    )

    print(f"Dataset: {dataset}")
    print(f"Number of samples: {len(dataset)}")

    # Get statistics
    stats = dataset.get_statistics()
    print(f"\nDataset statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Get a sample
    print(f"\nFetching sample 0...")
    graph, label = dataset[0]
    print(f"  Graph: {graph.num_nodes} nodes, {graph.edge_index.size(1)} edges")
    print(f"  Label: {label}")

    # Save vocabulary
    print(f"\nSaving vocabulary...")
    dataset.save_vocabulary()
    print(f"  Saved to: {dataset.processed_dir}")


def example_5_complete_workflow():
    """Example 5: Complete workflow from triples to model input."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Complete Workflow")
    print("=" * 60)

    # Step 1: Define domain knowledge as triples
    print("\nStep 1: Define semantic triples...")
    triples = [
        # Concrete level: Robot actions
        SemanticTriple("robot", "has_capability", "move", level=1),
        SemanticTriple("robot", "has_capability", "grasp", level=1),
        SemanticTriple("robot", "executes", "move", confidence=0.95, level=1),
        SemanticTriple("move", "precedes", "grasp", confidence=0.9, level=1),
        SemanticTriple("grasp", "requires", "move", confidence=0.85, level=1),

        # Abstract level: Goals and capabilities
        SemanticTriple("system", "has_goal", "retrieve_object", level=2),
        SemanticTriple("retrieve_object", "requires", "move", confidence=0.8, level=2),
        SemanticTriple("retrieve_object", "requires", "grasp", confidence=0.9, level=2),
    ]
    print(f"  Created {len(triples)} triples")

    # Step 2: Build vocabulary
    print("\nStep 2: Build vocabulary...")
    vocab = TripleVocabulary()
    for triple in triples:
        vocab.add_triple_entities(triple.subject, triple.predicate, triple.object)
    print(f"  {vocab}")

    # Step 3: Construct graph
    print("\nStep 3: Construct PyG graph...")
    constructor = GraphConstructor(vocabulary=vocab, node_feature_dim=32)
    graph = constructor.construct(triples)
    print(f"{visualize_graph_structure(graph)}")

    # Step 4: Prepare for model input
    print("\nStep 4: Prepare batch for model...")
    # In practice, you'd use DataLoader, but here's the concept
    batch_graphs = constructor.batch_construct([triples[:4], triples[4:]])
    print(f"  Created {len(batch_graphs)} graphs in batch")
    for i, g in enumerate(batch_graphs):
        print(f"    Graph {i}: {g.num_nodes} nodes, {g.edge_index.size(1)} edges")

    # Step 5: Ready for R-GCN processing
    print("\nStep 5: Ready for R-GCN!")
    print("  The graphs can now be processed by:")
    print("  - R-GCN layers (NSM-17)")
    print("  - WHY/WHAT operations (NSM-16)")
    print("  - Confidence propagation (NSM-15)")

    print("\n" + "=" * 60)
    print("See CLAUDE.md for next steps:")
    print("  - NSM-17: R-GCN Message Passing")
    print("  - NSM-16: Symmetric WHY/WHAT Operations")
    print("  - NSM-15: Confidence Propagation")
    print("=" * 60)


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("NSM DATA STRUCTURES - BASIC USAGE EXAMPLES")
    print("=" * 60)

    example_1_semantic_triples()
    example_2_vocabularies()
    example_3_graph_construction()
    example_4_datasets()
    example_5_complete_workflow()

    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
