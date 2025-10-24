"""
Knowledge Graph Dataset Example

Demonstrates loading and using the KnowledgeGraphTripleDataset
for relational reasoning tasks.
"""

import torch
from pathlib import Path

from nsm.data.knowledge_graph_dataset import KnowledgeGraphTripleDataset
from nsm.data.graph import visualize_graph_structure


def main():
    """Run Knowledge Graph dataset examples."""
    print("=" * 80)
    print("Knowledge Graph Dataset Example")
    print("=" * 80)

    # Create dataset
    print("\n1. Creating Knowledge Graph Dataset...")
    dataset = KnowledgeGraphTripleDataset(
        root="data/kg_example",
        split='train',
        num_entities=1000,
        num_triples=5000,
        seed=42
    )
    print(f"   Dataset created: {dataset}")

    # Display statistics
    print("\n2. Dataset Statistics:")
    stats = dataset.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Show sample triples
    print("\n3. Sample Triples (First 10):")
    print("   " + "-" * 76)
    for i, triple in enumerate(dataset.triples[:10]):
        print(f"   {i+1}. ({triple.subject}, {triple.predicate}, {triple.object})")
        print(f"      Confidence: {triple.confidence:.3f}, Level: {triple.level}")

    # Show Level 1 (facts) examples
    print("\n4. Level 1 Triples (Facts/Instances):")
    print("   " + "-" * 76)
    level1_triples = [t for t in dataset.triples if t.level == 1][:10]
    for i, triple in enumerate(level1_triples):
        print(f"   {i+1}. {triple.subject} --[{triple.predicate}]--> {triple.object}")
        print(f"      Confidence: {triple.confidence:.3f}")

    # Show Level 2 (types) examples
    print("\n5. Level 2 Triples (Types/Categories):")
    print("   " + "-" * 76)
    level2_triples = [t for t in dataset.triples if t.level == 2][:10]
    for i, triple in enumerate(level2_triples):
        print(f"   {i+1}. {triple.subject} --[{triple.predicate}]--> {triple.object}")
        print(f"      Confidence: {triple.confidence:.3f}")

    # Show predicate diversity
    print("\n6. Predicate Types:")
    predicates = {}
    for triple in dataset.triples:
        pred = triple.predicate
        level = triple.level
        key = f"L{level}: {pred}"
        predicates[key] = predicates.get(key, 0) + 1

    # Show top 20 predicates
    sorted_preds = sorted(predicates.items(), key=lambda x: x[1], reverse=True)[:20]
    for pred, count in sorted_preds:
        print(f"   {pred}: {count} triples")

    # Entity categories
    print("\n7. Entity Categories:")
    print(f"   People: {len(dataset.people)}")
    print(f"   Places: {len(dataset.places)}")
    print(f"   Organizations: {len(dataset.organizations)}")
    print(f"   Concepts: {len(dataset.concepts)}")
    print(f"   Awards: {len(dataset.awards)}")
    print(f"   Total unique entities: {len(dataset.entities)}")

    # Show some famous entities
    print("\n8. Sample Named Entities:")
    all_entities = set()
    for triple in dataset.triples:
        all_entities.add(triple.subject)
        all_entities.add(triple.object)

    famous_people = [p for p in dataset.PERSON_NAMES if p in all_entities][:5]
    famous_places = [p for p in dataset.PLACES if p in all_entities][:5]

    print(f"   People: {', '.join(famous_people)}")
    print(f"   Places: {', '.join(famous_places)}")

    # Get and visualize a graph
    print("\n9. PyTorch Geometric Graph (Sample):")
    graph, label = dataset[0]
    print(visualize_graph_structure(graph))
    print(f"   Label (confidence): {label.item():.3f}")

    # Multi-hop reasoning queries
    print("\n10. Multi-hop Reasoning Queries (Sample):")
    print("    " + "-" * 74)
    queries = dataset.get_multi_hop_queries(num_queries=5)
    for i, query in enumerate(queries, 1):
        print(f"    Query {i}:")
        print(f"      Start: {query['start_entity']}")
        print(f"      Path: {' -> '.join(query['relations'])}")
        if 'intermediate' in query:
            print(f"      Via: {query['intermediate']}")
        print(f"      Answer: {query['expected_answer']}")
        print()

    # Type consistency checking
    print("\n11. Type Consistency Pairs (Sample):")
    print("    " + "-" * 74)
    pairs = dataset.get_type_consistency_pairs(num_pairs=10)
    for i, (entity, entity_type, is_consistent) in enumerate(pairs, 1):
        status = "✓ VALID" if is_consistent else "✗ INVALID"
        print(f"    {i}. {entity} : {entity_type} -> {status}")

    # Show biographical chain example
    print("\n12. Example Biographical Reasoning Chain:")
    print("    " + "-" * 74)
    # Find a person with multiple relations
    person_triples = {}
    for triple in dataset.triples:
        if triple.subject in dataset.people:
            if triple.subject not in person_triples:
                person_triples[triple.subject] = []
            person_triples[triple.subject].append(triple)

    # Find person with rich biography
    for person, triples in list(person_triples.items())[:5]:
        if len(triples) >= 3:
            print(f"    Person: {person}")
            for triple in triples[:5]:
                print(f"      {triple.predicate} -> {triple.object} (conf: {triple.confidence:.3f})")
            break

    # Type hierarchy example
    print("\n13. Type Hierarchy:")
    print("    " + "-" * 74)
    hierarchy = dataset.type_hierarchy
    for child, parent in list(hierarchy.items())[:10]:
        print(f"    {child} --[subclass_of]--> {parent}")

    # Confidence distribution
    print("\n14. Confidence Score Distribution:")
    confidences = [t.confidence for t in dataset.triples]
    ranges = [
        (0.5, 0.6, "0.5-0.6"),
        (0.6, 0.7, "0.6-0.7"),
        (0.7, 0.8, "0.7-0.8"),
        (0.8, 0.9, "0.8-0.9"),
        (0.9, 1.0, "0.9-1.0"),
    ]

    for low, high, label in ranges:
        count = sum(1 for c in confidences if low <= c <= high)
        pct = 100 * count / len(confidences)
        bar = "█" * int(pct / 2)
        print(f"    {label}: {bar} {pct:.1f}% ({count} triples)")

    # PyG Data Loader example
    print("\n15. PyTorch Geometric DataLoader Example:")
    from torch_geometric.loader import DataLoader

    # Create small batch
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    print(f"    Batch of {batch.num_graphs} graphs:")
    print(f"      Total nodes: {batch.num_nodes}")
    print(f"      Total edges: {batch.edge_index.size(1)}")
    print(f"      Node features: {batch.x.shape}")
    print(f"      Edge features: {batch.edge_attr.shape}")

    # Show specific reasoning patterns
    print("\n16. Specific Reasoning Patterns:")
    print("    " + "-" * 74)

    # Find instance-of relations
    instance_of = [t for t in dataset.triples if t.predicate == "instance_of"][:5]
    print("    a) Instance-of Relations:")
    for triple in instance_of:
        print(f"       {triple.subject} is a {triple.object}")

    # Find born_in + located_in chains
    print("\n    b) Geographic Chains:")
    born_in = {t.subject: t.object for t in dataset.triples if t.predicate == "born_in"}
    located_in = {t.subject: t.object for t in dataset.triples if t.predicate == "located_in"}

    count = 0
    for person, city in list(born_in.items())[:20]:
        if city in located_in:
            country = located_in[city]
            print(f"       {person} born in {city}, which is in {country}")
            count += 1
            if count >= 3:
                break

    # Find work relations
    print("\n    c) Professional Relations:")
    works_at = [t for t in dataset.triples if t.predicate == "works_at"][:5]
    for triple in works_at:
        print(f"       {triple.subject} works at {triple.object}")

    print("\n" + "=" * 80)
    print("Example complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
