"""
Causal Reasoning Dataset Example

Demonstrates:
1. Loading the CausalTripleDataset
2. Exploring causal chains
3. Examining confounders
4. Counterfactual queries
5. Graph construction and visualization
"""

import torch
from pathlib import Path
from nsm.data.causal_dataset import CausalTripleDataset


def print_separator(title: str = ""):
    """Print a formatted separator."""
    if title:
        print(f"\n{'=' * 80}")
        print(f"  {title}")
        print(f"{'=' * 80}\n")
    else:
        print(f"{'-' * 80}\n")


def explore_basic_dataset():
    """Create and explore basic dataset properties."""
    print_separator("1. Creating Causal Reasoning Dataset")

    # Create dataset
    dataset = CausalTripleDataset(
        root="data/causal_example",
        split='train',
        num_scenarios=50,
        num_treatments=8,
        num_symptoms=8,
        num_confounders=6,
        confound_prob=0.4,
        seed=42
    )

    print(f"Dataset: {dataset}")
    print(f"Total triples: {len(dataset.triples)}")
    print(f"Total scenarios: {len(dataset.scenarios)}")

    # Get statistics
    stats = dataset.get_statistics()
    print(f"\nStatistics:")
    print(f"  Unique entities: {stats['num_entities']}")
    print(f"  Unique predicates: {stats['num_predicates']}")
    print(f"  Average confidence: {stats['avg_confidence']:.3f}")
    print(f"  Level distribution: {stats['level_distribution']}")

    return dataset


def show_scenario_structure(dataset: CausalTripleDataset):
    """Show detailed structure of a scenario."""
    print_separator("2. Examining Scenario Structure")

    # Get first scenario
    scenario = dataset.get_scenario(0)

    print(f"Scenario 0:")
    print(f"  Patient: {scenario['patient']}")
    print(f"  Treatment: {scenario['treatment']}")
    print(f"  Symptom: {scenario['symptom']}")
    print(f"  Effect mechanism: {scenario['effect']}")
    print(f"  Confounder: {scenario['confounder']}")

    print(f"\nTriples in this scenario ({len(scenario['triple_indices'])} total):")

    # Show all triples
    for idx in scenario['triple_indices']:
        triple = dataset.triples[idx]
        print(f"  Level {triple.level}: ({triple.subject}, {triple.predicate}, "
              f"{triple.object}) [conf={triple.confidence:.3f}]")
        if triple.metadata:
            print(f"    Metadata: {triple.metadata}")


def show_causal_chain(dataset: CausalTripleDataset):
    """Illustrate a causal chain from treatment to outcome."""
    print_separator("3. Causal Chain: Treatment → Effect → Outcome")

    scenario = dataset.get_scenario(0)
    patient = scenario['patient']
    treatment = scenario['treatment']
    symptom = scenario['symptom']
    effect = scenario['effect']

    # Find relevant triples
    print("LEVEL 1 - OBSERVATIONS:")

    # Intervention
    intervention = [
        t for t in dataset.triples
        if t.subject == treatment and t.predicate == "taken_by"
        and t.object == patient
    ]
    if intervention:
        t = intervention[0]
        print(f"  Intervention: {t.subject} → {t.predicate} → {t.object}")
        print(f"    Confidence: {t.confidence:.3f}")

    # Initial symptom
    initial = [
        t for t in dataset.triples
        if t.subject == patient and t.predicate == "has_symptom"
        and t.object == symptom
    ]
    if initial:
        t = initial[0]
        print(f"  Initial state: {t.subject} → {t.predicate} → {t.object}")
        print(f"    Confidence: {t.confidence:.3f}")

    # Outcome
    outcome = [
        t for t in dataset.triples
        if t.subject == patient and t.object == symptom
        and t.predicate in ["symptom_reduced", "symptom_persists"]
    ]
    if outcome:
        t = outcome[0]
        print(f"  Outcome: {t.subject} → {t.predicate} → {t.object}")
        print(f"    Confidence: {t.confidence:.3f}")

    print("\nLEVEL 2 - CAUSAL MECHANISMS:")

    # Treatment causes effect
    causal = [
        t for t in dataset.triples
        if t.subject == treatment and t.predicate == "causes"
        and t.object == effect and t.level == 2
    ]
    if causal:
        t = causal[0]
        print(f"  Mechanism: {t.subject} → {t.predicate} → {t.object}")
        print(f"    Causal strength: {t.confidence:.3f}")

    # Effect treats symptom
    mechanism = [
        t for t in dataset.triples
        if t.subject == effect and t.predicate == "treats"
        and t.object == symptom and t.level == 2
    ]
    if mechanism:
        t = mechanism[0]
        print(f"  Action: {t.subject} → {t.predicate} → {t.object}")
        print(f"    Confidence: {t.confidence:.3f}")

    # Patient response
    response = [
        t for t in dataset.triples
        if t.subject == patient and t.object == treatment
        and t.predicate in ["responds_to", "resistant_to"] and t.level == 2
    ]
    if response:
        t = response[0]
        print(f"  Individual effect: {t.subject} → {t.predicate} → {t.object}")
        print(f"    Confidence: {t.confidence:.3f}")


def analyze_confounders(dataset: CausalTripleDataset):
    """Analyze confounding relationships."""
    print_separator("4. Analyzing Confounders")

    # Count confounded scenarios
    confounded = [s for s in dataset.scenarios if s['confounder'] is not None]
    print(f"Scenarios with confounders: {len(confounded)} / {len(dataset.scenarios)}")

    if confounded:
        # Show first confounded scenario
        scenario = confounded[0]
        patient = scenario['patient']
        confounder = scenario['confounder']

        print(f"\nExample confounded scenario:")
        print(f"  Patient: {patient}")
        print(f"  Treatment: {scenario['treatment']}")
        print(f"  Confounder: {confounder}")

        # Find confounder triples
        print("\n  Confounder observation (Level 1):")
        obs = [
            t for t in dataset.triples
            if t.subject == patient and t.predicate == "has_condition"
            and t.object == confounder
        ]
        if obs:
            t = obs[0]
            print(f"    {t.subject} → {t.predicate} → {t.object} [conf={t.confidence:.3f}]")

        print("\n  Confounding relationship (Level 2):")
        conf = [
            t for t in dataset.triples
            if t.subject == confounder and t.predicate == "confounds"
        ]
        if conf:
            t = conf[0]
            print(f"    {t.subject} → {t.predicate} → {t.object} [conf={t.confidence:.3f}]")
            print(f"    This creates spurious correlation!")

        # Check for mediation
        med = [
            t for t in dataset.triples
            if t.subject == confounder and t.predicate == "mediates"
        ]
        if med:
            t = med[0]
            print(f"\n  Mediation (Level 2):")
            print(f"    {t.subject} → {t.predicate} → {t.object} [conf={t.confidence:.3f}]")


def explore_counterfactuals(dataset: CausalTripleDataset):
    """Explore counterfactual reasoning."""
    print_separator("5. Counterfactual Reasoning")

    pairs = dataset.get_counterfactual_pairs()
    print(f"Found {len(pairs)} counterfactual pairs")

    if pairs:
        # Show first pair
        idx1, idx2 = pairs[0]
        s1 = dataset.get_scenario(idx1)
        s2 = dataset.get_scenario(idx2)

        print(f"\nCounterfactual pair example:")
        print(f"\nScenario {idx1}:")
        print(f"  Patient: {s1['patient']}")
        print(f"  Symptom: {s1['symptom']}")
        print(f"  Treatment: {s1['treatment']}")
        print(f"  Confounder: {s1['confounder']}")

        # Get outcome
        outcome1 = [
            dataset.triples[i] for i in s1['triple_indices']
            if dataset.triples[i].predicate in ["symptom_reduced", "symptom_persists"]
        ]
        if outcome1:
            t = outcome1[0]
            print(f"  Outcome: {t.predicate} [conf={t.confidence:.3f}]")

        print(f"\nScenario {idx2}:")
        print(f"  Patient: {s2['patient']}")
        print(f"  Symptom: {s2['symptom']}")
        print(f"  Treatment: {s2['treatment']}")
        print(f"  Confounder: {s2['confounder']}")

        # Get outcome
        outcome2 = [
            dataset.triples[i] for i in s2['triple_indices']
            if dataset.triples[i].predicate in ["symptom_reduced", "symptom_persists"]
        ]
        if outcome2:
            t = outcome2[0]
            print(f"  Outcome: {t.predicate} [conf={t.confidence:.3f}]")

        print("\nCounterfactual question:")
        print(f"  What if {s1['patient']} had taken {s2['treatment']} instead of {s1['treatment']}?")
        print(f"  Would the outcome be like scenario {idx2}?")


def show_graph_construction(dataset: CausalTripleDataset):
    """Demonstrate graph construction from triples."""
    print_separator("6. PyTorch Geometric Graph Construction")

    # Get a single triple as graph
    graph, label = dataset[0]

    print(f"Single triple graph:")
    print(f"  Nodes: {graph.x.shape}")
    print(f"  Edges: {graph.edge_index.shape}")
    print(f"  Edge attributes: {graph.edge_attr.shape}")
    print(f"  Label: {label.item()} (0=ineffective, 1=effective)")

    # Construct full scenario graph
    scenario = dataset.get_scenario(0)
    scenario_graph = dataset.get_graph_for_triples(scenario['triple_indices'])

    print(f"\nFull scenario graph:")
    print(f"  Nodes: {scenario_graph.x.shape}")
    print(f"  Edges: {scenario_graph.edge_index.shape}")
    print(f"  Edge attributes: {scenario_graph.edge_attr.shape}")

    # Show level information if available
    if hasattr(scenario_graph, 'level'):
        print(f"  Node levels: {scenario_graph.level.shape}")


def query_by_treatment(dataset: CausalTripleDataset):
    """Query scenarios by treatment."""
    print_separator("7. Querying by Treatment")

    # Get unique treatments
    treatments = set(s['treatment'] for s in dataset.scenarios)
    print(f"Available treatments: {len(treatments)}")

    # Query for a specific treatment
    if treatments:
        treatment = list(treatments)[0]
        scenarios = dataset.get_scenarios_by_treatment(treatment)

        print(f"\nScenarios using {treatment}: {len(scenarios)}")

        # Show outcomes
        effective = 0
        ineffective = 0

        for idx in scenarios:
            s = dataset.scenarios[idx]
            outcome = [
                dataset.triples[i] for i in s['triple_indices']
                if dataset.triples[i].predicate == "symptom_reduced"
            ]
            if outcome:
                effective += 1
            else:
                ineffective += 1

        print(f"  Effective: {effective}")
        print(f"  Ineffective: {ineffective}")
        if effective + ineffective > 0:
            success_rate = effective / (effective + ineffective)
            print(f"  Success rate: {success_rate:.1%}")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("  CAUSAL REASONING DATASET EXAMPLES")
    print("=" * 80)

    # Create dataset
    dataset = explore_basic_dataset()

    # Show examples
    show_scenario_structure(dataset)
    show_causal_chain(dataset)
    analyze_confounders(dataset)
    explore_counterfactuals(dataset)
    show_graph_construction(dataset)
    query_by_treatment(dataset)

    print_separator()
    print("Examples completed successfully!")
    print(f"Dataset saved to: {dataset.root}")
    print()


if __name__ == '__main__':
    main()
