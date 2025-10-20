"""
Knowledge Graph Triple Dataset

Generates synthetic knowledge graph triples for evaluating NSM's ability
to perform relational reasoning, type inference, and analogical reasoning.

This dataset focuses on entity-centric knowledge with rich relations,
hierarchical types, and partial observability through confidence scores.
"""

from typing import List, Set, Tuple, Dict
import random
import torch

from .dataset import BaseSemanticTripleDataset
from .triple import SemanticTriple


class KnowledgeGraphTripleDataset(BaseSemanticTripleDataset):
    """
    Knowledge Graph dataset for relational reasoning evaluation.

    Generates synthetic but realistic knowledge graphs with:
    - Level 1: Facts/Instances (born_in, won, located_in, works_at, created)
    - Level 2: Categories/Relations (instance_of, subclass_of, typically_has)
    - 50+ predicate types for rich semantic relations
    - 5K entities, 20K triples
    - Confidence scores varying widely (0.5-1.0) for partial observability

    Domain Properties:
        - Entity-centric: People, places, organizations, concepts
        - Rich relations: Biography, geography, achievements, creations
        - Type hierarchy: Instances → categories → abstractions
        - Multi-hop reasoning: Requires chaining 2-5 facts

    Examples:
        >>> dataset = KnowledgeGraphTripleDataset(
        ...     root="data/kg",
        ...     split="train",
        ...     num_entities=1000,
        ...     num_triples=5000
        ... )
        >>> graph, label = dataset[0]
        >>> stats = dataset.get_statistics()

    Mathematical Foundation:
        Knowledge graphs represent entity-relation-entity triples:
        G = (E, R, T) where:
        - E: Set of entities (people, places, concepts)
        - R: Set of typed relations (50+ predicates)
        - T ⊆ E × R × E: Set of typed triples
        - Level 1: Ground facts (high confidence 0.8-1.0)
        - Level 2: Type assertions and generalizations (0.5-0.95)
    """

    # Level 1 predicates: Facts and instances
    LEVEL1_PREDICATES = [
        # Biographical relations
        "born_in", "died_in", "born_on", "died_on",
        "parent_of", "child_of", "sibling_of", "spouse_of",
        "nationality", "citizenship", "ethnicity",

        # Geographic relations
        "located_in", "capital_of", "borders", "part_of",
        "adjacent_to", "near", "contains",

        # Professional relations
        "works_at", "employed_by", "founded", "leads",
        "member_of", "collaborates_with", "reports_to",

        # Educational relations
        "studied_at", "graduated_from", "degree_from",
        "advisor_of", "student_of", "taught_at",

        # Creative relations
        "created", "authored", "composed", "painted",
        "designed", "invented", "discovered", "produced",

        # Achievement relations
        "won", "received", "awarded", "nominated_for",
        "achieved", "accomplished",

        # Temporal relations
        "occurred_in", "started_on", "ended_on",
        "during", "before", "after",

        # Property relations
        "has_property", "characterized_by", "known_for",
        "famous_for", "associated_with",
    ]

    # Level 2 predicates: Types and categories
    LEVEL2_PREDICATES = [
        # Type hierarchy
        "instance_of", "type_of", "kind_of",
        "subclass_of", "superclass_of", "category_of",

        # Typical relations (generalizations)
        "typically_has", "usually_in", "often_associated_with",
        "commonly_has", "generally_requires",

        # Abstract relations
        "related_to", "similar_to", "analogous_to",
        "implies", "suggests", "indicates",
        "enables", "requires", "depends_on",

        # Property generalizations
        "has_attribute", "has_characteristic", "defined_by",
        "characterized_by_type", "property_of_type",
    ]

    # Entity categories for generation
    PERSON_NAMES = [
        "Albert_Einstein", "Marie_Curie", "Isaac_Newton", "Ada_Lovelace",
        "Leonardo_da_Vinci", "Mozart", "Beethoven", "Shakespeare",
        "Aristotle", "Plato", "Confucius", "Gandhi", "Mandela",
        "Turing", "Von_Neumann", "Noether", "Ramanujan",
        "Darwin", "Mendel", "Watson", "Crick", "Franklin",
    ]

    PLACES = [
        "London", "Paris", "Berlin", "Rome", "Madrid",
        "New_York", "Tokyo", "Beijing", "Moscow", "Delhi",
        "California", "Texas", "Bavaria", "Tuscany", "Provence",
        "England", "France", "Germany", "Italy", "Spain",
        "Europe", "Asia", "Africa", "Americas", "Oceania",
    ]

    ORGANIZATIONS = [
        "MIT", "Harvard", "Oxford", "Cambridge", "Stanford",
        "NASA", "CERN", "Max_Planck_Institute", "Bell_Labs",
        "Google", "Microsoft", "Apple", "IBM", "Intel",
        "UN", "WHO", "UNESCO", "Red_Cross",
    ]

    CONCEPTS = [
        "Physics", "Mathematics", "Biology", "Chemistry",
        "Computer_Science", "Philosophy", "Art", "Music",
        "Literature", "History", "Psychology", "Sociology",
        "Quantum_Mechanics", "Relativity", "Evolution",
        "Democracy", "Freedom", "Justice", "Peace",
    ]

    AWARDS = [
        "Nobel_Prize", "Fields_Medal", "Turing_Award",
        "Pulitzer_Prize", "Oscar", "Grammy", "Emmy",
        "National_Medal_of_Science", "Lasker_Award",
    ]

    def __init__(
        self,
        root: str,
        split: str = 'train',
        num_entities: int = 5000,
        num_triples: int = 20000,
        seed: int = 42,
        **kwargs
    ):
        """
        Initialize Knowledge Graph dataset.

        Args:
            root: Root directory for dataset
            split: Dataset split ('train', 'val', 'test')
            num_entities: Target number of unique entities
            num_triples: Number of triples to generate
            seed: Random seed for reproducibility
            **kwargs: Additional arguments for BaseSemanticTripleDataset
        """
        self.num_entities_target = num_entities
        self.num_triples_target = num_triples
        self.seed = seed

        # Set random seeds
        random.seed(seed)
        torch.manual_seed(seed)

        # Entity pools (will be populated during generation)
        self.entities: Set[str] = set()
        self.people: List[str] = []
        self.places: List[str] = []
        self.organizations: List[str] = []
        self.concepts: List[str] = []
        self.awards: List[str] = []
        self.dates: List[str] = []

        # Type mappings for Level 2 reasoning
        self.entity_types: Dict[str, str] = {}
        self.type_hierarchy: Dict[str, str] = {}

        super().__init__(root, split, **kwargs)

    def _generate_entities(self):
        """Generate diverse entity pool."""
        # Start with base entities
        self.people.extend(self.PERSON_NAMES)
        self.places.extend(self.PLACES)
        self.organizations.extend(self.ORGANIZATIONS)
        self.concepts.extend(self.CONCEPTS)
        self.awards.extend(self.AWARDS)

        # Generate additional entities to reach target
        num_base = len(self.people) + len(self.places) + len(self.organizations) + \
                   len(self.concepts) + len(self.awards)

        if num_base < self.num_entities_target:
            # Generate more people
            for i in range((self.num_entities_target - num_base) // 5):
                self.people.append(f"Person_{i}")

            # Generate more places
            for i in range((self.num_entities_target - num_base) // 5):
                self.places.append(f"Place_{i}")

            # Generate more organizations
            for i in range((self.num_entities_target - num_base) // 5):
                self.organizations.append(f"Org_{i}")

            # Generate more concepts
            for i in range((self.num_entities_target - num_base) // 5):
                self.concepts.append(f"Concept_{i}")

            # Generate more awards
            for i in range((self.num_entities_target - num_base) // 5):
                self.awards.append(f"Award_{i}")

        # Generate dates
        for year in range(1800, 2025):
            self.dates.append(f"{year}")

        # Collect all entities
        self.entities.update(self.people)
        self.entities.update(self.places)
        self.entities.update(self.organizations)
        self.entities.update(self.concepts)
        self.entities.update(self.awards)
        self.entities.update(self.dates)

        # Build type mappings
        for person in self.people:
            self.entity_types[person] = "Person"
        for place in self.places:
            self.entity_types[place] = "Place"
        for org in self.organizations:
            self.entity_types[org] = "Organization"
        for concept in self.concepts:
            self.entity_types[concept] = "Concept"
        for award in self.awards:
            self.entity_types[award] = "Award"
        for date in self.dates:
            self.entity_types[date] = "Date"

        # Type hierarchy
        self.type_hierarchy = {
            "Person": "Living_Being",
            "Place": "Location",
            "Organization": "Institution",
            "Concept": "Abstract_Entity",
            "Award": "Recognition",
            "Date": "Temporal_Entity",
            "Living_Being": "Entity",
            "Location": "Entity",
            "Institution": "Entity",
            "Abstract_Entity": "Entity",
            "Recognition": "Entity",
            "Temporal_Entity": "Entity",
        }

    def _generate_biographical_triples(self) -> List[SemanticTriple]:
        """Generate biographical fact triples (Level 1)."""
        triples = []

        # Select subset of people to create rich biographies
        num_rich_bios = min(100, len(self.people))
        people_with_bios = random.sample(self.people, num_rich_bios)

        for person in people_with_bios:
            # Birth information
            if random.random() > 0.3:
                birth_place = random.choice(self.places)
                triples.append(SemanticTriple(
                    subject=person,
                    predicate="born_in",
                    object=birth_place,
                    confidence=random.uniform(0.85, 1.0),
                    level=1,
                    metadata={'category': 'biographical'}
                ))

            # Birth year
            if random.random() > 0.4:
                birth_year = random.choice([y for y in self.dates if int(y) < 1980])
                triples.append(SemanticTriple(
                    subject=person,
                    predicate="born_on",
                    object=birth_year,
                    confidence=random.uniform(0.8, 0.99),
                    level=1,
                    metadata={'category': 'biographical'}
                ))

            # Education
            if random.random() > 0.5:
                university = random.choice(self.organizations)
                triples.append(SemanticTriple(
                    subject=person,
                    predicate="studied_at",
                    object=university,
                    confidence=random.uniform(0.75, 0.98),
                    level=1,
                    metadata={'category': 'educational'}
                ))

            # Work
            if random.random() > 0.4:
                org = random.choice(self.organizations)
                triples.append(SemanticTriple(
                    subject=person,
                    predicate="works_at",
                    object=org,
                    confidence=random.uniform(0.7, 0.95),
                    level=1,
                    metadata={'category': 'professional'}
                ))

            # Achievements
            if random.random() > 0.7:
                award = random.choice(self.awards)
                year = random.choice([y for y in self.dates if int(y) >= 1900])
                award_instance = f"{award}_{year}"
                self.entities.add(award_instance)
                self.entity_types[award_instance] = "Award_Instance"

                triples.append(SemanticTriple(
                    subject=person,
                    predicate="won",
                    object=award_instance,
                    confidence=random.uniform(0.9, 1.0),
                    level=1,
                    metadata={'category': 'achievement'}
                ))

            # Field of work
            if random.random() > 0.5:
                field = random.choice(self.concepts)
                triples.append(SemanticTriple(
                    subject=person,
                    predicate="known_for",
                    object=field,
                    confidence=random.uniform(0.75, 0.95),
                    level=1,
                    metadata={'category': 'professional'}
                ))

        return triples

    def _generate_geographic_triples(self) -> List[SemanticTriple]:
        """Generate geographic relation triples (Level 1)."""
        triples = []

        # Create geographic containment hierarchy
        continents = [p for p in self.places if p in ["Europe", "Asia", "Africa", "Americas", "Oceania"]]
        countries = [p for p in self.places if p in ["England", "France", "Germany", "Italy", "Spain"]]
        cities = [p for p in self.places if p in ["London", "Paris", "Berlin", "Rome", "Madrid"]]

        # Cities in countries
        city_country_map = {
            "London": "England",
            "Paris": "France",
            "Berlin": "Germany",
            "Rome": "Italy",
            "Madrid": "Spain",
        }

        for city, country in city_country_map.items():
            if city in self.places and country in self.places:
                triples.append(SemanticTriple(
                    subject=city,
                    predicate="located_in",
                    object=country,
                    confidence=1.0,
                    level=1,
                    metadata={'category': 'geographic'}
                ))

                triples.append(SemanticTriple(
                    subject=city,
                    predicate="capital_of",
                    object=country,
                    confidence=0.99,
                    level=1,
                    metadata={'category': 'geographic'}
                ))

        # Countries in continents
        country_continent_map = {
            "England": "Europe",
            "France": "Europe",
            "Germany": "Europe",
            "Italy": "Europe",
            "Spain": "Europe",
        }

        for country, continent in country_continent_map.items():
            if country in self.places and continent in self.places:
                triples.append(SemanticTriple(
                    subject=country,
                    predicate="part_of",
                    object=continent,
                    confidence=1.0,
                    level=1,
                    metadata={'category': 'geographic'}
                ))

        # Additional geographic relations
        for _ in range(min(500, len(self.places) * 2)):
            place1 = random.choice(self.places)
            place2 = random.choice(self.places)
            if place1 != place2:
                pred = random.choice(["near", "adjacent_to", "borders"])
                triples.append(SemanticTriple(
                    subject=place1,
                    predicate=pred,
                    object=place2,
                    confidence=random.uniform(0.6, 0.9),
                    level=1,
                    metadata={'category': 'geographic'}
                ))

        return triples

    def _generate_creative_triples(self) -> List[SemanticTriple]:
        """Generate creative work and contribution triples (Level 1)."""
        triples = []

        # Sample of people who created things
        creators = random.sample(self.people, min(50, len(self.people)))

        for creator in creators:
            # Create works
            if random.random() > 0.5:
                work = f"Work_by_{creator}_{random.randint(1, 10)}"
                self.entities.add(work)
                self.entity_types[work] = "Creative_Work"

                pred = random.choice(["created", "authored", "composed", "designed"])
                triples.append(SemanticTriple(
                    subject=creator,
                    predicate=pred,
                    object=work,
                    confidence=random.uniform(0.8, 1.0),
                    level=1,
                    metadata={'category': 'creative'}
                ))

                # Work in a field
                field = random.choice(self.concepts)
                triples.append(SemanticTriple(
                    subject=work,
                    predicate="related_to",
                    object=field,
                    confidence=random.uniform(0.7, 0.95),
                    level=1,
                    metadata={'category': 'creative'}
                ))

        return triples

    def _generate_type_triples(self) -> List[SemanticTriple]:
        """Generate type and category triples (Level 2)."""
        triples = []

        # Instance-of relations
        for entity, entity_type in self.entity_types.items():
            # Sample some entities to avoid too many type triples
            if random.random() > 0.7 or entity in self.PERSON_NAMES + self.PLACES[:10]:
                triples.append(SemanticTriple(
                    subject=entity,
                    predicate="instance_of",
                    object=entity_type,
                    confidence=random.uniform(0.85, 0.99),
                    level=2,
                    metadata={'category': 'type'}
                ))

        # Subclass relations (type hierarchy)
        for child_type, parent_type in self.type_hierarchy.items():
            triples.append(SemanticTriple(
                subject=child_type,
                predicate="subclass_of",
                object=parent_type,
                confidence=random.uniform(0.9, 1.0),
                level=2,
                metadata={'category': 'type_hierarchy'}
            ))

        # Typical relations (generalizations)
        generalizations = [
            ("Person", "typically_has", "Birth_Place", 0.95),
            ("Person", "typically_has", "Nationality", 0.98),
            ("Award", "usually_in", "Recognition_Domain", 0.85),
            ("Organization", "commonly_has", "Location", 0.9),
            ("Creative_Work", "often_associated_with", "Creator", 0.99),
            ("Place", "commonly_has", "Geographic_Coordinates", 0.95),
        ]

        for subj, pred, obj, conf in generalizations:
            # Add these abstract entities
            self.entities.add(obj)
            triples.append(SemanticTriple(
                subject=subj,
                predicate=pred,
                object=obj,
                confidence=conf,
                level=2,
                metadata={'category': 'generalization'}
            ))

        # Abstract relations between concepts
        for _ in range(min(200, len(self.concepts) * 3)):
            concept1 = random.choice(self.concepts)
            concept2 = random.choice(self.concepts)
            if concept1 != concept2:
                pred = random.choice(["related_to", "similar_to", "requires", "enables"])
                triples.append(SemanticTriple(
                    subject=concept1,
                    predicate=pred,
                    object=concept2,
                    confidence=random.uniform(0.5, 0.85),
                    level=2,
                    metadata={'category': 'conceptual'}
                ))

        return triples

    def generate_triples(self) -> List[SemanticTriple]:
        """
        Generate knowledge graph triples.

        Returns:
            List of SemanticTriple objects combining facts (L1) and types (L2)
        """
        # Generate entity pool
        self._generate_entities()

        triples = []

        # Generate Level 1 triples (facts)
        triples.extend(self._generate_biographical_triples())
        triples.extend(self._generate_geographic_triples())
        triples.extend(self._generate_creative_triples())

        # Generate Level 2 triples (types and generalizations)
        triples.extend(self._generate_type_triples())

        # If we have fewer triples than target, add more random relations
        while len(triples) < self.num_triples_target:
            # Random Level 1 facts
            entity1 = random.choice(list(self.entities))
            entity2 = random.choice(list(self.entities))
            if entity1 != entity2:
                pred = random.choice(self.LEVEL1_PREDICATES)
                triples.append(SemanticTriple(
                    subject=entity1,
                    predicate=pred,
                    object=entity2,
                    confidence=random.uniform(0.6, 0.9),
                    level=1,
                    metadata={'category': 'random'}
                ))

        # Shuffle and trim to exact target
        random.shuffle(triples)
        return triples[:self.num_triples_target]

    def generate_labels(self, idx: int) -> torch.Tensor:
        """
        Generate link prediction labels with negative sampling.

        For knowledge graphs, the task is link prediction:
        given (subject, predicate, ?), predict if a candidate object is valid.

        Strategy:
        - First 50% of indices: True triples (label=1)
        - Last 50% of indices: Corrupted triples (label=0)

        Corrupted triples are generated by randomly replacing the object
        with another entity, creating invalid facts.

        Args:
            idx: Triple index

        Returns:
            Binary label (0 or 1) for link prediction
        """
        num_true_triples = len(self.triples) // 2

        if idx < num_true_triples:
            # True triple (positive example)
            return torch.tensor(1, dtype=torch.long)
        else:
            # Corrupted triple (negative example)
            return torch.tensor(0, dtype=torch.long)

    def get_multi_hop_queries(self, num_queries: int = 100) -> List[Dict]:
        """
        Generate multi-hop reasoning queries.

        Returns:
            List of query dictionaries with:
                - start_entity: Starting entity
                - relations: List of relations to traverse
                - expected_answers: Set of valid answer entities
        """
        queries = []

        # Find chains in the data
        # Build adjacency for each predicate
        graph = {}
        for triple in self.triples:
            if triple.level == 1:  # Focus on facts
                if triple.subject not in graph:
                    graph[triple.subject] = []
                graph[triple.subject].append((triple.predicate, triple.object))

        # Generate 2-hop queries
        for _ in range(num_queries):
            # Pick random starting entity with outgoing edges
            entities_with_edges = [e for e in graph.keys() if len(graph[e]) > 0]
            if not entities_with_edges:
                break

            start = random.choice(entities_with_edges)

            # First hop
            if start not in graph or len(graph[start]) == 0:
                continue
            pred1, intermediate = random.choice(graph[start])

            # Second hop
            if intermediate not in graph or len(graph[intermediate]) == 0:
                continue
            pred2, end = random.choice(graph[intermediate])

            queries.append({
                'start_entity': start,
                'relations': [pred1, pred2],
                'intermediate': intermediate,
                'expected_answer': end,
                'query_type': '2-hop'
            })

        return queries

    def get_type_consistency_pairs(self, num_pairs: int = 100) -> List[Tuple[str, str, bool]]:
        """
        Generate entity-type pairs for consistency checking.

        Returns:
            List of (entity, type, is_consistent) tuples
        """
        pairs = []

        # Positive examples (consistent)
        entities_with_types = [(e, t) for e, t in self.entity_types.items()
                               if e in self.entities]
        positive_samples = random.sample(
            entities_with_types,
            min(num_pairs // 2, len(entities_with_types))
        )

        for entity, entity_type in positive_samples:
            pairs.append((entity, entity_type, True))

        # Negative examples (inconsistent)
        for _ in range(num_pairs - len(pairs)):
            entity = random.choice(list(self.entities))
            if entity in self.entity_types:
                # Pick wrong type
                wrong_type = random.choice(list(set(self.entity_types.values()) - {self.entity_types[entity]}))
                pairs.append((entity, wrong_type, False))

        return pairs

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"KnowledgeGraphTripleDataset("
            f"split='{self.split}', "
            f"num_triples={len(self.triples)}, "
            f"num_entities={len(self.entities)}, "
            f"num_predicates={self.vocabulary.num_predicates})"
        )
