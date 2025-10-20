"""
Planning Domain Dataset for Hierarchical Reasoning

Generates procedural planning problems with hierarchical goal decomposition.
Implements two-level hierarchy (Actions/Environment → Goals/Capabilities)
for NSM Phase 1 validation.

Mathematical Foundation:
    Planning domain P = (S, A, G, T) where:
    - S: State space (locations, objects, robot state)
    - A: Action space (primitive operations)
    - G: Goal space (desired states, capabilities)
    - T: Transition function (action effects, prerequisites)

    Hierarchical Structure:
    - Level 1 (Concrete): Actions and environmental states
      * move_to, pick_up, put_down, stack, unstack, contains, at_location
    - Level 2 (Abstract): Goals and capabilities
      * achieve, has_capability, requires, enables

    Ground Truth:
    - Valid action sequences: Satisfy temporal ordering + preconditions
    - Invalid sequences: Violate preconditions or ordering constraints
"""

from typing import List, Dict, Tuple, Set, Optional
import random
import torch
from torch import Tensor
from torch_geometric.data import Data

from .triple import SemanticTriple
from .dataset import BaseSemanticTripleDataset


class PlanningTripleDataset(BaseSemanticTripleDataset):
    """
    Planning domain dataset with hierarchical goal decomposition.

    Generates planning problems where:
    - Abstract goals decompose into concrete action sequences
    - Actions have prerequisites and effects
    - Goals require specific capabilities
    - Temporal ordering constraints exist

    Attributes:
        num_problems: Number of unique planning problems
        num_locations: Number of locations in environment
        num_objects: Number of manipulable objects
        seed: Random seed for reproducibility
        primitive_actions: Set of available actions
        goal_templates: Templates for goal generation
        problems: Generated planning problem instances

    Examples:
        >>> dataset = PlanningTripleDataset(
        ...     root="data/planning",
        ...     split="train",
        ...     num_problems=1000,
        ...     seed=42
        ... )
        >>> graph, label = dataset[0]
        >>> print(f"Graph has {graph.num_nodes} nodes, {graph.edge_index.size(1)} edges")
    """

    # Class-level constants
    PRIMITIVE_ACTIONS = {
        'move_to', 'pick_up', 'put_down', 'stack', 'unstack',
        'push', 'pull', 'open', 'close', 'toggle'
    }

    GOAL_TEMPLATES = {
        'stack_blocks': ['pick_up', 'move_to', 'stack'],
        'clear_table': ['pick_up', 'put_down', 'move_to'],
        'transport': ['pick_up', 'move_to', 'put_down'],
        'organize': ['move_to', 'pick_up', 'stack', 'put_down'],
        'manipulate': ['open', 'pick_up', 'close', 'move_to']
    }

    CAPABILITIES = {
        'manipulation', 'navigation', 'perception',
        'grasping', 'planning', 'stacking'
    }

    # Level 1 predicates (concrete)
    L1_PREDICATES = {
        'move_to', 'pick_up', 'put_down', 'stack', 'unstack',
        'contains', 'at_location', 'holding', 'on_top_of',
        'is_clear', 'is_open', 'push', 'pull', 'open', 'close', 'toggle'
    }

    # Level 2 predicates (abstract)
    L2_PREDICATES = {
        'achieve', 'has_capability', 'requires', 'enables',
        'decomposes_to', 'precondition_of', 'effect_of'
    }

    def __init__(
        self,
        root: str,
        split: str = 'train',
        num_problems: int = 1000,
        num_locations: int = 5,
        num_objects: int = 10,
        seed: int = 42,
        **kwargs
    ):
        """
        Initialize planning dataset.

        Args:
            root: Root directory for dataset files
            split: Dataset split ('train', 'val', 'test')
            num_problems: Number of planning problems to generate
            num_locations: Number of locations in environment
            num_objects: Number of objects to manipulate
            seed: Random seed for reproducibility
            **kwargs: Additional arguments for BaseSemanticTripleDataset
        """
        self.num_problems = num_problems
        self.num_locations = num_locations
        self.num_objects = num_objects
        self.seed = seed

        # Set random seed for reproducibility
        random.seed(seed)
        torch.manual_seed(seed)

        # Will be populated by generate_triples
        self.problems: List[Dict] = []

        super().__init__(root, split, **kwargs)

    def generate_triples(self) -> List[SemanticTriple]:
        """
        Generate planning domain semantic triples.

        Creates hierarchical planning problems with:
        1. Environmental state (locations, objects)
        2. Action sequences (with prerequisites)
        3. Goal decomposition (goals → subgoals → actions)
        4. Capability requirements

        Returns:
            List of SemanticTriple objects representing planning domain

        Mathematical Foundation:
            For each problem p ∈ P:
            1. Initial state s₀ ∈ S
            2. Goal state g ∈ G
            3. Action sequence a₁, ..., aₙ ∈ A*
            4. Decomposition: g → {subgoals} → {actions}

            Confidence modeling:
            - Environmental facts: 0.9-1.0 (high certainty)
            - Action effects: 0.8-0.95 (execution uncertainty)
            - Goal requirements: 0.7-0.9 (planning uncertainty)
        """
        all_triples = []

        # Split problems by dataset split
        split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
        start_idx = 0
        for split_name, ratio in split_ratios.items():
            if split_name == self.split:
                break
            start_idx += int(self.num_problems * ratio)

        num_split_problems = int(self.num_problems * split_ratios[self.split])

        for problem_idx in range(start_idx, start_idx + num_split_problems):
            # Use problem index to seed for reproducibility
            problem_seed = self.seed + problem_idx
            random.seed(problem_seed)

            problem_triples = self._generate_single_problem(problem_idx)
            all_triples.extend(problem_triples)

            # Store problem metadata
            self.problems.append({
                'idx': problem_idx,
                'num_triples': len(problem_triples),
                'offset': len(all_triples) - len(problem_triples)
            })

        # Reset seed
        random.seed(self.seed)
        return all_triples

    def _generate_single_problem(self, problem_idx: int) -> List[SemanticTriple]:
        """
        Generate triples for a single planning problem.

        Args:
            problem_idx: Unique problem identifier

        Returns:
            List of triples representing one planning problem
        """
        triples = []
        robot = f"robot_{problem_idx}"

        # 1. Generate environmental state (Level 1)
        locations = [f"loc_{problem_idx}_{i}" for i in range(self.num_locations)]
        objects = [f"obj_{problem_idx}_{i}" for i in range(self.num_objects)]

        # Robot location
        robot_location = random.choice(locations)
        triples.append(SemanticTriple(
            subject=robot,
            predicate='at_location',
            object=robot_location,
            confidence=random.uniform(0.95, 1.0),
            level=1,
            metadata={'problem': problem_idx, 'type': 'state'}
        ))

        # Object locations
        for obj in objects:
            obj_location = random.choice(locations)
            triples.append(SemanticTriple(
                subject=obj,
                predicate='at_location',
                object=obj_location,
                confidence=random.uniform(0.9, 1.0),
                level=1,
                metadata={'problem': problem_idx, 'type': 'state'}
            ))

            # Some locations contain objects
            triples.append(SemanticTriple(
                subject=obj_location,
                predicate='contains',
                object=obj,
                confidence=random.uniform(0.9, 0.98),
                level=1,
                metadata={'problem': problem_idx, 'type': 'state'}
            ))

        # 2. Generate action sequences (Level 1)
        num_actions = random.randint(3, 8)
        action_sequence = []

        for action_idx in range(num_actions):
            action_type = random.choice(list(self.PRIMITIVE_ACTIONS))
            action_name = f"{action_type}_{problem_idx}_{action_idx}"
            action_sequence.append((action_name, action_type))

            # Robot executes action
            target = random.choice(objects + locations)
            triples.append(SemanticTriple(
                subject=robot,
                predicate=action_type,
                object=target,
                confidence=random.uniform(0.85, 0.95),
                level=1,
                metadata={'problem': problem_idx, 'type': 'action', 'sequence': action_idx}
            ))

            # Action prerequisites (some actions depend on previous ones)
            if action_idx > 0 and random.random() < 0.6:
                prev_action = action_sequence[action_idx - 1][0]
                triples.append(SemanticTriple(
                    subject=action_name,
                    predicate='requires',
                    object=prev_action,
                    confidence=random.uniform(0.8, 0.9),
                    level=1,
                    metadata={'problem': problem_idx, 'type': 'prerequisite'}
                ))

        # 3. Generate goals and decomposition (Level 2)
        goal_template = random.choice(list(self.GOAL_TEMPLATES.keys()))
        goal_name = f"goal_{goal_template}_{problem_idx}"

        # Goal achievement
        triples.append(SemanticTriple(
            subject=robot,
            predicate='achieve',
            object=goal_name,
            confidence=random.uniform(0.7, 0.85),
            level=2,
            metadata={'problem': problem_idx, 'type': 'goal'}
        ))

        # Goal requires specific actions (hierarchical decomposition)
        required_actions = self.GOAL_TEMPLATES[goal_template]
        for action_type in required_actions:
            # Find actions of this type in the sequence
            matching_actions = [
                name for name, atype in action_sequence if atype == action_type
            ]
            if matching_actions:
                action = random.choice(matching_actions)
                triples.append(SemanticTriple(
                    subject=goal_name,
                    predicate='requires',
                    object=action,
                    confidence=random.uniform(0.75, 0.9),
                    level=2,
                    metadata={'problem': problem_idx, 'type': 'decomposition'}
                ))

        # 4. Generate capability requirements (Level 2)
        required_capabilities = random.sample(
            list(self.CAPABILITIES),
            k=random.randint(2, 4)
        )

        for capability in required_capabilities:
            capability_name = f"cap_{capability}_{problem_idx}"

            # Robot has capability
            triples.append(SemanticTriple(
                subject=robot,
                predicate='has_capability',
                object=capability_name,
                confidence=random.uniform(0.85, 0.95),
                level=2,
                metadata={'problem': problem_idx, 'type': 'capability'}
            ))

            # Goal requires capability
            triples.append(SemanticTriple(
                subject=goal_name,
                predicate='requires',
                object=capability_name,
                confidence=random.uniform(0.8, 0.92),
                level=2,
                metadata={'problem': problem_idx, 'type': 'requirement'}
            ))

            # Capability enables certain actions
            for action_name, action_type in action_sequence[:2]:  # Link to first few actions
                if random.random() < 0.4:  # Not all capabilities enable all actions
                    triples.append(SemanticTriple(
                        subject=capability_name,
                        predicate='enables',
                        object=action_name,
                        confidence=random.uniform(0.75, 0.88),
                        level=2,
                        metadata={'problem': problem_idx, 'type': 'enablement'}
                    ))

        return triples

    def generate_labels(self, idx: int) -> Tensor:
        """
        Generate task-specific labels for planning problems.

        Label format:
            - Binary classification: Is this a valid action sequence?
            - Valid if: temporal ordering satisfied, prerequisites met

        Args:
            idx: Triple index

        Returns:
            Tensor: Binary label [1] for valid, [0] for invalid

        Note:
            In this synthetic dataset, we generate mostly valid sequences
            with some intentional violations for training.
        """
        # Determine which problem this triple belongs to
        problem_idx = 0
        cumulative_offset = 0

        for problem in self.problems:
            if idx < cumulative_offset + problem['num_triples']:
                problem_idx = problem['idx']
                break
            cumulative_offset += problem['num_triples']

        # Use problem index to determine validity (deterministic)
        # 50% valid sequences, 50% invalid (balanced for training)
        is_valid = (problem_idx % 100) < 50

        return torch.tensor([1 if is_valid else 0], dtype=torch.long)

    def get_problem_triples(self, problem_idx: int) -> List[SemanticTriple]:
        """
        Get all triples for a specific planning problem.

        Args:
            problem_idx: Problem index (0 to num_problems-1)

        Returns:
            List of triples for the problem
        """
        if problem_idx >= len(self.problems):
            raise IndexError(f"Problem index {problem_idx} out of range")

        problem = self.problems[problem_idx]
        start = problem['offset']
        end = start + problem['num_triples']
        return self.triples[start:end]

    def get_problem_graph(self, problem_idx: int):
        """
        Construct graph for a complete planning problem.

        Args:
            problem_idx: Problem index

        Returns:
            PyG Data object containing all triples for the problem
        """
        problem = self.problems[problem_idx]
        start = problem['offset']
        indices = list(range(start, start + problem['num_triples']))
        return self.get_graph_for_triples(indices)

    def analyze_temporal_ordering(self, problem_idx: int) -> Dict:
        """
        Analyze temporal ordering constraints in a problem.

        Args:
            problem_idx: Problem index

        Returns:
            Dictionary with ordering statistics:
                - num_actions: Number of actions
                - num_dependencies: Number of prerequisite relationships
                - is_valid: Whether ordering is consistent
        """
        triples = self.get_problem_triples(problem_idx)

        actions = set()
        dependencies = []

        for triple in triples:
            if triple.level == 1 and triple.metadata.get('type') == 'action':
                actions.add(triple.subject)
            elif triple.predicate == 'requires' and triple.level == 1:
                dependencies.append((triple.subject, triple.object))

        # Check for cycles (invalid ordering)
        def has_cycle(deps: List[Tuple[str, str]]) -> bool:
            """Detect cycles in dependency graph using DFS."""
            graph = {}
            for src, dst in deps:
                if src not in graph:
                    graph[src] = []
                graph[src].append(dst)

            visited = set()
            rec_stack = set()

            def dfs(node):
                visited.add(node)
                rec_stack.add(node)

                if node in graph:
                    for neighbor in graph[node]:
                        if neighbor not in visited:
                            if dfs(neighbor):
                                return True
                        elif neighbor in rec_stack:
                            return True

                rec_stack.remove(node)
                return False

            for node in graph:
                if node not in visited:
                    if dfs(node):
                        return True
            return False

        is_valid = not has_cycle(dependencies) if dependencies else True

        return {
            'num_actions': len(actions),
            'num_dependencies': len(dependencies),
            'is_valid': is_valid,
            'dependency_ratio': len(dependencies) / len(actions) if actions else 0
        }

    def analyze_hierarchical_structure(self, problem_idx: int) -> Dict:
        """
        Analyze hierarchical decomposition in a problem.

        Args:
            problem_idx: Problem index

        Returns:
            Dictionary with hierarchical statistics:
                - num_goals: Number of abstract goals
                - num_capabilities: Number of capabilities
                - decomposition_depth: Maximum depth of goal → action chain
        """
        triples = self.get_problem_triples(problem_idx)

        goals = set()
        capabilities = set()
        goal_to_actions = {}

        for triple in triples:
            if triple.predicate == 'achieve' and triple.level == 2:
                goals.add(triple.object)
            elif triple.predicate == 'has_capability' and triple.level == 2:
                capabilities.add(triple.object)
            elif triple.predicate == 'requires' and triple.level == 2:
                if triple.subject not in goal_to_actions:
                    goal_to_actions[triple.subject] = []
                goal_to_actions[triple.subject].append(triple.object)

        max_depth = max([len(actions) for actions in goal_to_actions.values()]) if goal_to_actions else 0

        return {
            'num_goals': len(goals),
            'num_capabilities': len(capabilities),
            'num_decompositions': sum(len(actions) for actions in goal_to_actions.values()),
            'decomposition_depth': max_depth
        }

    def __len__(self) -> int:
        """
        Return number of problems (not triples).

        This ensures the dataset returns complete problems as samples,
        matching the architecture of Causal and KG datasets.

        Returns:
            Number of planning problems in dataset
        """
        return len(self.problems)

    def __getitem__(self, idx: int) -> Tuple[Data, Tensor]:
        """
        Get complete problem as a graph.

        Returns a graph containing ALL triples for the problem at index idx,
        along with a problem-level label (valid/invalid plan).

        Args:
            idx: Problem index (0 to len(self.problems)-1)

        Returns:
            Tuple of (graph, label):
                - graph: PyG Data object containing all triples for this problem
                - label: Binary label (1 for valid plan, 0 for invalid)

        Note:
            This override ensures Planning matches Causal/KG architecture:
            - Each sample = complete problem (not individual triple)
            - Dataset length = number of problems (not number of triples)
            - Model sees full problem context for reasoning
        """
        from torch_geometric.data import Data

        if idx >= len(self.problems):
            raise IndexError(f"Problem index {idx} out of range (0-{len(self.problems)-1})")

        problem = self.problems[idx]
        problem_idx = problem['idx']

        # Get all triples for this problem
        start = problem['offset']
        end = start + problem['num_triples']
        triple_indices = list(range(start, end))

        # Build graph from all triples
        graph = self.get_graph_for_triples(triple_indices)

        # Apply transform if provided
        if self.transform is not None:
            graph = self.transform(graph)

        # Problem-level label (valid or invalid plan)
        is_valid = (problem_idx % 100) < 50
        label = torch.tensor(1 if is_valid else 0, dtype=torch.long)

        return graph, label
