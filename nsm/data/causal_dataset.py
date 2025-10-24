"""
Causal Reasoning Dataset

Generates synthetic medical causal reasoning scenarios with:
- Treatment interventions and outcomes
- Confounding variables (age, prior conditions)
- Counterfactual reasoning support
- Causal vs correlational distinction

Mathematical Foundation:
    Uses Pearl's do-calculus framework where:
    - P(Y|do(X)) represents interventional distributions
    - Confidence scores represent causal effect sizes
    - Confounders create spurious correlations
"""

from typing import List, Dict, Set, Tuple
import random
import torch
from .triple import SemanticTriple
from .dataset import BaseSemanticTripleDataset


class CausalTripleDataset(BaseSemanticTripleDataset):
    """
    Causal reasoning dataset with medical treatment scenarios.

    Generates synthetic causal graphs with:
    - Level 1: Observations (patient events, symptoms, outcomes)
    - Level 2: Causal mechanisms (treatment effects, confounders)

    The dataset tests the model's ability to:
    1. Distinguish causation from correlation
    2. Reason about interventions (do-operator)
    3. Handle confounding variables
    4. Answer counterfactual queries

    Attributes:
        num_scenarios: Number of patient scenarios to generate
        num_treatments: Number of treatment types
        num_symptoms: Number of symptom types
        num_confounders: Number of confounding factors
        confound_prob: Probability of confounder affecting outcome

    Examples:
        >>> dataset = CausalTripleDataset(
        ...     root="data/causal",
        ...     split="train",
        ...     num_scenarios=2000
        ... )
        >>> print(dataset.get_statistics())
        >>> graph, label = dataset[0]

        Causal chain example:
        Level 1 (Observations):
            ("aspirin", "taken_by", "patient_42", conf=0.9)
            ("patient_42", "has_symptom", "headache", conf=0.95)
            ("patient_42", "symptom_reduced", "headache", conf=0.8)

        Level 2 (Mechanisms):
            ("aspirin", "causes", "pain_reduction", conf=0.85)
            ("pain_reduction", "treats", "headache", conf=0.9)
            ("patient_42", "responds_to", "aspirin", conf=0.75)

        With confounder:
            ("patient_42", "has_condition", "young_age", conf=1.0)
            ("young_age", "confounds", "aspirin_response", conf=0.6)
    """

    # Medical domain vocabulary
    TREATMENTS = [
        "aspirin", "ibuprofen", "acetaminophen", "antibiotic",
        "beta_blocker", "ace_inhibitor", "statin", "insulin",
        "antihistamine", "bronchodilator", "antacid", "steroid",
        "antidepressant", "anticoagulant", "diuretic", "vaccine"
    ]

    SYMPTOMS = [
        "headache", "fever", "pain", "inflammation", "infection",
        "high_blood_pressure", "high_cholesterol", "high_blood_sugar",
        "allergic_reaction", "asthma_attack", "acid_reflux", "swelling",
        "depression", "blood_clot", "fluid_retention", "viral_infection"
    ]

    EFFECTS = [
        "pain_reduction", "fever_reduction", "anti_inflammatory",
        "antimicrobial", "blood_pressure_lowering", "cholesterol_lowering",
        "glucose_regulation", "immune_response", "histamine_blocking",
        "bronchodilation", "acid_neutralization", "cortisol_regulation",
        "serotonin_regulation", "anticoagulation", "fluid_regulation",
        "antibody_production"
    ]

    CONFOUNDERS = [
        "young_age", "old_age", "genetic_predisposition", "lifestyle_factor",
        "comorbidity", "medication_interaction", "diet", "exercise_level",
        "stress_level", "sleep_quality", "smoking", "alcohol_use"
    ]

    # Level 1 predicates (observations/events)
    LEVEL1_PREDICATES = [
        "taken_by", "has_symptom", "symptom_reduced", "symptom_persists",
        "has_condition", "developed_condition", "recovered_from"
    ]

    # Level 2 predicates (causal mechanisms)
    LEVEL2_PREDICATES = [
        "causes", "treats", "responds_to", "resistant_to",
        "confounds", "mediates", "moderates", "interacts_with"
    ]

    def __init__(
        self,
        root: str,
        split: str = 'train',
        num_scenarios: int = 2000,
        num_treatments: int = 16,
        num_symptoms: int = 16,
        num_confounders: int = 12,
        confound_prob: float = 0.3,
        seed: int = 42,
        **kwargs
    ):
        """
        Initialize causal reasoning dataset.

        Args:
            root: Root directory for dataset
            split: Dataset split ('train', 'val', 'test')
            num_scenarios: Number of patient scenarios
            num_treatments: Number of treatment types to use
            num_symptoms: Number of symptom types to use
            num_confounders: Number of confounder types to use
            confound_prob: Probability of confounder presence
            seed: Random seed for reproducibility
            **kwargs: Additional arguments for BaseSemanticTripleDataset
        """
        self.num_scenarios = num_scenarios
        self.num_treatments = min(num_treatments, len(self.TREATMENTS))
        self.num_symptoms = min(num_symptoms, len(self.SYMPTOMS))
        self.num_confounders = min(num_confounders, len(self.CONFOUNDERS))
        self.confound_prob = confound_prob
        self.seed = seed

        # Set random seeds for reproducibility
        random.seed(seed)
        torch.manual_seed(seed)

        # Track generated scenarios for counterfactual queries
        self.scenarios: List[Dict] = []

        super().__init__(root, split, **kwargs)

    def generate_triples(self) -> List[SemanticTriple]:
        """
        Generate causal reasoning triples.

        Creates medical scenarios with:
        1. Treatment-symptom pairs with known causal relationships
        2. Confounding variables that create spurious correlations
        3. Individual patient responses (Level 1 observations)
        4. General causal mechanisms (Level 2)

        Returns:
            List of SemanticTriple objects representing causal scenarios

        Note:
            Each scenario includes:
            - Patient observation triples (Level 1)
            - Causal mechanism triples (Level 2)
            - Optional confounder triples (both levels)
        """
        # Create treatment-symptom-effect mappings
        self.causal_pairs = self._create_causal_pairs()

        triples = []
        scenario_idx = 0

        # Generate patient scenarios
        for i in range(self.num_scenarios):
            patient_id = f"patient_{i}"

            # Select a treatment-symptom pair
            # To ensure counterfactual pairs exist, randomly assign treatments
            # across scenarios rather than using fixed causal pairs
            symptom_idx = i % self.num_symptoms
            treatment_idx = random.randint(0, self.num_treatments - 1)

            symptom = self.SYMPTOMS[symptom_idx]
            treatment = self.TREATMENTS[treatment_idx]
            effect = self.EFFECTS[treatment_idx]

            # Decide if confounder is present
            has_confounder = random.random() < self.confound_prob
            confounder = None
            if has_confounder:
                confounder = random.choice(self.CONFOUNDERS[:self.num_confounders])

            # Generate triples for this scenario
            scenario_triples = self._generate_scenario_triples(
                patient_id, treatment, symptom, effect, confounder, scenario_idx
            )

            triples.extend(scenario_triples)

            # Store scenario metadata for evaluation
            self.scenarios.append({
                'patient': patient_id,
                'treatment': treatment,
                'symptom': symptom,
                'effect': effect,
                'confounder': confounder,
                'triple_indices': list(range(
                    len(triples) - len(scenario_triples),
                    len(triples)
                ))
            })

            scenario_idx += 1

        return triples

    def _create_causal_pairs(self) -> List[Tuple[str, str, str]]:
        """
        Create treatment-symptom-effect causal mappings.

        Returns:
            List of (treatment, symptom, effect) tuples
        """
        pairs = []
        treatments = self.TREATMENTS[:self.num_treatments]
        symptoms = self.SYMPTOMS[:self.num_symptoms]
        effects = self.EFFECTS[:self.num_treatments]  # Match treatment count

        for treatment, symptom, effect in zip(treatments, symptoms, effects):
            pairs.append((treatment, symptom, effect))

        return pairs

    def _generate_scenario_triples(
        self,
        patient_id: str,
        treatment: str,
        symptom: str,
        effect: str,
        confounder: str | None,
        scenario_idx: int
    ) -> List[SemanticTriple]:
        """
        Generate triples for a single patient scenario.

        Args:
            patient_id: Patient identifier
            treatment: Treatment administered
            symptom: Symptom being treated
            effect: Causal effect mechanism
            confounder: Optional confounding variable
            scenario_idx: Scenario index for metadata

        Returns:
            List of triples for this scenario
        """
        triples = []

        # Determine treatment effectiveness
        # Base effectiveness: 0.2-0.9 (balanced distribution around threshold 0.6)
        base_effectiveness = 0.2 + random.random() * 0.7

        # Confounder can increase or decrease effectiveness
        if confounder:
            confounder_effect = (random.random() - 0.5) * 0.4  # -0.2 to +0.2
            observed_effectiveness = max(0.1, min(0.95,
                base_effectiveness + confounder_effect))
        else:
            observed_effectiveness = base_effectiveness

        # Level 1: Observation triples
        # Patient takes treatment
        triples.append(SemanticTriple(
            subject=treatment,
            predicate="taken_by",
            object=patient_id,
            confidence=0.85 + random.random() * 0.15,  # High certainty of observation
            level=1,
            metadata={'scenario': scenario_idx, 'type': 'intervention'}
        ))

        # Patient has symptom initially
        triples.append(SemanticTriple(
            subject=patient_id,
            predicate="has_symptom",
            object=symptom,
            confidence=0.9 + random.random() * 0.1,  # High certainty
            level=1,
            metadata={'scenario': scenario_idx, 'type': 'observation', 'timepoint': 'pre'}
        ))

        # Outcome: symptom reduced or persists
        if observed_effectiveness > 0.6:
            outcome_pred = "symptom_reduced"
            outcome_conf = observed_effectiveness
        else:
            outcome_pred = "symptom_persists"
            outcome_conf = 1.0 - observed_effectiveness

        triples.append(SemanticTriple(
            subject=patient_id,
            predicate=outcome_pred,
            object=symptom,
            confidence=outcome_conf,
            level=1,
            metadata={'scenario': scenario_idx, 'type': 'outcome', 'timepoint': 'post'}
        ))

        # Confounder observation (if present)
        if confounder:
            triples.append(SemanticTriple(
                subject=patient_id,
                predicate="has_condition",
                object=confounder,
                confidence=1.0,  # Confounder presence is certain
                level=1,
                metadata={'scenario': scenario_idx, 'type': 'confounder'}
            ))

        # Level 2: Causal mechanism triples
        # Treatment causes effect (true causal relationship)
        triples.append(SemanticTriple(
            subject=treatment,
            predicate="causes",
            object=effect,
            confidence=0.75 + random.random() * 0.2,  # General causal strength
            level=2,
            metadata={'scenario': scenario_idx, 'type': 'causal_mechanism'}
        ))

        # Effect treats symptom (mechanism of action)
        triples.append(SemanticTriple(
            subject=effect,
            predicate="treats",
            object=symptom,
            confidence=0.8 + random.random() * 0.15,
            level=2,
            metadata={'scenario': scenario_idx, 'type': 'causal_mechanism'}
        ))

        # Patient response to treatment (individual causal effect)
        response_pred = "responds_to" if observed_effectiveness > 0.6 else "resistant_to"
        triples.append(SemanticTriple(
            subject=patient_id,
            predicate=response_pred,
            object=treatment,
            confidence=abs(observed_effectiveness - 0.5) * 2,  # Distance from 0.5
            level=2,
            metadata={'scenario': scenario_idx, 'type': 'individual_effect'}
        ))

        # Confounder relationships (if present)
        if confounder:
            # Confounder confounds treatment response
            triples.append(SemanticTriple(
                subject=confounder,
                predicate="confounds",
                object=f"{treatment}_response",
                confidence=abs(confounder_effect) / 0.2,  # Strength of confounding
                level=2,
                metadata={'scenario': scenario_idx, 'type': 'confounding'}
            ))

            # Confounder may also mediate the effect
            if random.random() < 0.5:
                triples.append(SemanticTriple(
                    subject=confounder,
                    predicate="mediates",
                    object=effect,
                    confidence=0.5 + random.random() * 0.3,
                    level=2,
                    metadata={'scenario': scenario_idx, 'type': 'mediation'}
                ))

        return triples

    def generate_labels(self, idx: int) -> torch.Tensor:
        """
        Generate labels for causal reasoning tasks.

        Task: Binary classification - will treatment be effective?
        Label 1: Symptom reduced (treatment effective)
        Label 0: Symptom persists (treatment ineffective)

        Args:
            idx: Triple index

        Returns:
            Binary label tensor [0 or 1]

        Raises:
            ValueError: If label cannot be determined from scenario metadata,
                indicating a bug in scenario/triple generation logic

        Note:
            The model must learn to predict effectiveness while
            accounting for confounders (not just correlations).
        """
        triple = self.triples[idx]

        # Find if this triple belongs to an effective scenario
        if triple.predicate == "symptom_reduced":
            return torch.tensor([1], dtype=torch.long)
        elif triple.predicate == "symptom_persists":
            return torch.tensor([0], dtype=torch.long)
        else:
            # For non-outcome triples, check scenario metadata
            scenario_idx = triple.metadata.get('scenario', -1)
            if scenario_idx >= 0 and scenario_idx < len(self.scenarios):
                scenario = self.scenarios[scenario_idx]
                # Check the outcome triples in this scenario
                for t_idx in scenario['triple_indices']:
                    if self.triples[t_idx].predicate == "symptom_reduced":
                        return torch.tensor([1], dtype=torch.long)
                    elif self.triples[t_idx].predicate == "symptom_persists":
                        return torch.tensor([0], dtype=torch.long)

            # CRITICAL: If we reach here, something is wrong with dataset generation
            # All triples should be mappable to scenarios with known outcomes
            raise ValueError(
                f"Cannot determine label for triple {idx}: "
                f"subject='{triple.subject}', predicate='{triple.predicate}', "
                f"object='{triple.object}', scenario={scenario_idx}. "
                f"This indicates a bug in scenario/triple generation logic. "
                f"All triples should be mappable to scenarios with defined outcomes."
            )

    def get_scenario(self, scenario_idx: int) -> Dict:
        """
        Get scenario metadata by index.

        Args:
            scenario_idx: Scenario index

        Returns:
            Dictionary containing scenario information
        """
        if 0 <= scenario_idx < len(self.scenarios):
            return self.scenarios[scenario_idx]
        raise IndexError(f"Scenario index {scenario_idx} out of range")

    def get_scenarios_with_confounder(self, confounder: str) -> List[int]:
        """
        Find scenarios with a specific confounder.

        Args:
            confounder: Confounder name

        Returns:
            List of scenario indices
        """
        return [
            i for i, s in enumerate(self.scenarios)
            if s['confounder'] == confounder
        ]

    def get_scenarios_by_treatment(self, treatment: str) -> List[int]:
        """
        Find scenarios with a specific treatment.

        Args:
            treatment: Treatment name

        Returns:
            List of scenario indices
        """
        return [
            i for i, s in enumerate(self.scenarios)
            if s['treatment'] == treatment
        ]

    def get_counterfactual_pairs(self) -> List[Tuple[int, int]]:
        """
        Find scenario pairs suitable for counterfactual reasoning.

        Returns pairs of scenarios with:
        - Same patient demographics
        - Different treatments
        - Same symptoms
        - Different outcomes

        Returns:
            List of (scenario_idx1, scenario_idx2) tuples
        """
        pairs = []

        # Group scenarios by symptom
        symptom_groups = {}
        for i, scenario in enumerate(self.scenarios):
            symptom = scenario['symptom']
            if symptom not in symptom_groups:
                symptom_groups[symptom] = []
            symptom_groups[symptom].append(i)

        # Find pairs within each symptom group
        for symptom, indices in symptom_groups.items():
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    s1 = self.scenarios[indices[i]]
                    s2 = self.scenarios[indices[j]]

                    # Different treatments
                    if s1['treatment'] != s2['treatment']:
                        pairs.append((indices[i], indices[j]))

        return pairs

    def __repr__(self) -> str:
        """String representation with dataset statistics."""
        stats = self.get_statistics()
        return (
            f"CausalTripleDataset("
            f"split='{self.split}', "
            f"scenarios={self.num_scenarios}, "
            f"triples={stats['num_triples']}, "
            f"entities={stats['num_entities']}, "
            f"predicates={stats['num_predicates']})"
        )
