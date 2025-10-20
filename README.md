# Neural Symbolic Model (NSM)

Hierarchical Semantic Reasoning Architecture

## Overview

NSM is a neurosymbolic language model where reasoning emerges from **recursive semantic triple decomposition** across a mathematically-grounded hierarchy. Unlike opaque transformer architectures, NSM makes every inference interpretable, editable, and falsifiable through explicit graph structures and confidence quantification.

## Core Concept

Language decomposes into nested subject-predicate-object triples that distill to primitive operators and values. Each triple exists at one abstraction level, with **symmetric WHY/WHAT operators** enabling bidirectional reasoning:

- **WHY**: Climbs the hierarchy (abstraction)
- **WHAT**: Descends the hierarchy (specification)

The mathematical foundation uses adjoint functors from category theory to ensure WHY and WHAT are symmetric inverses, with information-theoretic pruning to eliminate redundant knowledge.

## Architecture

### Hierarchy Framework (BDI-HTN-HRL)

Six semantic levels:

1. **Purpose/Values** - Mission-level utility functions
2. **Goals/Intentions** - Desired end states
3. **Plans/Strategies** - Action sequences
4. **Capabilities/Skills** - Reusable operation triples
5. **Actions/Behaviors** - Primitive operations
6. **Environment/Perception** - Sensorimotor grounding

**Property**: `WHY(level_n) = WHAT(level_n+1)` via adjoint functor construction

### Key Components

- **Symbolic Layer**: Semantic graph with typed edges (R-GCN), level tags, confidence values
- **Neural Layer**: Learnable confidence tensors (provenance semirings), message passing with gradient flow
- **Training**: Cycle consistency loss `||WHY(WHAT(x))-x||²`, information-theoretic pruning (80-85% sparsification)

## Current Phase: Phase 1 Foundation

**Goal**: Proof-of-concept 2-level hierarchy demonstrating symmetric operations

**Timeline**: 4 weeks

**Success Criteria**:
- <20% reconstruction error (WHY/WHAT cycle consistency)
- ≥80% task accuracy across multiple domains
- Interpretable reasoning traces
- Production-ready infrastructure for Phase 2 scaling

**Implementation**: PyTorch Geometric with provenance semirings for confidence propagation

## Advantages

- **Interpretable**: Trace reasoning to purpose
- **Editable**: Modify triples without retraining
- **Compositional**: Explicit logical structure
- **Continual Learning**: Add knowledge without catastrophic forgetting
- **Falsifiable**: Mathematical utility measurement

## Research Foundation

Built on validated AI research:
- Options Framework (Sutton 1999)
- Graph Networks (Battaglia 2018)
- Scallop (Li 2023)
- Category Theory (Mac Lane 1998)
- BDI Architecture (Rao & Georgeff 1995)

## Project Status

Phase 1 implementation in progress. See [Linear Project](https://linear.app/imajn/project/neural-symbolic-model-nsm-hierarchical-semantic-reasoning-architecture-af57a8ece32c) for detailed roadmap.

## License

TBD
