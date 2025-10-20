# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Neural Symbolic Model (NSM) is a neurosymbolic language model architecture using recursive semantic triple decomposition across a mathematically-grounded 6-level hierarchy. The system makes every inference **interpretable, editable, and falsifiable** through explicit graph structures and confidence quantification.

**Current Phase**: Phase 1 Foundation Implementation (2-level hierarchy proof-of-concept)

**Linear Project**: [NSM-20](https://linear.app/imajn/issue/NSM-20) - Main implementation issue with centralized architectural blueprint in first comment

## Core Architecture

### Validated Framework (NOT Dilts)

Research conclusively demonstrated Dilts' Neurological Levels lack formal basis. **Replacement**: BDI-HTN-HRL hybrid with category-theoretic formalization.

**Six Levels** (Phase 2+):
1. Purpose/Values (BDI Desires)
2. Goals/Intentions (BDI Intentions)
3. Plans/Strategies (HTN Methods)
4. Capabilities/Skills (HRL Options)
5. Actions/Behaviors (Policy Gradients)
6. Environment/Perception (PCT + OODA)

**Phase 1 Scope**: 2-level hierarchy only (Levels 5-6: Actions/Environment)

### Key Principle: Symmetric WHY/WHAT Operations

- **WHY(level_n) = WHAT(level_n+1)** via adjoint functors (category theory)
- **WHY**: Upward abstraction (concrete → abstract) via graph pooling
- **WHAT**: Downward specification (abstract → concrete) via unpooling
- **Constraint**: `||WHY(WHAT(x)) - x||² < 0.2` (20% reconstruction error)

### Technology Stack

- **PyTorch + PyTorch Geometric**: Graph neural network framework
- **R-GCN**: Relational graph convolutions for typed semantic triples
- **Provenance Semirings**: Differentiable confidence propagation
- **SAGPool + Unpooling**: Symmetric hierarchical coarsening/refinement
- **Coupling Layers**: Invertible transformations (RealNVP)

## Development Commands

### Environment Setup

```bash
# Install core dependencies
pip install torch==2.1.0
pip install torch-geometric==2.4.0
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Utilities
pip install numpy scipy networkx matplotlib
pip install pytest pytest-cov
pip install tensorboard wandb
```

### Testing

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_symmetry.py      # WHY/WHAT inversion tests
pytest tests/test_confidence.py    # Semiring property tests
pytest tests/test_gradient.py      # Gradient flow validation

# Run with coverage
pytest --cov=nsm --cov-report=html tests/
```

### Training

```bash
# Train on synthetic dataset
python -m nsm.training.train --config configs/phase1_baseline.yaml

# Evaluate model
python -m nsm.training.evaluate --checkpoint checkpoints/best_model.pt

# Run comparative evaluation (for exploration branches)
python -m tests.evaluation_suite --dataset planning --output results/planning.json
```

## Code Organization

```
nsm/
├── data/
│   ├── triple.py          # SemanticTriple class representation
│   ├── graph.py           # Graph construction from triples
│   └── dataset.py         # PyG Dataset wrappers (BaseSemanticTripleDataset)
├── models/
│   ├── rgcn.py            # R-GCN with confidence weighting
│   ├── coupling.py        # Invertible coupling layers (RealNVP)
│   ├── pooling.py         # SAGPool + unpooling operations
│   ├── confidence.py      # Provenance semiring operations
│   └── hierarchical.py    # SymmetricHierarchicalLayer (WHY/WHAT)
├── training/
│   ├── loss.py            # Cycle consistency + task losses
│   ├── metrics.py         # Reconstruction error, calibration
│   └── trainer.py         # Training loop with gradient clipping
├── utils/
│   ├── visualization.py   # Graph plotting utilities
│   └── logging.py         # Experiment tracking
└── tests/
    ├── test_symmetry.py
    ├── test_confidence.py
    └── test_gradient.py
```

## Implementation Architecture

### Semantic Triple Representation

```python
class SemanticTriple:
    subject: Node           # Entity (e.g., "John", "System", "Goal_42")
    predicate: EdgeType     # Relation (e.g., "executes", "requires", "enables")
    object: Node            # Entity or value
    confidence: Tensor      # Learnable [0,1] weight
    level: int              # 1 (concrete) or 2 (abstract)
    metadata: Dict          # Provenance, timestamp, etc.
```

### Graph Structure

- **Nodes**: Entities/concepts at two abstraction levels (Phase 1)
  - Level 1: Concrete (actions, environmental states)
  - Level 2: Abstract (goals, capabilities)
- **Edges**: Typed relationships via R-GCN basis decomposition
  - Intra-level: Same-level connections
  - Inter-level: Cross-level connections (implements, requires)
- **Features**: Learned embeddings (64-128 dim), confidence weights

### Critical Mathematical Constraints

**Cycle Consistency Loss**:
```python
L_cycle = λ_recon * ||WHY(WHAT(x)) - x||² + λ_recon * ||WHAT(WHY(z)) - z||²
```
Target: <20% reconstruction error (λ_recon = 0.1 initially)

**Confidence Propagation** (Product Semiring):
```python
# Sequential reasoning
c_combined = c₁ * c₂ * ... * cₙ

# Alternative paths
c_aggregate = softmax_weighted_sum([c₁, c₂, ..., cₙ], temperature=τ)

# Temperature annealing: τ(epoch) = 1.0 * (0.9999)^epoch
```

## Git Workflow & Parallel Exploration

### Standard Development Workflow

**IMPORTANT**: Always create a new branch for each Linear issue and submit a PR for review.

#### For Regular Implementation Issues:

```bash
# 1. Fetch latest changes
git checkout main
git pull origin main

# 2. Create feature branch (use Linear issue ID)
git checkout -b nsm-XX-short-description

# Example: For NSM-17 (R-GCN Message Passing)
git checkout -b nsm-17-rgcn-message-passing

# 3. Implement the feature with logical commits
git add <files>
git commit -m "Add R-GCN layer with basis decomposition

- Implement ConfidenceWeightedRGCN class
- Support typed edges for semantic predicates
...

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# 4. Push branch and create PR
git push origin nsm-XX-short-description

# 5. Create PR using GitHub CLI
gh pr create --title "NSM-XX: Feature Name" \
  --body "## Summary
- Implementation details
- Key design decisions
- Integration points

## Testing
- Unit tests added
- Integration tests passing

## References
- Implements NSM-XX
- Related to NSM-YY

🤖 Generated with [Claude Code](https://claude.com/claude-code)"

# 6. After PR approval and merge, delete local branch
git checkout main
git pull origin main
git branch -d nsm-XX-short-description
```

#### Commit Message Guidelines:

- **Format**: `<Type>: <Summary>` followed by detailed body
- **Types**: Add, Update, Fix, Refactor, Test, Docs
- **Summary**: Imperative mood, concise (<72 chars)
- **Body**: Why (not what), design decisions, integration points
- **Footer**: Always include Claude Code attribution

**Example**:
```
Implement confidence propagation with product semiring

Use provenance semiring (Scallop-style) for differentiable confidence:
- Product for sequential reasoning (c₁ * c₂)
- Softmax aggregation for alternative paths
- Temperature annealing schedule

Integration points:
- Works with NSM-17 edge_attr
- Ready for NSM-14 training loop

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

### Branching Strategy for Parallel Exploration

Phase 1 uses **parallel exploration via git worktrees** to empirically validate critical design decisions:

**Week 1-2**: Dataset Domain Exploration (NSM-10)
```bash
git worktree add ../nsm-planning dataset-planning
git worktree add ../nsm-kg dataset-knowledge-graph
git worktree add ../nsm-causal dataset-causal
```

**Week 3**: Confidence Semantics Exploration (NSM-12)
```bash
git worktree add ../nsm-product confidence-product-semiring
git worktree add ../nsm-minmax confidence-minmax-semiring
git worktree add ../nsm-learned confidence-learned-aggregation
```

**Week 3-4**: Pooling Strategy Exploration (NSM-11, if needed)
```bash
git worktree add ../nsm-sagpool pooling-sagpool
git worktree add ../nsm-diffpool pooling-diffpool
git worktree add ../nsm-hybrid pooling-hybrid
```

### Evaluation Protocol

All exploration branches use **identical test suite** for fair comparison:

```bash
# In each worktree
python -m tests.evaluation_suite --dataset [name] --output results/[branch].json

# Compare results
python compare_results.py results/*.json
```

**Decision Criteria**:
- Primary: Task accuracy (40% weight)
- Calibration: Expected calibration error (20% weight)
- Multi-hop: Long reasoning chains (20% weight)
- Interpretability: Debugging/explainability (20% weight)

## Phase 1 Implementation Sequence

Strict dependency chain (see [NSM-20](https://linear.app/imajn/issue/NSM-20)):

1. **NSM-18**: PyG Environment & Data Structures (2-3d) ← START HERE
2. **NSM-17**: R-GCN Message Passing (5-7d)
3. **NSM-16**: Coupling Layers only (3-5d)
4. **NSM-15**: Base Semiring Interfaces (4-6d)
5. **NSM-10**: Dataset Exploration (3 branches, parallel)
6. **NSM-12**: Confidence Exploration (3 branches, parallel)
7. **NSM-11**: Pooling Exploration (3 branches, optional)
8. **NSM-9**: Integration Issue (coupling + pooling winner)
9. **NSM-14**: Training Loop (5-7d)
10. **NSM-13**: Validation & Comparison (5-7d)

**Total Timeline**: 4 weeks with empirical validation

## Success Criteria

### Quantitative (Phase 1)

- Reconstruction error: `||WHY(WHAT(x)) - x||²/||x||² < 0.2` (20%)
- Task accuracy: ≥95% of baseline on synthetic reasoning
- Gradient flow: Non-vanishing (norm >1e-6) at all layers
- Training stability: Monotonic loss decrease (smoothed)
- Confidence calibration: ECE <0.1

### Qualitative

- **Interpretable**: Trace reasoning from abstract goal to concrete actions
- **Symmetric**: WHAT produces plausible concrete implementations
- **Coherent**: Pooled nodes represent meaningful abstract concepts

### Performance

- Training time: <1 hour for 10K triple dataset (single GPU)
- Inference: <100ms for 1K triple graph
- Memory: <8GB GPU for batch size 32

## Common Pitfalls & Solutions

### Vanishing Gradients
- **Solution**: Add residual connections, LayerNorm, monitor gradient norms
- If <1e-6, increase learning rate or add skip connections

### GNN Oversmoothing
- **Solution**: Limit depth to 2-3 layers, use initial residual: `h_out = h_0 + layer(h_in)`

### Poor Reconstruction (>0.5 cycle loss)
- **Solution**: Increase pool_ratio, strengthen coupling layers, verify unpooling indices
- Check for information bottlenecks

### Confidence Collapse (all → 0 or → 1)
- **Solution**: Add entropy regularization, clip gradients, temperature annealing
- Initialize near 0.5, not extremes

### Memory Explosion
- **Solution**: Gradient checkpointing, reduce batch size, mixed precision (torch.cuda.amp)

## Reference Implementations

Key architectural patterns are documented in [NSM-20 main comment](https://linear.app/imajn/issue/NSM-20/nsm-phase-1-foundation-implementation-2-level-hierarchy-with-symmetric#comment-70ae40da):

- ConfidenceWeightedRGCN with basis decomposition
- SymmetricHierarchicalLayer (WHY/WHAT operations)
- Training loop template with cycle consistency

## Research Foundation

This implementation builds on validated research:

- **Graph Neural Networks**: Schlichtkrull et al. (2018) R-GCN, Fey & Lenssen (2019) PyG
- **Hierarchical Methods**: Lee et al. (2019) SAGPool, Ying et al. (2018) DiffPool
- **Confidence**: Li et al. (2023) Scallop provenance semirings
- **Theoretical**: Sutton (1999) Options Framework, Rao & Georgeff (1995) BDI, Mac Lane (1998) Adjoint Functors

## Important Notes

### Architecture Decisions Already Made

- **Hierarchy**: BDI-HTN-HRL (NOT Dilts - research invalidated it)
- **R-GCN**: Basis decomposition confirmed (70% parameter reduction)
- **Symmetry**: Adjoint functors (WHY ⊣ WHAT) - mathematically required

### Architecture Decisions Under Exploration

- **Confidence semantics**: Product vs MinMax vs Learned (NSM-12)
- **Pooling strategy**: SAGPool vs DiffPool vs Hybrid (NSM-11)
- **Dataset domains**: Planning vs Knowledge Graph vs Causal (NSM-10)

**Do NOT commit to specific approaches until exploration branches evaluated.**

### When Working on Sub-Issues

1. Always reference [NSM-20 main comment](https://linear.app/imajn/issue/NSM-20#comment-70ae40da) for architectural blueprint
2. Check dependencies complete before starting
3. Use identical test suite across exploration branches
4. Document findings in `results/BRANCH_NAME_report.md`
5. Update parent issue (NSM-20) when complete

### Testing Requirements

All new components must include:
- **Unit tests**: Symmetry, invertibility, gradient flow
- **Integration tests**: End-to-end pipeline, batch processing
- **Validation metrics**: Reconstruction error, task accuracy

### Documentation Standards

- **Docstrings**: Google style with mathematical foundations
- **Inline comments**: Explain non-obvious math/design choices
- **Type hints**: All function signatures
- **Examples**: Usage patterns in docstrings

## Hardware Requirements

**Minimum**: 8GB VRAM (RTX 3070 / V100), 16GB RAM
**Recommended**: 24GB VRAM (RTX 3090 / A5000), 32GB RAM

## Additional Resources

- **Linear Project**: [NSM Project](https://linear.app/imajn/project/neural-symbolic-model-nsm-hierarchical-semantic-reasoning-architecture-af57a8ece32c)
- **Main Issue**: [NSM-20](https://linear.app/imajn/issue/NSM-20) with complete architectural blueprint
- **Research Issues**: NSM-2, NSM-3, NSM-4, NSM-5, NSM-6 (foundations)

## Linear Issue Mapping

For reference, the Linear issues were renumbered from IMA-* to NSM-* format:

- **NSM-20** (was IMA-137): Phase 1 Foundation Implementation - Main Issue
- **NSM-18** (was IMA-138): PyG Environment & Data Structures
- **NSM-17** (was IMA-139): R-GCN Message Passing
- **NSM-16** (was IMA-140): Coupling Layers
- **NSM-15** (was IMA-141): Base Semiring Interfaces
- **NSM-14** (was IMA-142): Training Loop
- **NSM-13** (was IMA-143): Validation & Comparison
- **NSM-12** (was IMA-144): Confidence Exploration
- **NSM-11** (was IMA-145): Pooling Exploration
- **NSM-10** (was IMA-146): Dataset Exploration
- **NSM-9** (was IMA-147): Integration Issue
- **NSM-6** (was IMA-131): Research - BDI-HTN-HRL Framework
- **NSM-5** (was IMA-134): Research - Adjoint Functors
- **NSM-4** (was IMA-133): Research - Provenance Semirings
- **NSM-3** (was IMA-135): Research - PyG Architecture
- **NSM-2** (was IMA-132): Research - Dilts Analysis
