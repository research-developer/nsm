# Two-Week Sprint Plan: External Review Readiness

**Goal**: Transform NSM from "early prototype" to "share-ready research demo"
**Timeline**: 14 days
**Estimated Effort**: 1 person, full-time
**Modal Compute Budget**: ~$100

---

## Week 1: Scientific Validation (Days 1-7)

### Day 1-2: Fix Multi-Seed Validation ⚠️ CRITICAL

**Problem**: Only seed 42 completed successfully (66.43%). Cannot claim results without statistical significance.

**Tasks**:
1. Debug why seeds 123, 456, 789, 1011 failed
   - Check Modal logs for timeout vs. crash vs. OOM
   - Likely issue: Dataset size variation or batch size
   - Fix: Add error handling, reduce batch size if needed

2. Run 3-seed minimum validation
   ```bash
   # Sequential to debug issues
   modal run experiments/modal_10x_baseline.py --seed 42  # Already done
   modal run experiments/modal_10x_baseline.py --seed 123
   modal run experiments/modal_10x_baseline.py --seed 456
   ```

3. Create results aggregation script
   ```python
   # scripts/aggregate_multi_seed.py
   import json
   import numpy as np
   from pathlib import Path

   def aggregate_results():
       results = []
       for seed in [42, 123, 456]:
           path = f"checkpoints/nsm-10x-baseline-seed{seed}_results.json"
           if Path(path).exists():
               with open(path) as f:
                   results.append(json.load(f))

       accuracies = [r['best_val_accuracy'] for r in results]
       print(f"Mean: {np.mean(accuracies):.4f}")
       print(f"Std:  {np.std(accuracies):.4f}")
       print(f"Results: {accuracies}")
   ```

**Deliverable**:
- `MULTI_SEED_RESULTS.md` with table:
  ```
  | Seed | Best Epoch | Val Accuracy | q_neural | Notes |
  |------|------------|--------------|----------|-------|
  | 42   | 11         | 66.43%       | 0.472    | ✓     |
  | 123  | ?          | ?            | ?        | ?     |
  | 456  | ?          | ?            | ?        | ?     |
  | Mean | -          | XX.XX ± Y.YY | -        | -     |
  ```

**Success Criterion**: ≥3 seeds complete with std < 5%

**Time**: 16 hours (2 days x 8 hours)
**Cost**: ~$30 Modal credits (3 full training runs)

---

### Day 3-4: Implement Baseline Comparisons ⚠️ CRITICAL

**Problem**: 66% accuracy is meaningless without context. Simple baseline might beat us.

**Tasks**:
1. Implement 3 baselines in `experiments/baselines.py`

   **Baseline 1: Vanilla RGCN (No Hierarchy)**
   ```python
   class SimpleRGCN(nn.Module):
       """Just message passing + pooling, no WHY/WHAT operations"""
       def __init__(self, node_features, num_relations, num_classes):
           super().__init__()
           self.conv1 = RGCNConv(node_features, 128, num_relations)
           self.conv2 = RGCNConv(128, 64, num_relations)
           self.fc = nn.Linear(64, num_classes)

       def forward(self, x, edge_index, edge_type, batch):
           x = F.relu(self.conv1(x, edge_index, edge_type))
           x = F.relu(self.conv2(x, edge_index, edge_type))
           x = global_mean_pool(x, batch)
           return self.fc(x)
   ```

   **Baseline 2: Graph Mean Pooling + MLP**
   ```python
   class GraphMLP(nn.Module):
       """Simplest possible: average node features → MLP"""
       def __init__(self, node_features, num_classes):
           super().__init__()
           self.fc1 = nn.Linear(node_features, 128)
           self.fc2 = nn.Linear(128, 64)
           self.fc3 = nn.Linear(64, num_classes)

       def forward(self, x, edge_index, edge_type, batch):
           x = global_mean_pool(x, batch)  # Ignore graph structure!
           x = F.relu(self.fc1(x))
           x = F.relu(self.fc2(x))
           return self.fc3(x)
   ```

   **Baseline 3: Standard GCN (Untyped Edges)**
   ```python
   class SimpleGCN(nn.Module):
       """GCN ignoring edge types"""
       def __init__(self, node_features, num_classes):
           super().__init__()
           self.conv1 = GCNConv(node_features, 128)
           self.conv2 = GCNConv(128, 64)
           self.fc = nn.Linear(64, num_classes)

       def forward(self, x, edge_index, edge_type, batch):
           x = F.relu(self.conv1(x, edge_index))
           x = F.relu(self.conv2(x, edge_index))
           x = global_mean_pool(x, batch)
           return self.fc(x)
   ```

2. Train each baseline (1 seed is enough for comparison)
   ```bash
   modal run experiments/baselines.py::train_simple_rgcn --seed 42
   modal run experiments/baselines.py::train_graph_mlp --seed 42
   modal run experiments/baselines.py::train_simple_gcn --seed 42
   ```

3. Compare parameter counts
   ```python
   def count_parameters(model):
       return sum(p.numel() for p in model.parameters())

   # NSM 6-level: 173,374 parameters
   # Simple RGCN: ~XX,XXX parameters (probably less)
   # Graph MLP: ~XX,XXX parameters
   ```

**Deliverable**:
- `BASELINE_COMPARISON.md` with table:
  ```
  | Model          | Params  | Accuracy | Advantage | Notes             |
  |----------------|---------|----------|-----------|-------------------|
  | Graph MLP      | ~50K    | XX.X%    | -        | No structure      |
  | Simple GCN     | ~80K    | XX.X%    | -        | No edge types     |
  | Simple RGCN    | ~120K   | XX.X%    | -        | No hierarchy      |
  | NSM 6-level    | 173K    | 66.4%    | +X.X%    | Ours (p<0.05?)    |
  ```

**Success Criterion**: NSM beats all baselines by ≥2% (statistically significant)

**Risk**: If baselines win, need to understand why and pivot framing

**Time**: 16 hours (debugging, training, analysis)
**Cost**: ~$20 Modal credits (3 baseline runs)

---

### Day 5-7: Create Interpretability Demonstrations ⚠️ CRITICAL

**Problem**: Core claim is "interpretable reasoning" but zero visualizations exist.

**Tasks**:

**Day 5: Extract Reasoning Traces**

1. Create trace extraction script
   ```python
   # scripts/extract_reasoning_trace.py
   import torch
   from nsm.models.chiral import FullChiralModel
   from nsm.utils.checkpoint_manager import load_nsm_checkpoint
   import networkx as nx
   import matplotlib.pyplot as plt

   def extract_trace(model, graph, max_nodes=20):
       """Extract hierarchical reasoning trace from input to prediction"""
       model.eval()

       with torch.no_grad():
           # Forward pass through all 6 levels
           x_l1 = model.left_trifold.level1(graph.x, graph.edge_index, graph.edge_type)
           x_l2 = model.left_trifold.level2(x_l1, ...)
           x_l3 = model.left_trifold.level3(x_l2, ...)

           # Pool to get representative nodes at each level
           top_nodes_l1 = torch.topk(x_l1.norm(dim=1), k=min(10, len(x_l1))).indices
           top_nodes_l2 = torch.topk(x_l2.norm(dim=1), k=min(5, len(x_l2))).indices
           # ... etc

           return {
               'level_1_nodes': top_nodes_l1,
               'level_2_nodes': top_nodes_l2,
               'level_3_nodes': top_nodes_l3,
               'prediction': model(graph.x, graph.edge_index, graph.edge_type, graph.batch),
               'attention_weights': ...,  # If available
           }
   ```

2. Create 5 example traces from validation set
   - 2 correct predictions (high confidence)
   - 2 correct predictions (low confidence)
   - 1 incorrect prediction (for honesty)

**Day 6: Visualize Hierarchical Structure**

3. Create visualization script
   ```python
   # scripts/visualize_trace.py
   def visualize_reasoning_trace(trace, save_path):
       """Create multi-level graph visualization"""
       fig, axes = plt.subplots(2, 3, figsize=(18, 12))

       # Level 1: Environment/Perception (bottom)
       plot_graph_level(axes[1, 0], trace['level_1'], title="L1: Actions/Environment")

       # Level 2: Actions/Behaviors
       plot_graph_level(axes[1, 1], trace['level_2'], title="L2: Actions")

       # ... up to Level 6: Purpose/Values
       plot_graph_level(axes[0, 2], trace['level_6'], title="L6: Purpose/Values")

       # Add arrows showing WHY/WHAT flow
       add_flow_arrows(axes)

       plt.savefig(save_path, dpi=300, bbox_inches='tight')
   ```

4. Generate visualizations for all 5 examples
   - Save as: `results/trace_example_{1-5}.png`

**Day 7: Create Narrative Walkthrough**

5. Write detailed interpretation for each example
   ```markdown
   # Example 1: Correct High-Confidence Prediction

   **Input**: Graph with 47 nodes representing planning state
   **Prediction**: Class 1 (confidence: 0.94)
   **Ground Truth**: Class 1 ✓

   ## Reasoning Trace

   ### Level 6 (Purpose): Top 3 Activated Nodes
   - Node 42: "Goal Achievement" (activation: 0.87)
   - Node 15: "Resource Optimization" (activation: 0.62)
   - Node 33: "Constraint Satisfaction" (activation: 0.54)

   **Interpretation**: Model identifies this as goal-oriented planning

   ### WHY→WHAT Flow
   L6 "Goal Achievement" abstracts to...
   → L5 "Sequential Planning" which decomposes to...
   → L4 "Action Sequencing" which implements as...
   → L3 "Resource Allocation" which executes as...
   → L2 "Primitive Actions" which observes...
   → L1 "Environmental State"

   ### Key Insight
   The model correctly identifies the hierarchical structure of the planning
   problem. High confidence stems from consistent activation across all levels.
   ```

6. Create `INTERPRETABILITY_DEMO.md` with all 5 examples

**Deliverable**:
- 5 visualization PNGs showing hierarchical reasoning
- `INTERPRETABILITY_DEMO.md` with narratives
- Script that others can run: `python scripts/visualize_trace.py --checkpoint checkpoints/nsm-10x-baseline_best.pt --example 1`

**Success Criterion**: Someone with ML background can look at visualizations and understand what the model is doing

**Time**: 24 hours (3 days x 8 hours)
**Cost**: $0 (inference only)

---

## Week 2: Documentation & Packaging (Days 8-14)

### Day 8-9: Update Documentation

**Problem**: Documentation contradicts reality (Phase 1 vs. NSM-33 work)

**Tasks**:

1. **Update README.md**
   ```markdown
   # Neural Symbolic Model (NSM)

   Hierarchical graph neural network with interpretable reasoning via symmetric
   abstraction/concretization operations.

   ## Current Status: NSM-33 Validation Complete ✓

   - **Best Accuracy**: 66.43 ± X.XX% (3-seed validation)
   - **Architecture**: 6-level chiral dual-trifold (173K parameters)
   - **Dataset**: Planning task with 20K training samples
   - **Novel Contribution**: Physics-inspired training stability metrics

   ## Quick Start

   ### Installation
   ```bash
   pip install torch==2.1.0 torch-geometric==2.4.0
   git clone https://github.com/research-developer/nsm.git
   cd nsm
   pip install -e .
   ```

   ### Run Demo
   ```bash
   # Visualize reasoning trace on example
   python scripts/demo.py --example 1

   # Train from scratch (requires Modal account)
   modal run experiments/modal_10x_baseline.py
   ```

   ## Architecture Overview

   [Insert simple diagram showing 6 levels with WHY/WHAT arrows]

   ## Key Results

   | Model          | Accuracy | Params | Interpretable |
   |----------------|----------|--------|---------------|
   | Simple RGCN    | XX.X%    | 120K   | ✗             |
   | NSM 6-level    | 66.4%    | 173K   | ✓             |
   | Improvement    | +X.X%    | -      | Unique        |

   ## Novel Contributions

   1. **Symmetric Hierarchical Operations**: WHY/WHAT as category-theoretic adjoints
   2. **Physics-Inspired Metrics**: Borrowed from plasma fusion (q_neural safety factor)
   3. **Interpretable Reasoning**: Explicit traces through 6-level hierarchy

   ## Documentation

   - [Two-Week Sprint Results](TWO_WEEK_SPRINT_RESULTS.md)
   - [Interpretability Demo](INTERPRETABILITY_DEMO.md)
   - [Baseline Comparisons](BASELINE_COMPARISON.md)
   - [Multi-Seed Validation](MULTI_SEED_RESULTS.md)

   ## Project History

   - **NSM-32**: 6-level architecture development
   - **NSM-33**: 10x dataset scaling, physics metrics (85.7% collapse prediction)
   - **NSM-34**: Checkpoint infrastructure, CGT investigation (negative result)

   ## Citation

   If you use this work, please cite:
   ```bibtex
   @software{nsm2025,
     title={Neural Symbolic Model: Interpretable Hierarchical Reasoning},
     author={[Your Name]},
     year={2025},
     url={https://github.com/research-developer/nsm}
   }
   ```
   ```

2. **Update CLAUDE.md** to match current state
   - Change "Phase 1: 2-level hierarchy" → "Phase 1.5: 6-level validation"
   - Add NSM-33 and NSM-34 to timeline
   - Document CGT investigation as completed (negative result)

3. **Create FAQ.md** for anticipated questions
   ```markdown
   # Frequently Asked Questions

   ## Why not use transformers?

   Transformers lack explicit hierarchical structure and interpretable reasoning
   traces. NSM provides provable symmetry (WHY∘WHAT ≈ id) via category theory.

   ## What's the "planning task"?

   Binary classification of planning problem instances from [dataset paper].
   Task: Predict if plan will succeed given initial state and constraints.
   Random baseline: 50%, Simple RGCN: XX%, NSM: 66.4%

   ## How do you ensure interpretability?

   Every prediction traces through 6 levels with explicit node activations.
   See INTERPRETABILITY_DEMO.md for 5 concrete examples.

   ## What are "physics-inspired metrics"?

   We borrowed q_neural (safety factor) from plasma fusion physics to predict
   training collapse. Achieved 85.7% accuracy in NSM-33 validation.

   ## What didn't work?

   Combinatorial Game Theory operators (NSM-34). Conway temperature was
   invariant (0.0000) across all epochs. Root cause: implementation flaw
   (deterministic operations). See PR #12 for details.

   ## Is this production-ready?

   No. This is a research prototype demonstrating novel ideas. Not optimized
   for deployment.
   ```

4. **Document Planning Dataset**
   ```markdown
   # Dataset Description

   ## Planning Triple Dataset

   **Source**: Synthetic generation based on PDDL-like planning formalism
   **Task**: Binary classification (plan feasible vs. infeasible)
   **Size**:
   - Training: 16,000 problems (80%)
   - Validation: 4,000 problems (20%)

   ## Graph Structure

   **Nodes**: Represent states, actions, and goals (avg: 47 nodes/graph)
   **Edges**: Typed relations (17 types):
   - precondition, effect, requires, enables, conflicts, ...
   **Node Features**: 64-dim learned embeddings

   ## Task Difficulty

   **Random Baseline**: 50% (balanced classes)
   **Simple MLP**: ~XX% (ignoring graph structure)
   **Simple RGCN**: ~XX% (no hierarchy)
   **NSM 6-level**: 66.4% (interpretable)

   ## Example Problem

   [Add simple visualization of one problem]
   ```

**Deliverable**:
- Updated README.md reflecting current state
- Updated CLAUDE.md matching reality
- FAQ.md addressing anticipated questions
- DATASET.md describing task clearly

**Time**: 16 hours (2 days x 8 hours)

---

### Day 10: Create Two-Page Summary

**Problem**: Need concise overview for busy researchers

**Tasks**:

1. Write `NSM_RESEARCH_SUMMARY.pdf` (2 pages max)

   **Page 1: Overview + Architecture**
   ```
   [Title] Neural Symbolic Model: Interpretable Hierarchical Reasoning

   [Abstract - 100 words]
   We present NSM, a 6-level graph neural network where abstraction (WHY)
   and concretization (WHAT) are symmetric operations proven via category
   theory. Novel physics-inspired metrics predict training collapse with
   85% accuracy. Achieves 66.4% accuracy on planning tasks with full
   interpretability - every prediction traces through explicit reasoning
   hierarchy.

   [Diagram: 6-level architecture with WHY/WHAT arrows]

   [Key Innovation bullets]
   - Symmetric hierarchical operations (adjoint functors)
   - Physics-inspired stability monitoring (q_neural from fusion)
   - Explicit interpretable reasoning traces

   [Results Table]
   | Model       | Acc   | Params | Interp |
   |-------------|-------|--------|--------|
   | Simple RGCN | XX.X% | 120K   | ✗      |
   | NSM 6-level | 66.4% | 173K   | ✓      |
   ```

   **Page 2: Results + Next Steps**
   ```
   [Figure: Example reasoning trace visualization]

   [Multi-Seed Results]
   Seed 42: 66.43%, Seed 123: XX.X%, Seed 456: XX.X%
   Mean: XX.XX ± Y.YY% (statistically significant improvement)

   [Interpretability Example - 50 words]
   Model identifies hierarchical structure: L6 "Goal Achievement" →
   L5 "Sequential Planning" → ... → L1 "Environmental State".
   High confidence stems from consistent activation across levels.

   [Limitations]
   - Synthetic dataset (not real-world planning)
   - Modest absolute accuracy (66% vs. potential 100%)
   - Requires PyTorch Geometric (deployment friction)

   [Next Steps]
   - Real-world benchmark evaluation
   - Scaling to larger models
   - Application to code reasoning / robotics planning

   [Contact]: [Your Email]
   [Code]: github.com/research-developer/nsm
   ```

**Deliverable**: `NSM_RESEARCH_SUMMARY.pdf` (2 pages, figures included)

**Time**: 8 hours

---

### Day 11-12: Build Standalone Demo Script

**Problem**: External reviewers can't easily run Modal-based experiments

**Tasks**:

1. Create `scripts/standalone_demo.py`
   ```python
   #!/usr/bin/env python3
   """
   Standalone NSM Demo - No Modal Required

   Downloads pre-trained checkpoint and runs inference on example graphs.
   Shows interpretable reasoning traces.

   Usage:
       python scripts/standalone_demo.py --example 1
       python scripts/standalone_demo.py --interactive
   """

   import torch
   import requests
   from pathlib import Path
   import matplotlib.pyplot as plt

   def download_checkpoint(url, path):
       """Download pre-trained checkpoint from GitHub releases"""
       if not Path(path).exists():
           print(f"Downloading checkpoint from {url}...")
           response = requests.get(url)
           Path(path).parent.mkdir(parents=True, exist_ok=True)
           with open(path, 'wb') as f:
               f.write(response.content)
           print("✓ Download complete")

   def load_model(checkpoint_path):
       """Load pre-trained NSM model"""
       from nsm.models.chiral import FullChiralModel

       checkpoint = torch.load(checkpoint_path, map_location='cpu')
       model = FullChiralModel(
           node_features=64,
           num_relations=17,
           num_classes=2,
           pool_ratio=0.5,
           task_type='classification',
           dropout=0.1
       )
       model.load_state_dict(checkpoint['model_state_dict'])
       model.eval()
       return model

   def run_example(model, example_id):
       """Run inference on pre-loaded example"""
       # Load example from data/examples/
       graph = torch.load(f'data/examples/example_{example_id}.pt')

       # Extract reasoning trace
       trace = extract_trace(model, graph)

       # Visualize
       visualize_trace(trace, save_path=f'results/demo_trace_{example_id}.png')

       # Print interpretation
       print_interpretation(trace)

   def interactive_mode(model):
       """Interactive exploration of reasoning traces"""
       print("Interactive NSM Demo")
       print("Commands: example <N>, quit")

       while True:
           cmd = input("> ").strip()
           if cmd == "quit":
               break
           elif cmd.startswith("example "):
               example_id = int(cmd.split()[1])
               run_example(model, example_id)

   if __name__ == "__main__":
       import argparse

       parser = argparse.ArgumentParser()
       parser.add_argument('--example', type=int, help='Run specific example')
       parser.add_argument('--interactive', action='store_true')
       parser.add_argument('--checkpoint', default='checkpoints/nsm-10x-baseline_best.pt')

       args = parser.parse_args()

       # Download checkpoint if needed (from GitHub releases)
       CHECKPOINT_URL = "https://github.com/research-developer/nsm/releases/download/v0.1/nsm-10x-baseline_best.pt"
       download_checkpoint(CHECKPOINT_URL, args.checkpoint)

       # Load model
       model = load_model(args.checkpoint)

       if args.interactive:
           interactive_mode(model)
       elif args.example:
           run_example(model, args.example)
       else:
           print("Running default examples...")
           for i in range(1, 6):
               run_example(model, i)
   ```

2. Package example graphs
   ```bash
   # Create data/examples/ with 5 pre-loaded graphs
   python scripts/package_examples.py
   ```

3. Upload checkpoint to GitHub Release
   ```bash
   # Create v0.1 release with checkpoint file
   gh release create v0.1 \
     checkpoints/nsm-10x-baseline_best.pt \
     --title "NSM v0.1 - Initial Release" \
     --notes "Pre-trained 6-level model (66.4% accuracy)"
   ```

4. Test standalone script works
   ```bash
   # Fresh conda environment test
   conda create -n nsm-test python=3.10
   conda activate nsm-test
   pip install torch==2.1.0 torch-geometric==2.4.0
   python scripts/standalone_demo.py --example 1
   # Should work without Modal
   ```

**Deliverable**:
- `scripts/standalone_demo.py` (fully functional)
- `data/examples/` with 5 pre-loaded graphs
- GitHub release v0.1 with checkpoint
- `STANDALONE_DEMO.md` with usage instructions

**Time**: 16 hours (2 days x 8 hours)

---

### Day 13-14: Create Hero Figure & Final Package

**Problem**: Need one compelling visual + polished presentation

**Tasks**:

**Day 13: Create Hero Figure**

1. Design comprehensive figure showing:
   - **Panel A**: 6-level architecture diagram
   - **Panel B**: Example reasoning trace (one of our 5 examples)
   - **Panel C**: Results comparison (NSM vs. baselines bar chart)
   - **Panel D**: Physics metrics (q_neural over epochs, showing prediction)

2. Create in presentation-quality tool
   ```python
   # scripts/create_hero_figure.py
   import matplotlib.pyplot as plt
   from matplotlib.gridspec import GridSpec

   def create_hero_figure():
       fig = plt.figure(figsize=(16, 10))
       gs = GridSpec(2, 2, figure=fig)

       # Panel A: Architecture
       ax1 = fig.add_subplot(gs[0, 0])
       plot_architecture_diagram(ax1)
       ax1.set_title('A) NSM Architecture', fontsize=14, fontweight='bold')

       # Panel B: Reasoning Trace
       ax2 = fig.add_subplot(gs[0, 1])
       plot_reasoning_trace(ax2, example_id=1)
       ax2.set_title('B) Interpretable Reasoning', fontsize=14, fontweight='bold')

       # Panel C: Results
       ax3 = fig.add_subplot(gs[1, 0])
       plot_results_comparison(ax3)
       ax3.set_title('C) Benchmark Comparison', fontsize=14, fontweight='bold')

       # Panel D: Physics Metrics
       ax4 = fig.add_subplot(gs[1, 1])
       plot_physics_metrics(ax4)
       ax4.set_title('D) Training Stability Prediction', fontsize=14, fontweight='bold')

       plt.tight_layout()
       plt.savefig('results/NSM_HERO_FIGURE.png', dpi=300, bbox_inches='tight')
       plt.savefig('results/NSM_HERO_FIGURE.pdf', bbox_inches='tight')
   ```

3. Generate figure
   ```bash
   python scripts/create_hero_figure.py
   # Output: results/NSM_HERO_FIGURE.png (for slides)
   # Output: results/NSM_HERO_FIGURE.pdf (for paper)
   ```

**Day 14: Final Packaging & QA**

4. Create sprint completion checklist
   ```markdown
   # Sprint Completion Checklist

   ## Week 1: Scientific Validation
   - [ ] Multi-seed validation (≥3 seeds, std < 5%)
   - [ ] Baseline comparisons (NSM beats all by ≥2%)
   - [ ] 5 interpretability examples with visualizations

   ## Week 2: Documentation
   - [ ] README.md updated to match reality
   - [ ] CLAUDE.md aligned with current state
   - [ ] FAQ.md addresses common questions
   - [ ] DATASET.md describes task clearly

   ## Deliverables
   - [ ] NSM_RESEARCH_SUMMARY.pdf (2 pages)
   - [ ] Standalone demo script works without Modal
   - [ ] Hero figure (PNG + PDF)
   - [ ] All results documented in results/

   ## GitHub
   - [ ] Release v0.1 with checkpoint uploaded
   - [ ] All markdown files committed
   - [ ] Code is clean and commented

   ## Ready to Share?
   - [ ] Can answer "what problem does this solve?"
   - [ ] Can defend accuracy claims with statistics
   - [ ] Can show concrete interpretability example
   - [ ] Can run demo for someone in <5 minutes
   ```

5. Run full quality check
   ```bash
   # Test all scripts work
   python scripts/standalone_demo.py --example 1
   python scripts/visualize_trace.py --checkpoint checkpoints/nsm-10x-baseline_best.pt
   python scripts/aggregate_multi_seed.py

   # Check documentation
   grep -r "Phase 1: 2-level" .  # Should return nothing
   grep -r "TODO" .  # Address any TODOs

   # Verify results files exist
   ls results/
   # Should have:
   # - NSM_HERO_FIGURE.png
   # - NSM_HERO_FIGURE.pdf
   # - trace_example_1.png (through 5)
   # - MULTI_SEED_RESULTS.md
   # - BASELINE_COMPARISON.md
   # - INTERPRETABILITY_DEMO.md
   ```

6. Create final summary document
   ```markdown
   # Two-Week Sprint Results

   **Dates**: [Start] - [End]
   **Goal**: Make NSM share-ready for external review
   **Status**: ✓ Complete

   ## What We Accomplished

   ### Scientific Rigor
   ✅ Multi-seed validation (3 seeds, mean: XX.XX ± Y.YY%)
   ✅ Baseline comparisons (NSM beats all by X.X%)
   ✅ Interpretability demonstrations (5 concrete examples)
   ✅ Task documentation (planning dataset fully described)

   ### Documentation Quality
   ✅ README matches current state
   ✅ 2-page research summary created
   ✅ FAQ addresses anticipated questions
   ✅ Hero figure shows key contributions

   ### Accessibility
   ✅ Standalone demo script (no Modal required)
   ✅ Pre-trained checkpoint on GitHub release
   ✅ 5-minute demo workflow established

   ## Key Results

   [Insert hero figure]

   **Main Finding**: NSM achieves 66.4 ± Y.Y% accuracy on planning task,
   beating simple baselines by X.X% while providing full interpretability
   via explicit 6-level reasoning traces.

   **Novel Contribution**: Physics-inspired q_neural metric predicts training
   collapse with 85.7% accuracy (NSM-33 validation).

   **Honest Limitations**:
   - Synthetic dataset (not real-world planning yet)
   - Modest absolute accuracy (room for improvement)
   - Requires PyTorch Geometric (deployment friction)

   ## Ready to Share

   **Recommended First Contact**: SSI or SoftMax (smaller orgs, early-stage work)

   **Conversation Starter**:
   > "We built hierarchical GNNs with symmetric abstraction/concretization
   > (via category theory). Interesting bit: borrowed plasma physics metrics
   > to predict training collapse (85% accuracy). Also tried game theory -
   > total failure, but interesting failure. Would love your thoughts on
   > [specific question relevant to their work]."

   **Demo Flow** (5 minutes):
   1. Show hero figure (1 min)
   2. Run standalone demo (2 min)
   3. Walk through one reasoning trace (2 min)

   ## What's Next

   **If feedback is positive**:
   - Evaluate on real-world benchmark (bAbI, CLEVR, etc.)
   - Scale to larger models
   - Develop Anthropic pitch (alignment angle)

   **If feedback identifies gaps**:
   - Address specific concerns
   - Iterate before wider sharing

   ## Files to Share

   Core package:
   - NSM_RESEARCH_SUMMARY.pdf (2-page overview)
   - NSM_HERO_FIGURE.png (key visual)
   - Link to GitHub repo
   - Link to standalone demo

   Optional (if they want details):
   - MULTI_SEED_RESULTS.md
   - BASELINE_COMPARISON.md
   - INTERPRETABILITY_DEMO.md
   ```

**Deliverable**:
- `results/NSM_HERO_FIGURE.png` (presentation-quality)
- `results/NSM_HERO_FIGURE.pdf` (publication-quality)
- `TWO_WEEK_SPRINT_RESULTS.md` (comprehensive summary)
- Completed checklist (all items checked)

**Time**: 16 hours (2 days x 8 hours)

---

## Cost & Resource Summary

**Total Time**: 14 days (1 person full-time = 112 hours)

**Modal Compute Costs**:
- Multi-seed validation: 3 runs × $10 = $30
- Baseline comparisons: 3 runs × $7 = $21
- Buffer for failures/reruns: $49
- **Total: ~$100**

**Required Skills**:
- Python/PyTorch (moderate)
- Matplotlib/visualization (basic)
- Technical writing (moderate)
- LaTeX/figure design (basic)

**External Dependencies**:
- Modal account (for training)
- GitHub account (for releases)
- LaTeX/Inkscape (for hero figure - optional, can use Python)

---

## Success Metrics

**Minimum Viable Demo** (must achieve):
- [ ] ≥3 seeds complete with std < 5%
- [ ] NSM beats all baselines by ≥2%
- [ ] 5 interpretability examples with visualizations
- [ ] Standalone demo runs in <5 minutes

**Share-Ready Package** (goal):
- [ ] 2-page summary is clear to non-experts
- [ ] Hero figure tells the story at a glance
- [ ] Can answer "what problem?" in one sentence
- [ ] No embarrassing gaps in anticipated questions

**Confidence to Share**:
- [ ] Would not waste their time
- [ ] Have defensible claims
- [ ] Can demo in real-time
- [ ] Honest about limitations

---

## Risk Mitigation

**If multi-seed experiments fail again**:
- Debug timeout issues (increase timeout, reduce batch size)
- Fall back to 2 seeds if necessary (acknowledge limitation)
- Emphasize single-seed result consistency

**If baselines beat NSM**:
- Investigate why (architecture issue? hyperparameters?)
- Pivot framing: "interpretability with competitive accuracy"
- Be honest: "baselines win on accuracy, we win on interpretability"

**If interpretability visualizations are nonsense**:
- Debug what each level actually learns
- May need to retrain with interpretability constraints
- Worst case: pivot to "physics metrics" as main contribution

**If we run out of time**:
- Prioritize: Multi-seed > Baselines > Interpretability > Documentation
- Can share incomplete package with "work in progress" framing
- Better to wait an extra week than share too early

---

## Next Steps After Sprint

**If sprint succeeds**:
1. Share with SSI/SoftMax contact
2. Collect feedback
3. Iterate based on input
4. Consider Anthropic if feedback is positive

**If sprint reveals fundamental issues**:
1. Document learnings
2. Decide: pivot vs. persist
3. May need month-long effort instead of 2 weeks

**Long-term (3-6 months)**:
- Real-world benchmark evaluation
- Publication submission (NeurIPS, ICLR)
- Deployment case study

---

## Daily Standup Template

Use this to track progress:

```markdown
# Day X Progress

## Completed Today
- [ ] Task 1
- [ ] Task 2

## Blocked On
- Issue 1: [description]

## Tomorrow's Plan
- [ ] Task 3
- [ ] Task 4

## Risks/Questions
- Concern 1
- Question 2
```

---

## Final Thoughts

This sprint is ambitious but achievable. The key is maintaining focus on the core question: **"Would sharing this waste someone's time?"**

After 2 weeks, you should have a compelling demo that:
1. Makes defensible scientific claims (multi-seed validation)
2. Shows clear value (beats baselines, provides interpretability)
3. Can be experienced in 5 minutes (standalone demo)
4. Acknowledges limitations honestly (FAQ, limitations section)

That's the difference between "interesting research prototype" and "half-baked work." The foundation is solid - we just need to package it properly.

**Let's make NSM share-worthy! 🚀**
