# Terminology Updates (Post Peer-Review)

**Date**: 2025-10-23
**Context**: Addressing peer-review feedback on NSM-33 physics metrics
**Status**: Applied to main codebase

---

## Summary of Changes

Following comprehensive peer review, we've updated terminology throughout the codebase to accurately reflect the nature of our physics-inspired metrics. The key change: acknowledging these are **empirical heuristics** inspired by physical systems, not rigorous mathematical isomorphisms.

## Key Terminology Changes

### 1. "Isomorphism" → "Empirical Heuristic"

**Rationale**: Peer review (research-assistant) identified that dimensional analysis fails for our physics metrics. True isomorphisms require:
- Dimensional consistency
- Coordinate invariance
- Preservation of mathematical structure

Our metrics lack these properties - they're **useful predictive tools** but not formal mappings.

**Files Updated**:
- `analysis/README_ISOMORPHISMS.md` → Title updated, disclaimer added
- `nsm/training/physics_metrics.py` → Module docstring clarified

**Pattern Applied**:
```markdown
# Before
"Physics Isomorphisms for Neural Collapse Prediction"
"Implements fusion-plasma isomorphism metrics"

# After
"Physics-Inspired Empirical Heuristics for Neural Collapse Prediction"
"Implements fusion-plasma-inspired metrics"

# With Disclaimer
"**Note**: These are empirical heuristics (not rigorous isomorphisms) inspired by structural
similarities to fusion plasma systems. Dimensional analysis reveals they lack true physical
correspondence, but remain useful predictive tools validated through NSM-33 experiments."
```

### 2. "Temperature" → "Representation Variance" (Outside Fusion Context)

**Rationale**: "Temperature" in our context means statistical variance/entropy of neural representations, NOT thermal temperature (kinetic energy). The fusion analogy remains valid only when explicitly acknowledged.

**Files Updated**:
- `nsm/training/physics_metrics.py` → Function `compute_temperature_profile()` docstring
- `analysis/README_ISOMORPHISMS.md` → Section headings
- `results/NSM-33_10x_validation_results.md` → Metric labels

**Pattern Applied**:
```python
# Function still named compute_temperature_profile() for backwards compatibility
# But docstring clarifies:

"""
Compute representation variance profile at each hierarchical level.

**Note**: "Temperature" here refers to representation variance/entropy, NOT thermal
temperature. The term is borrowed from fusion physics by analogy but represents a
fundamentally different quantity (statistical dispersion, not kinetic energy).

In the fusion analogy: temperature profiles T(r) determine confinement quality.
In neural networks: representation variance serves structurally analogous role:
    - High variance: Diverse, information-rich representations
    - Low variance: Collapsed, uniform representations
    - Inverted profile (variance decreasing with abstraction): Instability indicator
"""
```

**Variable Names** (retain T_ prefix for brevity, clarify in documentation):
- `T_L1`, `T_L2`, `T_L3` → Keep, but document as variance
- `T_gradient` → Keep, but clarify as "variance gradient"
- Display labels → Changed to "Representation Variance Profile"

### 3. "Physics Metrics" → "Empirical Stability Metrics"

**Context-Dependent**:
- **Keep "Physics Metrics"** in technical documentation where fusion analogy is explicit
- **Use "Empirical Stability Metrics"** in results/user-facing docs for clarity

**Example from NSM-33 Results**:
```markdown
# Before
**Physics Metrics (Final Epoch)**:
- **Temperature Profile**: T_L1=0.381, T_L2=3.268, T_L3=13.590

# After
**Empirical Stability Metrics (Final Epoch)**:
- **Representation Variance Profile**: T_L1=0.381, T_L2=3.268, T_L3=13.590
  - Note: "T" denotes variance/entropy, not thermal temperature
```

---

## What We DIDN'T Change

### Preserved Terminology (With Context)

1. **Variable names** (`T_L1`, `q_neural`, `Q_factor`) - Backwards compatibility
2. **Function names** (`compute_temperature_profile`) - API stability
3. **Fusion references** - When explicitly discussing the analogy
4. **Module names** (`physics_metrics.py`) - Established convention

### Fusion Context (Terminology OK)

When discussing the **fusion plasma analogy explicitly**, original terminology is appropriate:

```python
# In physics_metrics.py docstring:
"""
Mathematical parallels (structural, not isomorphic):
- Neural class collapse ↔ Plasma confinement loss
- α/β hinge parameters ↔ α/β fusion parameters
- Representation variance ↔ Temperature in fusion systems

References:
- Lawson, J.D. (1957). "Some Criteria for a Power Producing Thermonuclear Reactor"
- Wesson, J. (2011). "Tokamak Physics" (safety factor q)
"""
```

Here "temperature" refers to the fusion system, so no change needed.

---

## Documentation Added

### New File: `docs/diversity_regularization.md`

Comprehensive documentation of the diversity regularization mechanism, including:
- Mathematical formulation
- Implementation details
- Hyperparameter tuning
- NSM-33 results analysis
- Theoretical justification (information bottleneck)
- Peer review concerns (confounds, causation)
- Recommended ablation studies

**Key Addition**: Explicit discussion of reviewer's critique that high variance may indicate instability, not health.

---

## Files Modified

### Core Changes
1. **nsm/training/physics_metrics.py** (lines 1-22, 106-157)
   - Module docstring: Clarified heuristic nature
   - Function docstring: Explained T = variance, not thermal
   - Comments: Replaced "temperature" with "variance" in implementation

2. **analysis/README_ISOMORPHISMS.md** (lines 1-62)
   - Title: "Physics-Inspired Empirical Heuristics..."
   - Added terminology disclaimer paragraph
   - Updated section headings

3. **results/NSM-33_10x_validation_results.md** (lines 11-62)
   - Executive summary: Added terminology note
   - Metric labels: "Empirical Stability Metrics"
   - Profile labels: "Representation Variance Profile"
   - Added "(NOT thermal temperature)" clarifications

### New Files
4. **docs/diversity_regularization.md** (250 lines)
   - Complete mechanism documentation
   - Addresses peer review concerns
   - Includes alternative interpretations

5. **TERMINOLOGY_UPDATES.md** (this file)
   - Change log and rationale

---

## Rationale from Peer Review

### Dimensional Analysis Failure

**Reviewer's Critique**:
> "Dimensional analysis fails: In tokamak physics, q has dimensions [dimensionless] from ratio of magnetic field ratios. Your q_neural combines arbitrary units from gradient norms and class balances. Cannot compare across models/scales."

**Response**: Acknowledged. Changed "isomorphism" to "heuristic" throughout.

### Temperature Interpretation

**Reviewer's Critique**:
> "High variance in L3 might indicate insufficient training (representations not converged), regularization preventing compression, or fighting against natural information bottleneck."

**Counter-evidence from NSM-33**:
- Fixed architecture has WORSE class balance (11.48% vs 5.91%)
- Fixed architecture has LOWER q_neural (0.625 vs 1.336)
- Scale alone achieves better results

**Conclusion**: Effect is CONFOUNDED - scale dominates diversity regularization.

**Action Taken**:
- Updated diversity_regularization.md with alternative interpretation
- Clarified "temperature" = variance (not claiming thermal correspondence)
- Recommended ablation at fixed scale to isolate effect

---

## Impact on Codebase

### Backwards Compatibility
✅ **Preserved**: All APIs, function signatures, variable names
- `compute_temperature_profile()` - function name unchanged
- `T_L1`, `T_L2`, `T_L3` - variable names unchanged
- `q_neural`, `Q_factor` - metric names unchanged

### User-Facing Changes
⚠️ **Updated**: Documentation, comments, docstrings
- Users will see clarified terminology in help text
- Results reports use "Empirical Stability Metrics"
- No code changes required for existing usage

### Semantic Changes
🔄 **Clarified**: Interpretation, not measurement
- Metrics compute the same values
- Interpretation is more accurate
- Claims are more modest

---

## Future Work

### Theoretical Strengthening (From Peer Review)

1. **Information-theoretic reformulation**:
   ```python
   # Replace variance with mutual information
   T_Lk = I(X_Lk; Y)  # Information about labels

   # From literature: Tishby & Zaslavsky (2015)
   # Predicts: I decreases with depth (compression)
   ```

2. **PAC learning bounds** for split ratios:
   ```python
   def compute_min_val_size(
       vc_dimension: int,
       error_bound: float = 0.05,
       confidence: float = 0.95
   ) -> int:
       """Derive from Vapnik (1998), not 'industry standard'"""
       delta = 1 - confidence
       return int((vc_dimension / error_bound**2) * (np.log(1/delta) + np.log(2)))
   ```

3. **Multi-seed validation**: Run 5 seeds, report mean ± std, significance tests

---

## References

### Peer Review Source
- **research-assistant** comprehensive review (2025-10-23)
- Grade: B+ (Strong execution, moderate theoretical rigor)
- Key feedback: "Physics isomorphism overclaimed - dimensional analysis fails"

### Literature Cited in Updates
- **Tishby & Zaslavsky (2015)**: Information Bottleneck Principle
- **Vapnik (1998)**: Statistical Learning Theory (PAC bounds)
- **Shwartz-Ziv & Tishby (2017)**: Opening Black Box of DNNs

---

## Commit Message Template

```
Update terminology: physics isomorphisms → empirical heuristics

Address peer review feedback on NSM-33 physics metrics:
- Clarify "isomorphisms" are empirical heuristics (not rigorous)
- Document "temperature" means variance/entropy (not thermal)
- Add diversity regularization mechanism documentation
- Preserve backwards compatibility (APIs unchanged)

Files modified:
- analysis/README_ISOMORPHISMS.md
- nsm/training/physics_metrics.py
- results/NSM-33_10x_validation_results.md
- docs/diversity_regularization.md (NEW)

Rationale: Dimensional analysis reveals metrics lack invariance
properties required for true physical analogies. Remain useful
predictive tools validated through experiment.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

---

## Status

✅ **Completed**: Terminology updates applied
🚧 **In Progress**: Multi-seed validation experiments (5 seeds × 3 conditions)
📋 **TODO**: Statistical significance analysis with confidence intervals

**Next Steps**:
1. Wait for multi-seed experiments to complete
2. Analyze results with proper significance testing
3. Create PR with terminology updates + multi-seed results
4. Address remaining peer review feedback (PAC bounds, information theory)
