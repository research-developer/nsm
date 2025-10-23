# CGT Results Interpretation Guide

Quick reference for understanding CGT experiment outputs.

## Conway Temperature (t(G))

### What It Measures
Temperature quantifies WHY/WHAT asymmetry in the model:
- `t(G) = min{ ||WHY(x) - WHAT⁻¹(x)||² } - max{ ||WHY(x) - WHAT⁻¹(x')||² }`
- Higher temperature = more asymmetry = more learned structure
- Lower temperature = less asymmetry = more symmetric/random

### Interpretation Table

| Temperature Range | Status | Meaning | Action |
|------------------|---------|---------|---------|
| `≈ 0.0000` | **EXPECTED** (untrained) | Random/untrained model has perfect symmetry | ✅ Normal for 0-10 epochs |
| `0.0000 - 0.0100` | **EXPECTED** (early) | Model beginning to learn, weak asymmetry | ✅ Normal for < 10 epochs |
| `0.0100 - 0.2000` | **CAUTION** | Model learning but approaching collapse threshold | ⚠️ Monitor closely |
| `0.2000 - 0.5000` | **HEALTHY** | Strong learned asymmetry, stable dynamics | ✅ Production-ready |
| `> 0.5000` | **STRONG** | Very asymmetric, well-learned structure | ✅ Excellent |

### Special Cases

#### "Temperature is 0.0000 - is this broken?"
**NO.** This is correct for:
- **Untrained models**: Random weights have no asymmetry
- **Very early training** (< 5 epochs): Not enough time to develop structure
- **Perfectly symmetric architecture**: Some models converge to WHY ≈ WHAT⁻¹

**What to check**:
1. How many epochs? If < 10, this is expected
2. Is model training? Check if accuracy is improving
3. Are operators working? Run `modal_cgt_validation_simple.py` to test operators

#### "Temperature dropped below 0.2 after being higher"
**WARNING.** This indicates potential collapse:
- **Prediction P1.2**: Temperature < 0.2 predicts collapse with >90% accuracy
- **Action**: Enable stability interventions (cycle loss weight, early stopping)
- **Diagnosis**: Model may be overfitting or losing learned asymmetry

## Cooling Rate (δT/δe)

### What It Measures
Rate of temperature change per epoch:
- `δT/δe = (T_current - T_previous) / 1`
- Negative = temperature decreasing (cooling)
- Monitors trajectory toward collapse

### Interpretation Table

| Cooling Rate | Status | Meaning | Action |
|--------------|---------|---------|---------|
| `> 0` | **HEATING** | Temperature increasing (learning) | ✅ Normal early training |
| `0` | **STABLE** | Temperature constant | ✅ Converged or plateau |
| `-0.05 to 0` | **MILD COOLING** | Slow decrease | ℹ️ Monitor |
| `< -0.05` | **RAPID COOLING** | Fast decrease → collapse risk | ⚠️ **Prediction P2.1 triggered** |

### Special Cases

#### "Cooling rate is -0.0001 every epoch"
**NORMAL.** This is gentle convergence:
- Model stabilizing after initial learning
- Temperature reaching equilibrium
- No immediate collapse risk

#### "Cooling rate suddenly dropped to -0.15"
**DANGER.** Rapid cooling detected:
- **Prediction P2.1**: Cooling < -0.05 predicts collapse within 2 epochs
- **Action**: Stop training, investigate cause
- **Diagnosis**: Check for gradient explosion, learning rate too high, or data shift

## Training Epochs vs. Expected Results

### Quick Validation (5 epochs)
**Purpose**: Smoke test operators, verify code works
**Expected Results**:
- Temperature: ~0.0000 - 0.0050 (near zero)
- Accuracy: ~0.50 - 0.55 (barely above random)
- Status: "PRELIMINARY" or "EXPECTED for early training"
**Interpretation**: Operators working, model barely trained

### Development (10 epochs)
**Purpose**: Early development checkpoint
**Expected Results**:
- Temperature: ~0.0050 - 0.0200
- Accuracy: ~0.55 - 0.65
- Status: "DEVELOPING"
**Interpretation**: Model learning, not yet stable

### Production (15+ epochs)
**Purpose**: Meaningful validation, production model
**Expected Results**:
- Temperature: > 0.2000 (healthy)
- Accuracy: > 0.70
- Status: "PRODUCTION-READY"
**Interpretation**: Model trained, results actionable

## Common Scenarios

### Scenario 1: First-time Run
```
Training: 5 epochs
Temperature: 0.0002
Accuracy: 0.51
```
**Interpretation**: ✅ **EXPECTED**
- Operators functioning correctly
- Model hasn't learned yet (too few epochs)
- This is a successful smoke test

**Action**: Run `--epochs=15` for real results

---

### Scenario 2: Development Run
```
Training: 10 epochs
Temperature: 0.0134
Accuracy: 0.62
```
**Interpretation**: ℹ️ **DEVELOPING**
- Model learning but not converged
- Temperature low but improving
- Heading in right direction

**Action**: Continue training or tune hyperparameters

---

### Scenario 3: Production Run (Healthy)
```
Training: 20 epochs
Temperature: 0.3421
Accuracy: 0.78
```
**Interpretation**: ✅ **PRODUCTION-READY**
- Strong asymmetry developed
- Good accuracy
- Stable learning dynamics

**Action**: Use this model for validation

---

### Scenario 4: Collapse Detected
```
Training: 30 epochs
Temperature: 0.1523 → 0.0421 (dropped)
Cooling Rate: -0.1102
Accuracy: 0.76 → 0.54 (dropped)
```
**Interpretation**: ⚠️ **COLLAPSE IN PROGRESS**
- P1.2 triggered (temp < 0.2)
- P2.1 triggered (cooling < -0.05)
- Accuracy degrading

**Action**:
1. Stop training immediately
2. Restore previous checkpoint
3. Enable stability interventions
4. Reduce learning rate or add cycle loss

## Command Reference

### Run Quick Validation (5 epochs)
```bash
modal run experiments/modal_cgt_training.py --epochs=5
```
Expect: Temperature ≈ 0, Status: "PRELIMINARY"

### Run Production Training (15+ epochs)
```bash
modal run experiments/modal_cgt_training.py --epochs=15
```
Expect: Temperature > 0.2, Status: "PRODUCTION-READY"

### Test Operators Only (No Training)
```bash
modal run experiments/modal_cgt_validation_simple.py
```
Validates operators work correctly (independent of model quality)

### Full Validation Suite
```bash
modal run experiments/modal_cgt_validation.py::validate_all_operators
```
Runs all CGT operators on current model

## Health Check Output Guide

### Status Labels

| Label | Meaning | Is This Bad? |
|-------|---------|--------------|
| **EXPECTED for untrained model** | Results typical for 0-10 epoch model | ❌ No, this is correct |
| **PRELIMINARY** | Early-stage results, not production-ready | ⚠️ No, but train more |
| **DEVELOPING** | Model learning, progressing normally | ℹ️ No, keep going |
| **PRODUCTION-READY** | Results are meaningful and stable | ✅ No, all good! |
| **CAUTION** | Potential issue detected | ⚠️ Yes, investigate |
| **DANGER** | Collapse imminent | ❌ Yes, take action |

### Warning Icons

| Icon | Meaning | Should I Worry? |
|------|---------|-----------------|
| ✅ | All good, working as intended | No |
| ℹ️ | Informational, for context | No |
| 📝 | Explanation of why you're seeing this | No |
| 💡 | Recommendation for next steps | No |
| ⚠️ | Caution, requires attention | Maybe (check context) |
| ❌ | Error or critical issue | Yes |

## FAQ

### Q: My temperature is 0.0000. Did the operator fail?
**A**: No. This is correct for untrained models. Random weights have perfect WHY/WHAT symmetry → t(G) ≈ 0.

### Q: How many epochs until I see meaningful temperature?
**A**: Typically 15-20 epochs. Depends on:
- Model complexity (6-level takes longer)
- Learning rate (slower = gradual asymmetry development)
- Cycle loss weight (higher = stronger symmetry constraint)

### Q: What's a "good" temperature value?
**A**: Depends on context:
- For collapse prediction validation: > 0.2 (healthy)
- For general training: Any positive value is learning
- For production models: > 0.3 indicates strong structure

### Q: Should I always run 15+ epochs?
**A**: No:
- **Quick tests**: 5 epochs is fine (just testing operators)
- **Development**: 10 epochs to check progress
- **Production**: 15+ epochs for meaningful results
- **Full validation**: 30+ epochs for research

### Q: Temperature was high, then dropped. Is this bad?
**A**: **Yes, investigate immediately.** This indicates:
- Potential collapse (P1.2)
- Overfitting
- Loss of learned asymmetry
Check: cooling rate, accuracy trend, gradient norms

### Q: All health checks say "EXPECTED" but I want better results
**A**: "EXPECTED" means operators are working correctly given your training duration. For better *model* results:
1. Train longer (15+ epochs)
2. Tune hyperparameters
3. Check dataset quality
4. Adjust cycle loss weight

## Related Files

- `CGT_UX_IMPROVEMENTS.md`: Details of UX changes made
- `MODAL_CGT_DIAGNOSTIC_REPORT.md`: Technical diagnostic report
- `modal_cgt_training.py`: Training with CGT tracking
- `modal_cgt_validation.py`: Full validation suite
- `modal_cgt_validation_simple.py`: Operator-only validation

## Support

If results are unexpected after reading this guide:
1. Check experiment logs for health check section
2. Review training duration (5 vs 15 vs 30 epochs)
3. Run simple validation to test operators: `modal run modal_cgt_validation_simple.py`
4. Compare with examples in this guide
5. File issue with health check output included
