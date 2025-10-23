"""
Simplified Modal deployment for CGT operator validation (NSM-34).

Validates Conway temperature and cooling operators using synthetic data and mock models.
This focuses on testing the operators themselves, not the full model integration.

Usage:
    modal run experiments/modal_cgt_validation_simple.py::validate_operators
"""

import modal
from pathlib import Path

app = modal.App("nsm-cgt-validation-simple")
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Minimal image for testing - only mount cgt_metrics.py to avoid import chain
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch==2.1.0",
        "numpy<2",  # Fix: torch 2.1.0 compiled with NumPy 1.x
        "scipy"
    )
    .add_local_file(
        PROJECT_ROOT / "nsm" / "training" / "cgt_metrics.py",
        remote_path="/root/cgt_metrics.py"
    )
)

volume = modal.Volume.from_name("nsm-cgt-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",  # Use cheaper GPU for testing
    timeout=1800,
    volumes={"/results": volume}
)
def validate_operators():
    """
    Validate CGT operators using mock models (like unit tests).

    This tests the operators themselves without needing full model architecture.
    """
    import torch
    import torch.nn as nn
    import numpy as np
    import json
    from datetime import datetime
    from pathlib import Path

    # Mock model with WHY/WHAT methods
    class MockModel(nn.Module):
        def __init__(self, hidden_dim=64, asymmetry=0.3):
            super().__init__()
            self.encoder = nn.Linear(hidden_dim, hidden_dim // 2)
            self.decoder = nn.Linear(hidden_dim // 2, hidden_dim)
            self.asymmetry = asymmetry

        def why(self, x):
            """Abstraction (with controlled noise for temperature)."""
            z = self.encoder(x)
            if self.training:
                z = z + torch.randn_like(z) * self.asymmetry
            return z

        def what(self, z):
            """Concretization."""
            return self.decoder(z)

    # Import CGT operators (standalone file)
    import sys
    sys.path.insert(0, "/root")
    from cgt_metrics import (
        temperature_conway,
        CoolingMonitor
    )

    print("\n" + "="*80)
    print("CGT OPERATORS VALIDATION (Simplified)")
    print("="*80)

    results = {}

    # ========================================================================
    # Test 1: Conway Temperature
    # ========================================================================
    print("\n📊 Test 1: Conway Temperature")

    model = MockModel(hidden_dim=64, asymmetry=0.3).cuda()
    model.eval()

    # Test on multiple batches
    temperatures = []
    for i in range(20):
        x = torch.randn(32, 64).cuda()
        temp, diag = temperature_conway(model, x, num_samples=10, metric='mse')
        temperatures.append(temp)

        if i == 0:
            print(f"   First batch: t(G) = {temp:.4f}")
            print(f"   max_left = {diag['max_left']:.4f}")
            print(f"   min_right = {diag['min_right']:.4f}")

    mean_temp = np.mean(temperatures)
    std_temp = np.std(temperatures)
    min_temp = min(temperatures)
    max_temp = max(temperatures)

    print(f"   Mean temperature: {mean_temp:.4f} ± {std_temp:.4f}")
    print(f"   Range: [{min_temp:.4f}, {max_temp:.4f}]")

    # Check prediction P1.2: temperature < 0.2 indicates collapse risk
    stable_count = sum(1 for t in temperatures if t > 0.2)
    print(f"   P1.2 check: {stable_count}/20 batches have t > 0.2 (stable)")

    results['temperature'] = {
        'mean': float(mean_temp),
        'std': float(std_temp),
        'min': float(min_temp),
        'max': float(max_temp),
        'stable_ratio': stable_count / 20,
        'temperatures': [float(t) for t in temperatures],
        'prediction_P1_2': f"threshold_check: {stable_count}/20 stable"
    }

    # ========================================================================
    # Test 2: Cooling Monitor
    # ========================================================================
    print("\n📊 Test 2: Cooling Monitor")

    monitor = CoolingMonitor(window_size=5)

    # Simulate training with α,β → 0.5 (cooling toward collapse)
    alphas = [0.9 - i * 0.05 for i in range(20)]  # 0.9 → -0.05
    betas = [0.1 + i * 0.05 for i in range(20)]   # 0.1 → 1.05

    temps = []
    rates = []
    predictions = []

    for epoch, (alpha, beta) in enumerate(zip(alphas, betas)):
        rate = monitor.update(alpha, beta)
        stats = monitor.get_statistics()

        temps.append(stats['current_temp'])
        if rate is not None:
            rates.append(rate)

            # Predict collapse time
            epochs_remaining = monitor.predict_collapse_time(threshold_temp=0.1)
            predictions.append(epochs_remaining)

            if epoch < 5 or epoch % 5 == 0:
                print(f"   Epoch {epoch:2d}: T={stats['current_temp']:.4f}, "
                      f"δT/δe={rate:.6f}, collapse_in={epochs_remaining}")

    # Analysis
    mean_cooling = np.mean(rates)
    rapid_cooling_events = sum(1 for r in rates if r < -0.05)
    temp_decreased = temps[0] > temps[-1]

    print(f"\n   Analysis:")
    print(f"   - Initial temp: {temps[0]:.4f} → Final temp: {temps[-1]:.4f}")
    print(f"   - Mean cooling rate: {mean_cooling:.6f}")
    print(f"   - Rapid cooling events (< -0.05): {rapid_cooling_events}")
    print(f"   - Temperature decreased: {temp_decreased}")

    # Check prediction P2.1: rapid cooling predicts collapse
    print(f"   P2.1 check: {rapid_cooling_events} rapid cooling events detected")

    results['cooling'] = {
        'initial_temp': float(temps[0]),
        'final_temp': float(temps[-1]),
        'temp_decreased': bool(temp_decreased),
        'mean_cooling_rate': float(mean_cooling),
        'rapid_cooling_events': int(rapid_cooling_events),
        'temperature_history': [float(t) for t in temps],
        'cooling_rate_history': [float(r) for r in rates],
        'prediction_P2_1': f"rapid_cooling_detected: {rapid_cooling_events} events"
    }

    # ========================================================================
    # Test 3: Integration (collapse simulation)
    # ========================================================================
    print("\n📊 Test 3: Collapse Simulation")

    monitor2 = CoolingMonitor()

    # Simulate aggressive cooling (collapse scenario)
    collapse_alphas = [0.95, 0.85, 0.70, 0.60, 0.52, 0.50, 0.50]
    collapse_betas =  [0.05, 0.15, 0.30, 0.40, 0.48, 0.50, 0.50]

    collapse_temps = []
    collapse_detected = False

    for epoch, (alpha, beta) in enumerate(zip(collapse_alphas, collapse_betas)):
        rate = monitor2.update(alpha, beta)
        stats = monitor2.get_statistics()
        collapse_temps.append(stats['current_temp'])

        # Check for collapse indicators
        if rate and rate < -0.05 and stats['current_temp'] < 0.2:
            if not collapse_detected:
                print(f"   ⚠️  Collapse detected at epoch {epoch}!")
                print(f"       T={stats['current_temp']:.4f}, δT/δe={rate:.6f}")
                collapse_detected = True

    print(f"   Collapse simulation result: {' detected' if collapse_detected else 'NOT detected'}")

    results['integration'] = {
        'collapse_detected': bool(collapse_detected),
        'temperature_trajectory': [float(t) for t in collapse_temps]
    }

    # ========================================================================
    # Save Results
    # ========================================================================
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'gpu': 'T4',
        'tests_passed': {
            'temperature': bool(mean_temp > 0),  # Non-negative (convert numpy bool_)
            'cooling': bool(temp_decreased),      # Temperature decreased (convert numpy bool_)
            'integration': bool(collapse_detected)  # Detected simulated collapse
        },
        'results': results
    }

    results_path = Path("/results/validation_simple.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)

    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print(f"✅ Temperature: mean={mean_temp:.4f}, stable_ratio={stable_count/20:.1%}")
    print(f"✅ Cooling: mean_rate={mean_cooling:.6f}, rapid_events={rapid_cooling_events}")
    print(f"✅ Integration: collapse_detected={collapse_detected}")

    return results_summary


@app.local_entrypoint()
def main():
    """Run simplified validation."""
    print("🚀 Running simplified CGT operators validation...")
    result = validate_operators.remote()
    print("\n📊 Final Results:")
    import json
    print(json.dumps(result['tests_passed'], indent=2))
    return result
