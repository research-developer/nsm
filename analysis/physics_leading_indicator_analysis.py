"""
Analyze if physics metrics are leading indicators of collapse.

Question: Does q_neural < 1 appear BEFORE class balance degrades?
Or is it just concurrent/lagging?

This determines if physics metrics provide predictive advantage.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Parse the validation results
results_data = {
    "epochs": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "q_neural": [1.039, 0.038, 2.724, 0.145, 0.133, 0.095, 0.021, 0.070, 0.026, 0.346],
    "balance_delta": [0.0368, 0.6222, 0.0130, 0.3097, 0.1690, 0.3425, 0.6552, 0.5327, 0.7152, 0.1443],
    "accuracy": [0.4667, 0.4782, 0.4782, 0.4816, 0.4483, 0.4276, 0.4770, 0.4483, 0.4747, 0.4437],
    "temp_L1": [0.400, 0.397, 0.395, 0.393, 0.391, 0.390, 0.388, 0.386, 0.384, 0.383],
    "temp_L3": [0.133, 0.133, 0.130, 0.132, 0.136, 0.134, 0.126, 0.133, 0.129, 0.135],
    "temp_gradient": [-0.267, -0.264, -0.265, -0.261, -0.256, -0.256, -0.261, -0.253, -0.255, -0.247]
}

def analyze_leading_indicator():
    """
    Test if q_neural provides advance warning before collapse.

    Definition of "leading":
    - q drops below threshold at epoch N
    - balance_delta increases significantly at epoch N+1 or N+2

    If they move together (same epoch), it's concurrent, not leading.
    """

    print("="*70)
    print("LEADING INDICATOR ANALYSIS")
    print("="*70)

    # Define thresholds
    Q_UNSTABLE = 1.0
    BALANCE_COLLAPSE = 0.4

    epochs = results_data["epochs"]
    q_vals = results_data["q_neural"]
    balance_vals = results_data["balance_delta"]

    print("\nEpoch-by-Epoch Analysis:")
    print(f"{'Epoch':<8} {'q_neural':<12} {'Balance Δ':<12} {'Analysis'}")
    print("-"*70)

    lead_count = 0
    concurrent_count = 0
    lag_count = 0

    for i, epoch in enumerate(epochs):
        q = q_vals[i]
        balance = balance_vals[i]

        # Check if q is unstable
        q_unstable = q < Q_UNSTABLE
        balance_collapsed = balance > BALANCE_COLLAPSE

        analysis = []

        if q_unstable and balance_collapsed:
            analysis.append("⚠️  CONCURRENT collapse")
            concurrent_count += 1
        elif q_unstable and not balance_collapsed:
            # Check if balance collapses in next 1-2 epochs
            if i + 1 < len(epochs) and balance_vals[i+1] > BALANCE_COLLAPSE:
                analysis.append("✅ LEADING indicator (+1 epoch)")
                lead_count += 1
            elif i + 2 < len(epochs) and balance_vals[i+2] > BALANCE_COLLAPSE:
                analysis.append("✅ LEADING indicator (+2 epochs)")
                lead_count += 1
            else:
                analysis.append("🟡 q unstable, no collapse follows")
        elif not q_unstable and balance_collapsed:
            analysis.append("❌ LAGGING (missed collapse)")
            lag_count += 1
        else:
            analysis.append("✓ Stable")

        print(f"{epoch:<8} {q:<12.3f} {balance:<12.3f} {' '.join(analysis)}")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Leading indicators:  {lead_count} / {len(epochs)} ({lead_count/len(epochs)*100:.1f}%)")
    print(f"Concurrent signals:  {concurrent_count} / {len(epochs)} ({concurrent_count/len(epochs)*100:.1f}%)")
    print(f"Lagging/missed:      {lag_count} / {len(epochs)} ({lag_count/len(epochs)*100:.1f}%)")

    # Correlation analysis
    from scipy.stats import pearsonr

    # Shift q_neural forward by 1 epoch to test if it predicts next balance
    q_shifted = q_vals[:-1]
    balance_next = balance_vals[1:]

    corr_concurrent, p_concurrent = pearsonr(q_vals, balance_vals)
    corr_leading, p_leading = pearsonr(q_shifted, balance_next)

    print(f"\nCorrelation Analysis:")
    print(f"  q(t) vs balance(t):   r = {corr_concurrent:.3f} (p = {p_concurrent:.3f})")
    print(f"  q(t) vs balance(t+1): r = {corr_leading:.3f} (p = {p_leading:.3f})")

    if abs(corr_leading) > abs(corr_concurrent):
        print("\n✅ q_neural is a LEADING indicator (stronger correlation with future)")
    else:
        print("\n⚠️  q_neural is CONCURRENT (not predictive of future)")

    # Check against simple heuristic
    print("\n" + "="*70)
    print("COMPARISON TO SIMPLE HEURISTIC")
    print("="*70)
    print("\nSimple rule: 'If balance(t) > 0.3, expect worse at t+1'")

    simple_warnings = 0
    simple_correct = 0

    for i in range(len(epochs) - 1):
        if balance_vals[i] > 0.3:
            simple_warnings += 1
            if balance_vals[i+1] > balance_vals[i]:
                simple_correct += 1

    physics_warnings = sum(1 for q in q_vals[:-1] if q < Q_UNSTABLE)
    physics_correct = lead_count + concurrent_count

    print(f"\nSimple heuristic: {simple_correct}/{simple_warnings} predictions correct ({simple_correct/max(simple_warnings,1)*100:.1f}%)")
    print(f"Physics q_neural: {physics_correct}/{physics_warnings} predictions correct ({physics_correct/max(physics_warnings,1)*100:.1f}%)")

    if physics_correct / max(physics_warnings, 1) > simple_correct / max(simple_warnings, 1):
        print("\n✅ Physics metrics OUTPERFORM simple heuristic")
    else:
        print("\n⚠️  Physics metrics do NOT outperform simple heuristic")

    return {
        "lead_count": lead_count,
        "concurrent_count": concurrent_count,
        "lag_count": lag_count,
        "corr_concurrent": corr_concurrent,
        "corr_leading": corr_leading,
        "outperforms_heuristic": physics_correct / max(physics_warnings, 1) > simple_correct / max(simple_warnings, 1)
    }


def plot_trajectories():
    """Plot physics metrics vs. outcomes to visualize relationships."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = results_data["epochs"]

    # Plot 1: q_neural vs balance delta
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()

    ax1.plot(epochs, results_data["q_neural"], 'b-o', label='q_neural', linewidth=2)
    ax1_twin.plot(epochs, results_data["balance_delta"], 'r-s', label='Balance Δ', linewidth=2)

    ax1.axhline(y=1.0, color='b', linestyle='--', alpha=0.5, label='q=1 (stability threshold)')
    ax1_twin.axhline(y=0.4, color='r', linestyle='--', alpha=0.5, label='Δ=0.4 (collapse threshold)')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('q_neural (safety factor)', color='b')
    ax1_twin.set_ylabel('Balance Δ', color='r')
    ax1.set_title('Physics Safety Factor vs. Class Balance')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')

    # Plot 2: Temperature profile trajectory
    ax2 = axes[0, 1]
    ax2.plot(epochs, results_data["temp_L1"], 'g-o', label='T_L1 (concrete)')
    ax2.plot(epochs, results_data["temp_L3"], 'purple', marker='s', label='T_L3 (abstract)')
    ax2.fill_between(epochs, results_data["temp_L1"], results_data["temp_L3"],
                      alpha=0.2, color='red', label='Inversion region')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Temperature (diversity)')
    ax2.set_title('Temperature Profile - INVERTED')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: q_neural shifted forward (leading indicator test)
    ax3 = axes[1, 0]
    ax3.scatter(results_data["q_neural"][:-1], results_data["balance_delta"][1:],
                s=100, alpha=0.6, c=epochs[:-1], cmap='viridis')
    ax3.set_xlabel('q_neural at epoch t')
    ax3.set_ylabel('Balance Δ at epoch t+1')
    ax3.set_title('Leading Indicator Test: q(t) vs Balance(t+1)')
    ax3.axvline(x=1.0, color='r', linestyle='--', alpha=0.5, label='q=1 threshold')
    ax3.axhline(y=0.4, color='r', linestyle='--', alpha=0.5, label='Collapse threshold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Temperature gradient vs balance
    ax4 = axes[1, 1]
    ax4.scatter(results_data["temp_gradient"], results_data["balance_delta"],
                s=100, alpha=0.6, c=epochs, cmap='viridis')
    ax4.set_xlabel('Temperature Gradient (L3 - L1)')
    ax4.set_ylabel('Balance Δ')
    ax4.set_title('Inverted Profile Correlation')
    ax4.axvline(x=0, color='g', linestyle='--', alpha=0.5, label='Normal profile (>0)')
    ax4.axhline(y=0.4, color='r', linestyle='--', alpha=0.5, label='Collapse threshold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.colorbar(ax3.collections[0], ax=axes[1, 0], label='Epoch')
    plt.colorbar(ax4.collections[0], ax=axes[1, 1], label='Epoch')

    plt.tight_layout()
    plt.savefig('/Users/preston/Projects/NSM/analysis/physics_leading_indicator_plots.png', dpi=150)
    print("\n📊 Plots saved to analysis/physics_leading_indicator_plots.png")


if __name__ == '__main__':
    # Need scipy for correlation
    try:
        import scipy
    except ImportError:
        print("Installing scipy for correlation analysis...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'scipy', 'matplotlib'])
        import scipy

    results = analyze_leading_indicator()
    plot_trajectories()

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    if results["lead_count"] > 0:
        print("✅ Physics metrics provide LEADING indicators")
        print("   q_neural drops before collapse in some cases")
    else:
        print("⚠️  Physics metrics are CONCURRENT, not leading")
        print("   They confirm collapse but don't predict it early")

    if results["outperforms_heuristic"]:
        print("✅ Physics metrics OUTPERFORM simple heuristics")
        print("   Worth the additional complexity")
    else:
        print("⚠️  Physics metrics do NOT outperform simple rules")
        print("   May be overcomplicated for practical use")
