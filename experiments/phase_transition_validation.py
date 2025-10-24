"""
Phase Transition Validation Experiment

Tests Isomorphism 1 predictions:
1. Critical slowing: σ²(ψ) spikes before collapse
2. Hysteresis: Forward/backward paths differ
3. Power law scaling: ψ ∝ (T - Tₖ)^β

Based on analysis/additional_isomorphisms.md
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from pathlib import Path
import json
from datetime import datetime

from nsm.models.chiral import FullChiralModel
from nsm.training.chiral_loss import ChiralCompositeLoss
from nsm.training.physics_metrics import compute_all_physics_metrics
from nsm.data.planning_dataset import PlanningTripleDataset
from torch_geometric.data import Batch


class PhaseTransitionValidator:
    """Validates phase transition hypothesis for neural collapse."""

    def __init__(self, output_dir: str = "results/phase_transition"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = {
            'critical_slowing': {},
            'hysteresis': {},
            'scaling': {},
            'timestamp': datetime.now().isoformat()
        }

    def compute_order_parameter(self, class_accuracies: Dict[str, float]) -> float:
        """
        Compute order parameter ψ = 1 - |acc₀ - acc₁|.

        ψ = 1: Perfect balance (ordered phase)
        ψ = 0: Complete collapse (disordered phase)
        """
        acc_0 = class_accuracies.get('accuracy_class_0', 0.5)
        acc_1 = class_accuracies.get('accuracy_class_1', 0.5)
        psi = 1.0 - abs(acc_0 - acc_1)
        return psi

    def test_critical_slowing(
        self,
        model: FullChiralModel,
        train_loader,
        val_loader,
        device,
        epochs: int = 15,
        window: int = 3
    ) -> Dict[str, any]:
        """
        Test Prediction 1: Variance σ²(ψ) increases before collapse.

        Expected: σ² spikes at epochs 1, 6, 8 (before collapses at 2, 7, 9)
        Null: Variance remains constant throughout training
        """
        print("\n" + "="*70)
        print("TEST 1: CRITICAL SLOWING (VARIANCE SPIKE)")
        print("="*70)

        # Initialize loss and optimizer
        criterion = ChiralCompositeLoss(
            task_weight=1.0,
            aux_weight=0.3,
            cycle_weight=0.01,
            diversity_weight=0.0  # No intervention
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Track order parameter trajectory
        psi_history = []
        variance_history = []
        collapse_epochs = []

        for epoch in range(epochs):
            # Train one epoch
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()

                output = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
                loss_dict = criterion(output, batch.y)
                loss = loss_dict['loss']

                loss.backward()
                optimizer.step()

            # Validate
            model.eval()
            correct_0 = 0
            correct_1 = 0
            total_0 = 0
            total_1 = 0

            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    output = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
                    pred = output['logits'].argmax(dim=1)

                    for p, y in zip(pred, batch.y):
                        if y == 0:
                            total_0 += 1
                            if p == 0:
                                correct_0 += 1
                        else:
                            total_1 += 1
                            if p == 1:
                                correct_1 += 1

            acc_0 = correct_0 / max(total_0, 1)
            acc_1 = correct_1 / max(total_1, 1)

            # Compute order parameter
            psi = 1.0 - abs(acc_0 - acc_1)
            psi_history.append(psi)

            # Compute rolling variance (critical slowing indicator)
            if len(psi_history) >= window:
                recent = psi_history[-window:]
                variance = np.var(recent)
            else:
                variance = 0.0
            variance_history.append(variance)

            # Detect collapse (discontinuous drop)
            if epoch > 0 and psi - psi_history[-2] < -0.3:
                collapse_epochs.append(epoch)
                print(f"  🔴 Collapse detected at epoch {epoch}: Δψ = {psi - psi_history[-2]:.3f}")

            print(f"Epoch {epoch:2d} | ψ = {psi:.3f} | σ²(ψ) = {variance:.4f}")

        # Analysis: Do variance spikes precede collapses?
        baseline_variance = np.median(variance_history[:5])  # First 5 epochs
        spike_threshold = 2.0 * baseline_variance

        precursor_epochs = []
        for i, var in enumerate(variance_history):
            if var > spike_threshold:
                precursor_epochs.append(i)

        # Check if precursors occur 1-2 epochs before collapses
        true_positives = 0
        false_positives = 0

        for precursor_epoch in precursor_epochs:
            # Is there a collapse within next 2 epochs?
            predicted_collapse = any(
                precursor_epoch < collapse_epoch <= precursor_epoch + 2
                for collapse_epoch in collapse_epochs
            )
            if predicted_collapse:
                true_positives += 1
                print(f"  ✅ Variance spike at epoch {precursor_epoch} predicted collapse")
            else:
                false_positives += 1
                print(f"  ⚠️  Variance spike at epoch {precursor_epoch} with no collapse")

        # False negatives: collapses without precursor
        false_negatives = 0
        for collapse_epoch in collapse_epochs:
            # Was there a variance spike 1-2 epochs before?
            had_precursor = any(
                collapse_epoch - 2 <= precursor_epoch < collapse_epoch
                for precursor_epoch in precursor_epochs
            )
            if not had_precursor:
                false_negatives += 1
                print(f"  ❌ Collapse at epoch {collapse_epoch} with no variance spike")

        # Compute precision, recall
        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)

        print(f"\nResults:")
        print(f"  Baseline variance: {baseline_variance:.4f}")
        print(f"  Spike threshold: {spike_threshold:.4f}")
        print(f"  True positives:  {true_positives} / {len(precursor_epochs)} spikes")
        print(f"  False negatives: {false_negatives} / {len(collapse_epochs)} collapses")
        print(f"  Precision: {precision:.2%}")
        print(f"  Recall: {recall:.2%}")
        print(f"  F1 Score: {f1:.2%}")

        # Verdict
        if recall >= 0.7:
            print(f"\n✅ HYPOTHESIS CONFIRMED: Variance is a leading indicator")
        elif recall >= 0.4:
            print(f"\n🟡 HYPOTHESIS PARTIAL: Variance predicts some collapses")
        else:
            print(f"\n❌ HYPOTHESIS REJECTED: Variance is not predictive")

        # Plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Plot 1: Order parameter trajectory
        ax1 = axes[0]
        ax1.plot(psi_history, 'b-o', linewidth=2, markersize=8, label='ψ (order parameter)')
        for epoch in collapse_epochs:
            ax1.axvline(epoch, color='red', linestyle='--', alpha=0.5, label='Collapse' if epoch == collapse_epochs[0] else '')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Order Parameter ψ')
        ax1.set_title('Phase Transition Trajectory')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Variance (critical slowing)
        ax2 = axes[1]
        ax2.plot(variance_history, 'g-s', linewidth=2, markersize=6, label='σ²(ψ)')
        ax2.axhline(spike_threshold, color='orange', linestyle='--', label=f'Spike threshold (2×baseline)')
        for epoch in precursor_epochs:
            ax2.axvline(epoch, color='orange', linestyle=':', alpha=0.5, label='Variance spike' if epoch == precursor_epochs[0] else '')
        for epoch in collapse_epochs:
            ax2.axvline(epoch, color='red', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Variance σ²(ψ)')
        ax2.set_title('Critical Slowing (Variance Precursor)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plot_path = self.output_dir / 'critical_slowing.png'
        plt.savefig(plot_path, dpi=150)
        print(f"\n📊 Plot saved: {plot_path}")

        results = {
            'psi_history': psi_history,
            'variance_history': variance_history,
            'collapse_epochs': collapse_epochs,
            'precursor_epochs': precursor_epochs,
            'baseline_variance': float(baseline_variance),
            'spike_threshold': float(spike_threshold),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'hypothesis_confirmed': recall >= 0.7
        }

        self.results['critical_slowing'] = results
        return results

    def test_hysteresis(
        self,
        model: FullChiralModel,
        train_loader,
        val_loader,
        device,
        epochs: int = 30
    ) -> Dict[str, any]:
        """
        Test Prediction 2: Forward/backward diversity paths differ (hysteresis loop).

        Expected: Heating (0 → 0.5) and cooling (0.5 → 0) trace different trajectories
        Null: Symmetric, reversible path (no memory)
        """
        print("\n" + "="*70)
        print("TEST 2: HYSTERESIS LOOP")
        print("="*70)

        # Schedule: 0 → 0.5 (heating) → 0 (cooling)
        diversity_schedule = []
        psi_forward = []
        psi_backward = []

        # Phase 1: Heating (epochs 0-14)
        for epoch in range(15):
            diversity = 0.5 * (epoch / 14)  # Linear ramp 0 → 0.5
            diversity_schedule.append(diversity)

        # Phase 2: Cooling (epochs 15-29)
        for epoch in range(15):
            diversity = 0.5 * (1 - epoch / 14)  # Linear ramp 0.5 → 0
            diversity_schedule.append(diversity)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Train with schedule
        for epoch, diversity in enumerate(diversity_schedule):
            criterion = ChiralCompositeLoss(
                task_weight=1.0,
                aux_weight=0.3,
                cycle_weight=0.01,
                diversity_weight=diversity
            )

            # Train
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()

                output = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
                loss_dict = criterion(output, batch.y)
                loss = loss_dict['loss']

                loss.backward()
                optimizer.step()

            # Validate
            model.eval()
            correct_0 = 0
            correct_1 = 0
            total_0 = 0
            total_1 = 0

            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    output = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
                    pred = output['logits'].argmax(dim=1)

                    for p, y in zip(pred, batch.y):
                        if y == 0:
                            total_0 += 1
                            if p == 0:
                                correct_0 += 1
                        else:
                            total_1 += 1
                            if p == 1:
                                correct_1 += 1

            acc_0 = correct_0 / max(total_0, 1)
            acc_1 = correct_1 / max(total_1, 1)
            psi = 1.0 - abs(acc_0 - acc_1)

            if epoch < 15:
                psi_forward.append(psi)
            else:
                psi_backward.append(psi)

            phase = "HEATING" if epoch < 15 else "COOLING"
            print(f"Epoch {epoch:2d} | {phase} | diversity = {diversity:.3f} | ψ = {psi:.3f}")

        # Analysis: Compute hysteresis loop area
        # Area = integral of (ψ_forward - ψ_backward) over diversity
        diversity_forward = diversity_schedule[:15]
        diversity_backward = diversity_schedule[15:][::-1]  # Reverse for comparison

        # Interpolate to common grid for area calculation
        common_diversity = np.linspace(0, 0.5, 100)
        psi_f_interp = np.interp(common_diversity, diversity_forward, psi_forward)
        psi_b_interp = np.interp(common_diversity, diversity_backward[::-1], psi_backward[::-1])

        loop_area = np.trapz(np.abs(psi_f_interp - psi_b_interp), common_diversity)

        print(f"\nHysteresis Loop Area: {loop_area:.4f}")

        if loop_area > 0.1:
            print(f"✅ HYPOTHESIS CONFIRMED: Significant hysteresis (area > 0.1)")
        elif loop_area > 0.05:
            print(f"🟡 HYPOTHESIS PARTIAL: Weak hysteresis (0.05 < area < 0.1)")
        else:
            print(f"❌ HYPOTHESIS REJECTED: No hysteresis (area < 0.05)")

        # Plot hysteresis loop
        fig, ax = plt.subplots(figsize=(10, 8))

        ax.plot(diversity_forward, psi_forward, 'b-o', linewidth=2, markersize=6, label='Forward (heating)')
        ax.plot(diversity_backward, psi_backward, 'r-s', linewidth=2, markersize=6, label='Backward (cooling)')
        ax.fill_betweenx(
            np.linspace(min(psi_forward + psi_backward), max(psi_forward + psi_backward), 100),
            np.interp(np.linspace(min(psi_forward + psi_backward), max(psi_forward + psi_backward), 100),
                     psi_forward, diversity_forward),
            np.interp(np.linspace(min(psi_forward + psi_backward), max(psi_forward + psi_backward), 100),
                     psi_backward, diversity_backward),
            alpha=0.2, color='purple', label=f'Hysteresis area = {loop_area:.3f}'
        )

        ax.set_xlabel('Diversity Weight (Control Parameter)')
        ax.set_ylabel('Order Parameter ψ')
        ax.set_title('Hysteresis Loop (First-Order Phase Transition)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plot_path = self.output_dir / 'hysteresis_loop.png'
        plt.savefig(plot_path, dpi=150)
        print(f"📊 Plot saved: {plot_path}")

        results = {
            'diversity_schedule': diversity_schedule,
            'psi_forward': psi_forward,
            'psi_backward': psi_backward,
            'loop_area': float(loop_area),
            'hypothesis_confirmed': loop_area > 0.1
        }

        self.results['hysteresis'] = results
        return results

    def test_scaling_exponent(
        self,
        model: FullChiralModel,
        train_loader,
        val_loader,
        device,
        critical_diversity: float = 0.3,
        epochs_per_point: int = 5
    ) -> Dict[str, any]:
        """
        Test Prediction 3: Power law scaling ψ ∝ (T - Tₖ)^β with β ≈ 0.5.

        Expected: Critical exponent β ∈ [0.3, 0.7] (mean-field universality)
        Null: Exponential decay (no power law)
        """
        print("\n" + "="*70)
        print("TEST 3: POWER LAW SCALING")
        print("="*70)

        # Test diversity values near critical point
        diversity_values = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        psi_values = []

        for diversity in diversity_values:
            # Re-initialize model for each point
            model_fresh = FullChiralModel(
                node_features=model.node_features,
                num_relations=model.num_relations,
                num_classes=model.num_classes,
                pool_ratio=model.pool_ratio,
                task_type='classification',
                dropout=0.1
            ).to(device)

            optimizer = torch.optim.Adam(model_fresh.parameters(), lr=1e-4)
            criterion = ChiralCompositeLoss(
                task_weight=1.0,
                aux_weight=0.3,
                cycle_weight=0.01,
                diversity_weight=diversity
            )

            # Train to equilibrium
            for epoch in range(epochs_per_point):
                model_fresh.train()
                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()

                    output = model_fresh(batch.x, batch.edge_index, batch.edge_type, batch.batch)
                    loss_dict = criterion(output, batch.y)
                    loss = loss_dict['loss']

                    loss.backward()
                    optimizer.step()

            # Measure final ψ
            model_fresh.eval()
            correct_0 = 0
            correct_1 = 0
            total_0 = 0
            total_1 = 0

            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    output = model_fresh(batch.x, batch.edge_index, batch.edge_type, batch.batch)
                    pred = output['logits'].argmax(dim=1)

                    for p, y in zip(pred, batch.y):
                        if y == 0:
                            total_0 += 1
                            if p == 0:
                                correct_0 += 1
                        else:
                            total_1 += 1
                            if p == 1:
                                correct_1 += 1

            acc_0 = correct_0 / max(total_0, 1)
            acc_1 = correct_1 / max(total_1, 1)
            psi = 1.0 - abs(acc_0 - acc_1)
            psi_values.append(psi)

            print(f"Diversity = {diversity:.2f} | ψ = {psi:.3f}")

        # Fit power law: ψ = A * (T - Tₖ)^β
        T_values = np.array(diversity_values)
        psi_array = np.array(psi_values)

        # Only fit points below critical (T < Tₖ)
        below_critical = T_values < critical_diversity
        if below_critical.sum() > 2:
            T_fit = T_values[below_critical]
            psi_fit = psi_array[below_critical]

            # Log-log fit: log(ψ) = log(A) + β * log(Tₖ - T)
            x = np.log(critical_diversity - T_fit)
            y = np.log(psi_fit + 1e-6)  # Avoid log(0)

            # Linear regression
            coeffs = np.polyfit(x, y, 1)
            beta = coeffs[0]
            log_A = coeffs[1]
            A = np.exp(log_A)

            # Compute R²
            y_pred = np.polyval(coeffs, x)
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - y.mean())**2)
            r_squared = 1 - ss_res / ss_tot

            print(f"\nPower Law Fit: ψ = {A:.3f} × (Tₖ - T)^{beta:.3f}")
            print(f"Critical exponent β = {beta:.3f}")
            print(f"R² = {r_squared:.3f}")

            if 0.3 <= beta <= 0.7 and r_squared > 0.8:
                print(f"✅ HYPOTHESIS CONFIRMED: Universal scaling (β ≈ 0.5, R² > 0.8)")
            elif 0.2 <= beta <= 0.8:
                print(f"🟡 HYPOTHESIS PARTIAL: Power law present but exponent off")
            else:
                print(f"❌ HYPOTHESIS REJECTED: No power law scaling")

        else:
            print("⚠️  Insufficient data below critical point")
            beta = None
            r_squared = None

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: ψ vs. diversity
        ax1 = axes[0]
        ax1.plot(diversity_values, psi_values, 'bo-', linewidth=2, markersize=8)
        ax1.axvline(critical_diversity, color='red', linestyle='--', label=f'Tₖ = {critical_diversity}')
        if beta is not None:
            T_theory = np.linspace(0.05, critical_diversity - 0.01, 100)
            psi_theory = A * (critical_diversity - T_theory)**beta
            ax1.plot(T_theory, psi_theory, 'r--', linewidth=2, label=f'ψ ∝ (Tₖ - T)^{beta:.2f}')
        ax1.set_xlabel('Diversity Weight T')
        ax1.set_ylabel('Order Parameter ψ')
        ax1.set_title('Order Parameter vs. Control Parameter')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Log-log plot
        if beta is not None:
            ax2 = axes[1]
            ax2.plot(x, y, 'go', markersize=8, label='Data')
            ax2.plot(x, y_pred, 'r--', linewidth=2, label=f'Fit: slope = {beta:.2f}')
            ax2.set_xlabel('log(Tₖ - T)')
            ax2.set_ylabel('log(ψ)')
            ax2.set_title(f'Power Law Validation (R² = {r_squared:.3f})')
            ax2.grid(True, alpha=0.3)
            ax2.legend()

        plt.tight_layout()
        plot_path = self.output_dir / 'scaling_exponent.png'
        plt.savefig(plot_path, dpi=150)
        print(f"📊 Plot saved: {plot_path}")

        results = {
            'diversity_values': diversity_values,
            'psi_values': psi_values,
            'critical_diversity': critical_diversity,
            'beta': float(beta) if beta is not None else None,
            'r_squared': float(r_squared) if r_squared is not None else None,
            'hypothesis_confirmed': (0.3 <= beta <= 0.7 and r_squared > 0.8) if beta is not None else False
        }

        self.results['scaling'] = results
        return results

    def run_all_tests(self):
        """Run all three validation tests."""
        print("\n" + "="*70)
        print("PHASE TRANSITION VALIDATION SUITE")
        print("="*70)
        print("\nBased on: analysis/additional_isomorphisms.md")
        print("Hypothesis: Neural collapse is a first-order phase transition\n")

        # Initialize device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Initialize model and data
        print("\nLoading Planning dataset...")
        train_dataset = PlanningTripleDataset(root="/tmp/planning", split="train", num_problems=1600)
        val_dataset = PlanningTripleDataset(root="/tmp/planning", split="val", num_problems=400)

        def pyg_collate(data_list):
            graphs = [item[0] for item in data_list]
            labels = torch.tensor([item[1].item() for item in data_list])
            batch = Batch.from_data_list(graphs)
            batch.y = labels
            return batch

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=pyg_collate)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=pyg_collate)

        # Get dataset properties
        sample = next(iter(train_loader))
        node_features = sample.x.size(1)
        num_relations = int(sample.edge_type.max().item()) + 1
        num_classes = 2

        print(f"Dataset: {node_features} features, {num_relations} relations, {num_classes} classes")

        # Test 1: Critical slowing
        print("\n[1/3] Testing critical slowing...")
        model1 = FullChiralModel(
            node_features=node_features,
            num_relations=num_relations,
            num_classes=num_classes,
            pool_ratio=0.5,
            task_type='classification',
            dropout=0.1
        ).to(device)
        self.test_critical_slowing(model1, train_loader, val_loader, device)

        # Test 2: Hysteresis
        print("\n[2/3] Testing hysteresis...")
        model2 = FullChiralModel(
            node_features=node_features,
            num_relations=num_relations,
            num_classes=num_classes,
            pool_ratio=0.5,
            task_type='classification',
            dropout=0.1
        ).to(device)
        self.test_hysteresis(model2, train_loader, val_loader, device)

        # Test 3: Scaling exponent
        print("\n[3/3] Testing power law scaling...")
        model3 = FullChiralModel(
            node_features=node_features,
            num_relations=num_relations,
            num_classes=num_classes,
            pool_ratio=0.5,
            task_type='classification',
            dropout=0.1
        ).to(device)
        self.test_scaling_exponent(model3, train_loader, val_loader, device)

        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)

        confirmed = [
            self.results['critical_slowing'].get('hypothesis_confirmed', False),
            self.results['hysteresis'].get('hypothesis_confirmed', False),
            self.results['scaling'].get('hypothesis_confirmed', False)
        ]

        print(f"Critical Slowing:  {'✅ CONFIRMED' if confirmed[0] else '❌ REJECTED'}")
        print(f"Hysteresis Loop:   {'✅ CONFIRMED' if confirmed[1] else '❌ REJECTED'}")
        print(f"Power Law Scaling: {'✅ CONFIRMED' if confirmed[2] else '❌ REJECTED'}")

        total_confirmed = sum(confirmed)
        print(f"\nOverall: {total_confirmed}/3 predictions confirmed")

        if total_confirmed == 3:
            print("\n🎯 STRONG EVIDENCE: Neural collapse is a first-order phase transition")
        elif total_confirmed == 2:
            print("\n🟡 MODERATE EVIDENCE: Phase transition behavior present but incomplete")
        else:
            print("\n⚠️  WEAK EVIDENCE: Phase transition hypothesis not well-supported")

        # Save results
        results_path = self.output_dir / 'validation_results.json'
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Results saved: {results_path}")


if __name__ == '__main__':
    validator = PhaseTransitionValidator()
    validator.run_all_tests()
