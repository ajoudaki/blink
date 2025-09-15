#!/usr/bin/env python3
"""
Verify the best model configuration by running it multiple times.
"""

import subprocess
import logging
import json
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_model(run_number):
    """Run the best model configuration."""

    cmd = "python train.py --config-name=unified_best_final"

    try:
        logger.info(f"  Starting run {run_number}...")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)

        # Parse output
        output_lines = result.stdout.split('\n')

        # Extract metrics
        metrics = {
            'attractive': {'acc': None, 'loss': None},
            'smart': {'acc': None, 'loss': None},
            'trustworthy': {'acc': None, 'loss': None},
            'average': {'acc': None, 'loss': None}
        }

        current_target = None
        for line in output_lines:
            # Check for target sections
            if 'ATTRACTIVE:' in line:
                current_target = 'attractive'
            elif 'SMART:' in line:
                current_target = 'smart'
            elif 'TRUSTWORTHY:' in line:
                current_target = 'trustworthy'

            # Extract accuracy and loss for current target
            if current_target and 'Accuracy:' in line:
                acc = float(line.split(':')[1].strip())
                metrics[current_target]['acc'] = acc
            elif current_target and 'CE Loss:' in line:
                loss = float(line.split(':')[1].strip())
                metrics[current_target]['loss'] = loss
                current_target = None  # Reset after getting both metrics

            # Get average metrics
            if 'Average Accuracy:' in line:
                metrics['average']['acc'] = float(line.split(':')[1].strip())
            elif 'Average CE Loss:' in line:
                metrics['average']['loss'] = float(line.split(':')[1].strip())

        return metrics

    except subprocess.TimeoutExpired:
        logger.error(f"  Run {run_number} timed out")
        return None
    except Exception as e:
        logger.error(f"  Run {run_number} failed: {e}")
        return None


def main():
    """Run verification."""

    logger.info("="*70)
    logger.info("VERIFYING BEST MODEL CONFIGURATION")
    logger.info("="*70)
    logger.info("Config: unified_best_final.yaml")
    logger.info("Architecture: [384, 256, 128, 64, 32]")
    logger.info("Dropout: 0.15, Weight Decay: 0.001, LR: 0.001")
    logger.info("-"*70)

    # Run model multiple times
    n_runs = 3
    all_results = []

    for i in range(1, n_runs + 1):
        logger.info(f"\nRun {i}/{n_runs}:")
        metrics = run_model(i)

        if metrics and metrics['average']['acc']:
            all_results.append(metrics)
            logger.info(f"  ✓ Average Accuracy: {metrics['average']['acc']:.4f}")
            logger.info(f"    Average CE Loss: {metrics['average']['loss']:.4f}")
            logger.info(f"    Per-target accuracy:")
            logger.info(f"      Attractive: {metrics['attractive']['acc']:.4f}")
            logger.info(f"      Smart: {metrics['smart']['acc']:.4f}")
            logger.info(f"      Trustworthy: {metrics['trustworthy']['acc']:.4f}")
        else:
            logger.error(f"  ✗ Failed to get results")

    # Calculate statistics
    if all_results:
        logger.info("\n" + "="*70)
        logger.info("VERIFICATION RESULTS")
        logger.info("="*70)

        # Extract average accuracies
        avg_accs = [r['average']['acc'] for r in all_results]
        avg_losses = [r['average']['loss'] for r in all_results]

        logger.info(f"\nRuns completed: {len(all_results)}/{n_runs}")
        logger.info(f"\nAverage Accuracy across runs:")
        logger.info(f"  Mean: {np.mean(avg_accs):.4f}")
        logger.info(f"  Std:  {np.std(avg_accs):.4f}")
        logger.info(f"  Min:  {np.min(avg_accs):.4f}")
        logger.info(f"  Max:  {np.max(avg_accs):.4f}")

        logger.info(f"\nAverage CE Loss across runs:")
        logger.info(f"  Mean: {np.mean(avg_losses):.4f}")
        logger.info(f"  Std:  {np.std(avg_losses):.4f}")

        # Per-target statistics
        logger.info("\nPer-target accuracy (mean across runs):")
        for target in ['attractive', 'smart', 'trustworthy']:
            target_accs = [r[target]['acc'] for r in all_results]
            logger.info(f"  {target.capitalize()}: {np.mean(target_accs):.4f} ± {np.std(target_accs):.4f}")

        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"best_model_verification_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump({
                'config': 'unified_best_final.yaml',
                'architecture': [384, 256, 128, 64, 32],
                'dropout': 0.15,
                'weight_decay': 0.001,
                'learning_rate': 0.001,
                'runs': all_results,
                'summary': {
                    'mean_accuracy': float(np.mean(avg_accs)),
                    'std_accuracy': float(np.std(avg_accs)),
                    'mean_loss': float(np.mean(avg_losses)),
                    'std_loss': float(np.std(avg_losses))
                }
            }, f, indent=2)

        logger.info(f"\n✓ Results saved to {results_file}")

        # Final verdict
        logger.info("\n" + "="*70)
        if np.mean(avg_accs) >= 0.72:
            logger.info("✓ VERIFICATION SUCCESSFUL")
            logger.info(f"  The model consistently achieves {np.mean(avg_accs):.4f} accuracy")
        else:
            logger.info("⚠ VERIFICATION SHOWS LOWER PERFORMANCE")
            logger.info(f"  Expected: ~73.4%, Got: {np.mean(avg_accs):.4f}")
        logger.info("="*70)

    else:
        logger.error("\n✗ No successful runs completed")


if __name__ == "__main__":
    main()