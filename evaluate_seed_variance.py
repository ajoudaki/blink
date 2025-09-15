#!/usr/bin/env python3
"""
Evaluate the best configuration with multiple seeds to assess variance.
"""

import subprocess
import logging
import json
import numpy as np
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_with_seed(seed, config_name="unified_optimal_final"):
    """Run training with a specific seed."""

    cmd = f"python train_with_early_stopping.py --config-name={config_name} seed={seed}"

    try:
        logger.info(f"  Running with seed {seed}...")
        start_time = time.time()
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        elapsed = time.time() - start_time

        # Parse output
        output_lines = result.stdout.split('\n')
        metrics = {
            'seed': seed,
            'val_acc': None,
            'test_acc': None,
            'val_loss': None,
            'test_loss': None,
            'best_epoch': None,
            'elapsed_time': elapsed
        }

        # Also collect per-target metrics
        val_accs = []
        test_accs = []

        for i, line in enumerate(output_lines):
            if 'Average Test Accuracy:' in line:
                metrics['test_acc'] = float(line.split(':')[1].strip())
            elif 'Average Val Accuracy:' in line:
                metrics['val_acc'] = float(line.split(':')[1].strip())
            elif 'Average Test CE Loss:' in line:
                metrics['test_loss'] = float(line.split(':')[1].strip())
            elif 'Average Val CE Loss:' in line:
                metrics['val_loss'] = float(line.split(':')[1].strip())
            elif 'Best Epoch:' in line and 'Task' not in output_lines[i-1]:
                try:
                    metrics['best_epoch'] = int(line.split(':')[1].strip())
                except:
                    pass
            elif 'Accuracy:' in line and 'Average' not in line:
                # Capture individual target accuracies
                try:
                    acc = float(line.split(':')[1].strip())
                    if i > 0:
                        if 'VALIDATION SET' in '\n'.join(output_lines[max(0,i-10):i]):
                            val_accs.append(acc)
                        elif 'TEST SET' in '\n'.join(output_lines[max(0,i-10):i]):
                            test_accs.append(acc)
                except:
                    pass

        if val_accs:
            metrics['val_acc_targets'] = val_accs
        if test_accs:
            metrics['test_acc_targets'] = test_accs

        if metrics['val_acc'] and metrics['test_acc']:
            logger.info(f"    ✓ Seed {seed}: Val={metrics['val_acc']:.4f}, Test={metrics['test_acc']:.4f}, "
                       f"Time={elapsed:.1f}s")
            return metrics
        else:
            logger.error(f"    ✗ Failed to parse results for seed {seed}")
            return None

    except subprocess.TimeoutExpired:
        logger.error(f"    ✗ Timeout for seed {seed}")
        return None
    except Exception as e:
        logger.error(f"    ✗ Error for seed {seed}: {e}")
        return None


def calculate_statistics(results):
    """Calculate mean, std, min, max for the results."""

    val_accs = [r['val_acc'] for r in results if r['val_acc']]
    test_accs = [r['test_acc'] for r in results if r['test_acc']]
    val_losses = [r['val_loss'] for r in results if r['val_loss']]
    test_losses = [r['test_loss'] for r in results if r['test_loss']]
    epochs = [r['best_epoch'] for r in results if r['best_epoch']]

    stats = {
        'validation': {
            'accuracies': val_accs,
            'mean': np.mean(val_accs),
            'std': np.std(val_accs),
            'min': np.min(val_accs),
            'max': np.max(val_accs),
            'range': np.max(val_accs) - np.min(val_accs),
            'cv': np.std(val_accs) / np.mean(val_accs) * 100  # Coefficient of variation
        },
        'test': {
            'accuracies': test_accs,
            'mean': np.mean(test_accs),
            'std': np.std(test_accs),
            'min': np.min(test_accs),
            'max': np.max(test_accs),
            'range': np.max(test_accs) - np.min(test_accs),
            'cv': np.std(test_accs) / np.mean(test_accs) * 100
        },
        'val_loss': {
            'mean': np.mean(val_losses),
            'std': np.std(val_losses)
        },
        'test_loss': {
            'mean': np.mean(test_losses),
            'std': np.std(test_losses)
        },
        'best_epoch': {
            'mean': np.mean(epochs) if epochs else None,
            'std': np.std(epochs) if epochs else None
        },
        'generalization_gap': {
            'mean': np.mean([v - t for v, t in zip(val_accs, test_accs)]),
            'std': np.std([v - t for v, t in zip(val_accs, test_accs)])
        }
    }

    return stats


# Main execution
logger.info("="*70)
logger.info("SEED VARIANCE EVALUATION")
logger.info("="*70)
logger.info("Testing best configuration with 5 different seeds")
logger.info("Config: LR=0.003, Dropout=0.1, Weight Decay=0.0, ReLU")
logger.info("")

# Seeds to test
seeds = [42, 123, 456, 789, 2024]

logger.info(f"Seeds to test: {seeds}")
logger.info("")

results = []
for seed in seeds:
    result = run_with_seed(seed)
    if result:
        results.append(result)
    time.sleep(2)  # Brief pause between runs

if len(results) < len(seeds):
    logger.warning(f"Only {len(results)}/{len(seeds)} runs completed successfully")

# Calculate statistics
stats = calculate_statistics(results)

# Report results
logger.info("")
logger.info("="*70)
logger.info("INDIVIDUAL RESULTS")
logger.info("="*70)
logger.info(f"{'Seed':<8} {'Val Acc':<10} {'Test Acc':<10} {'Val Loss':<10} {'Test Loss':<10} {'Gap':<8}")
logger.info("-"*70)

for r in results:
    gap = r['val_acc'] - r['test_acc']
    logger.info(f"{r['seed']:<8} {r['val_acc']:<10.4f} {r['test_acc']:<10.4f} "
               f"{r['val_loss']:<10.4f} {r['test_loss']:<10.4f} {gap:<8.4f}")

logger.info("")
logger.info("="*70)
logger.info("STATISTICAL ANALYSIS")
logger.info("="*70)

logger.info("\nValidation Accuracy:")
logger.info(f"  Mean ± Std: {stats['validation']['mean']:.4f} ± {stats['validation']['std']:.4f}")
logger.info(f"  Range: [{stats['validation']['min']:.4f}, {stats['validation']['max']:.4f}]")
logger.info(f"  Spread: {stats['validation']['range']:.4f} ({stats['validation']['cv']:.2f}% CV)")

logger.info("\nTest Accuracy:")
logger.info(f"  Mean ± Std: {stats['test']['mean']:.4f} ± {stats['test']['std']:.4f}")
logger.info(f"  Range: [{stats['test']['min']:.4f}, {stats['test']['max']:.4f}]")
logger.info(f"  Spread: {stats['test']['range']:.4f} ({stats['test']['cv']:.2f}% CV)")

logger.info("\nGeneralization Gap (Val - Test):")
logger.info(f"  Mean ± Std: {stats['generalization_gap']['mean']:.4f} ± {stats['generalization_gap']['std']:.4f}")

logger.info("\nValidation Loss:")
logger.info(f"  Mean ± Std: {stats['val_loss']['mean']:.4f} ± {stats['val_loss']['std']:.4f}")

logger.info("\nTest Loss:")
logger.info(f"  Mean ± Std: {stats['test_loss']['mean']:.4f} ± {stats['test_loss']['std']:.4f}")

if stats['best_epoch']['mean']:
    logger.info(f"\nBest Epoch:")
    logger.info(f"  Mean ± Std: {stats['best_epoch']['mean']:.1f} ± {stats['best_epoch']['std']:.1f}")

# Confidence intervals (assuming normal distribution)
logger.info("")
logger.info("="*70)
logger.info("95% CONFIDENCE INTERVALS")
logger.info("="*70)

# 95% CI = mean ± 1.96 * std / sqrt(n)
n = len(results)
z = 1.96  # 95% confidence

val_ci = z * stats['validation']['std'] / np.sqrt(n)
test_ci = z * stats['test']['std'] / np.sqrt(n)

logger.info(f"Validation: {stats['validation']['mean']:.4f} ± {val_ci:.4f} "
           f"[{stats['validation']['mean']-val_ci:.4f}, {stats['validation']['mean']+val_ci:.4f}]")
logger.info(f"Test:       {stats['test']['mean']:.4f} ± {test_ci:.4f} "
           f"[{stats['test']['mean']-test_ci:.4f}, {stats['test']['mean']+test_ci:.4f}]")

# Save results
output = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'config': 'LR=0.003, Dropout=0.1, Weight Decay=0.0, ReLU',
    'seeds': seeds,
    'individual_results': results,
    'statistics': {k: {kk: float(vv) if vv is not None else None
                       for kk, vv in v.items() if kk != 'accuracies'}
                  for k, v in stats.items()},
    'confidence_intervals': {
        'validation': {
            'mean': stats['validation']['mean'],
            'ci': val_ci,
            'lower': stats['validation']['mean'] - val_ci,
            'upper': stats['validation']['mean'] + val_ci
        },
        'test': {
            'mean': stats['test']['mean'],
            'ci': test_ci,
            'lower': stats['test']['mean'] - test_ci,
            'upper': stats['test']['mean'] + test_ci
        }
    }
}

with open('seed_variance_results.json', 'w') as f:
    json.dump(output, f, indent=2)

logger.info("")
logger.info(f"Results saved to seed_variance_results.json")
logger.info("="*70)

# Final summary
logger.info("\nFINAL SUMMARY:")
logger.info(f"  Model is {'STABLE' if stats['test']['cv'] < 2 else 'MODERATELY STABLE' if stats['test']['cv'] < 5 else 'UNSTABLE'}")
logger.info(f"  Test accuracy: {stats['test']['mean']:.2f}% ± {stats['test']['std']:.2f}%")
logger.info(f"  Expected range: [{stats['test']['mean']-2*stats['test']['std']:.2f}%, "
           f"{stats['test']['mean']+2*stats['test']['std']:.2f}%] (95% of runs)")
logger.info("="*70)