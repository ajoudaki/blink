#!/usr/bin/env python3
"""
Iterative optimization of hyperparameters for the best ReLU configuration.
Optimize based on validation score, report test score.
"""

import subprocess
import logging
import json
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_experiment(param_name, param_value, base_config_overrides=""):
    """Run a single experiment with given parameter."""

    # Build command with overrides
    override = f"{base_config_overrides} {param_name}={param_value}" if base_config_overrides else f"{param_name}={param_value}"
    cmd = f"python train_with_early_stopping.py --config-name=unified_best_relu {override}"

    try:
        logger.info(f"  Testing {param_name}={param_value}...")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)

        # Parse output
        output_lines = result.stdout.split('\n')
        metrics = {
            'param': param_name,
            'value': param_value,
            'val_acc': None,
            'test_acc': None,
            'val_loss': None,
            'test_loss': None,
            'best_epoch': None
        }

        for line in output_lines:
            if 'Average Test Accuracy:' in line:
                metrics['test_acc'] = float(line.split(':')[1].strip())
            elif 'Average Val Accuracy:' in line:
                metrics['val_acc'] = float(line.split(':')[1].strip())
            elif 'Average Test CE Loss:' in line:
                metrics['test_loss'] = float(line.split(':')[1].strip())
            elif 'Average Val CE Loss:' in line:
                metrics['val_loss'] = float(line.split(':')[1].strip())
            elif 'Best Epoch:' in line and 'best' in line.lower():
                try:
                    # Extract epoch number from "Best model was at epoch X"
                    epoch_str = line.split('epoch')[1].split()[0]
                    metrics['best_epoch'] = int(epoch_str)
                except:
                    pass

        if metrics['val_acc']:
            logger.info(f"    ✓ Val: {metrics['val_acc']:.4f}, Test: {metrics['test_acc']:.4f}, Epoch: {metrics['best_epoch']}")
            return metrics
        else:
            logger.error(f"    ✗ Failed to parse results")
            return None

    except subprocess.TimeoutExpired:
        logger.error(f"    ✗ Timeout")
        return None
    except Exception as e:
        logger.error(f"    ✗ Error: {e}")
        return None


def optimize_parameter(param_name, param_values, base_overrides="", selection_metric='val_acc'):
    """Optimize a single parameter, return best value and results."""

    logger.info(f"\n{'='*70}")
    logger.info(f"Optimizing {param_name}")
    logger.info(f"{'='*70}")

    if base_overrides:
        logger.info(f"Base configuration: {base_overrides}")

    results = []
    for value in param_values:
        result = run_experiment(param_name, value, base_overrides)
        if result:
            results.append(result)
        time.sleep(2)  # Brief pause

    if not results:
        return None, []

    # Sort by validation accuracy (descending) or loss (ascending)
    if 'loss' in selection_metric:
        results.sort(key=lambda x: x[selection_metric])
        best = results[0]
    else:
        results.sort(key=lambda x: x[selection_metric], reverse=True)
        best = results[0]

    # Report results
    logger.info(f"\nResults for {param_name}:")
    logger.info(f"{'Value':<12} {'Val Acc':<8} {'Test Acc':<8} {'Val Loss':<8} {'Best Epoch':<10}")
    logger.info("-" * 60)

    for r in results:
        epoch_str = str(r['best_epoch']) if r['best_epoch'] else 'N/A'
        logger.info(f"{str(r['value']):<12} {r['val_acc']:<8.4f} {r['test_acc']:<8.4f} "
                   f"{r['val_loss']:<8.4f} {epoch_str:<10}")

    logger.info(f"\n✓ Best {param_name}: {best['value']} (Val Acc: {best['val_acc']:.4f}, Test: {best['test_acc']:.4f})")

    return best['value'], results


# Main optimization
logger.info("="*70)
logger.info("ITERATIVE HYPERPARAMETER OPTIMIZATION")
logger.info("="*70)
logger.info("Starting from best ReLU configuration")
logger.info("Optimizing based on VALIDATION accuracy")
logger.info("Reporting TEST accuracy (untouched during selection)")
logger.info("")

all_results = {}

# Step 1: Optimize Learning Rate (push as high as possible)
logger.info("\nSTEP 1: LEARNING RATE OPTIMIZATION")
logger.info("Testing progressively higher learning rates...")

lr_values = [
    0.0005,   # Lower baseline
    0.001,    # Current best
    0.002,    # 2x
    0.003,    # 3x
    0.005,    # 5x
    0.007,    # 7x
    0.01,     # 10x
    0.015,    # 15x
    0.02,     # 20x
    0.03,     # 30x
    0.05,     # 50x
]

best_lr, lr_results = optimize_parameter('training.learning_rate', lr_values)
all_results['learning_rate'] = lr_results

if best_lr is None:
    logger.error("Failed to optimize learning rate")
    exit(1)

# Step 2: Optimize Dropout (with best LR)
logger.info("\nSTEP 2: DROPOUT OPTIMIZATION")
logger.info(f"Using best learning rate: {best_lr}")

dropout_values = [
    0.0,      # No dropout
    0.05,     # Very light
    0.1,      # Light
    0.15,     # Current best
    0.2,      # Higher
    0.25,     # Higher
    0.3,      # High
    0.4,      # Very high
    0.5,      # Maximum reasonable
]

best_dropout, dropout_results = optimize_parameter(
    'model.dropout',
    dropout_values,
    f"training.learning_rate={best_lr}"
)
all_results['dropout'] = dropout_results

if best_dropout is None:
    logger.error("Failed to optimize dropout")
    best_dropout = 0.15  # Use default

# Step 3: Optimize Weight Decay (with best LR and dropout)
logger.info("\nSTEP 3: WEIGHT DECAY OPTIMIZATION")
logger.info(f"Using best LR: {best_lr}, best dropout: {best_dropout}")

wd_values = [
    0.0,       # No weight decay
    0.0001,    # Very light
    0.0005,    # Light
    0.001,     # Current best
    0.002,     # 2x
    0.005,     # 5x
    0.01,      # 10x
    0.02,      # 20x
    0.05,      # High
]

best_wd, wd_results = optimize_parameter(
    'training.weight_decay',
    wd_values,
    f"training.learning_rate={best_lr} model.dropout={best_dropout}"
)
all_results['weight_decay'] = wd_results

# Final verification with best configuration
logger.info("\n" + "="*70)
logger.info("FINAL VERIFICATION")
logger.info("="*70)

final_config = f"training.learning_rate={best_lr} model.dropout={best_dropout} training.weight_decay={best_wd}"
logger.info(f"Best configuration: {final_config}")
logger.info("Running final verification...")

final_result = run_experiment('training.epochs', 200, final_config)

# Save all results
output = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'best_configuration': {
        'learning_rate': best_lr,
        'dropout': best_dropout,
        'weight_decay': best_wd
    },
    'optimization_results': all_results,
    'final_verification': final_result
}

with open('final_optimization_results.json', 'w') as f:
    json.dump(output, f, indent=2)

# Final report
logger.info("\n" + "="*70)
logger.info("OPTIMIZATION COMPLETE")
logger.info("="*70)
logger.info(f"Best Learning Rate: {best_lr}")
logger.info(f"Best Dropout: {best_dropout}")
logger.info(f"Best Weight Decay: {best_wd}")

if final_result:
    logger.info(f"\nFinal Performance:")
    logger.info(f"  Validation Accuracy: {final_result['val_acc']:.4f}")
    logger.info(f"  Test Accuracy: {final_result['test_acc']:.4f}")
    logger.info(f"  Generalization Gap: {abs(final_result['val_acc'] - final_result['test_acc']):.4f}")

logger.info(f"\nResults saved to final_optimization_results.json")
logger.info("="*70)