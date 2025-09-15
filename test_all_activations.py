#!/usr/bin/env python3
"""
Comprehensive test of ALL popular activation functions.
"""

import subprocess
import logging
import json
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_activation(activation_name, description):
    """Test a specific activation function."""

    # Create config with the activation
    config_override = f"model.activation={activation_name}"

    cmd = f"python train_with_early_stopping.py --config-name=unified_best_early_stop {config_override}"

    try:
        logger.info(f"Testing {description}...")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)

        # Parse output for test accuracy
        output_lines = result.stdout.split('\n')
        test_acc = None
        val_acc = None
        test_loss = None
        val_loss = None
        best_epoch = None

        for i, line in enumerate(output_lines):
            if 'Average Test Accuracy:' in line:
                test_acc = float(line.split(':')[1].strip())
            elif 'Average Val Accuracy:' in line:
                val_acc = float(line.split(':')[1].strip())
            elif 'Average Test CE Loss:' in line:
                test_loss = float(line.split(':')[1].strip())
            elif 'Average Val CE Loss:' in line:
                val_loss = float(line.split(':')[1].strip())
            elif 'Best Epoch:' in line and 'epoch' not in line.lower():
                try:
                    best_epoch = int(line.split(':')[1].strip())
                except:
                    pass

        if test_acc:
            logger.info(f"  ✓ Val: {val_acc:.4f}, Test: {test_acc:.4f}, Epoch: {best_epoch if best_epoch else 'N/A'}")
            return {
                'activation': activation_name,
                'description': description,
                'val_accuracy': val_acc,
                'test_accuracy': test_acc,
                'val_loss': val_loss,
                'test_loss': test_loss,
                'best_epoch': best_epoch,
                'generalization_gap': abs(val_acc - test_acc)
            }
        else:
            logger.error(f"  ✗ Failed to parse results")
            return None

    except subprocess.TimeoutExpired:
        logger.error(f"  ✗ Timeout after 5 minutes")
        return None
    except Exception as e:
        logger.error(f"  ✗ Error: {e}")
        return None


# Comprehensive list of activation functions
logger.info("="*70)
logger.info("COMPREHENSIVE ACTIVATION FUNCTION SWEEP")
logger.info("="*70)
logger.info("Testing ALL popular activation functions with early stopping")
logger.info("Base config: 5-layer [384,256,128,64,32], dropout=0.15, wd=0.001")
logger.info("")

activations_to_test = [
    # Already tested high performers
    ('leaky_relu', 'LeakyReLU'),
    ('relu', 'ReLU'),

    # Smooth activations
    ('gelu', 'GELU'),
    ('silu', 'SiLU/Swish'),
    ('mish', 'Mish'),
    ('softplus', 'Softplus'),

    # Self-normalizing
    ('selu', 'SELU'),

    # Exponential family
    ('elu', 'ELU'),

    # Bounded activations
    ('tanh', 'Tanh'),
    ('sigmoid', 'Sigmoid'),
    ('hardsigmoid', 'HardSigmoid'),
    ('hardtanh', 'HardTanh'),
    ('hardswish', 'HardSwish'),

    # Rectified variants
    ('relu6', 'ReLU6'),
    ('prelu', 'PReLU'),
    ('rrelu', 'RReLU'),

    # Special cases
    ('none', 'None (Linear)'),
    ('celu', 'CELU'),
    ('logsigmoid', 'LogSigmoid'),
]

results = []
tested = set()

for activation, description in activations_to_test:
    if activation not in tested:
        result = test_activation(activation, description)
        if result:
            results.append(result)
            tested.add(activation)
        time.sleep(2)  # Brief pause between runs

# Sort by test accuracy
results.sort(key=lambda x: x['test_accuracy'], reverse=True)

# Report results
logger.info("")
logger.info("="*70)
logger.info("FINAL RESULTS (sorted by test accuracy)")
logger.info("="*70)
logger.info("")
logger.info("Top 10 Activation Functions:")
logger.info("-"*70)

for i, r in enumerate(results[:10], 1):
    gap = r['generalization_gap']
    logger.info(f"{i:2d}. {r['description']:15s}: Test={r['test_accuracy']:.4f}, Val={r['val_accuracy']:.4f}, "
                f"Gap={gap:.4f}")

# Detailed analysis
logger.info("")
logger.info("-"*70)
logger.info("ANALYSIS:")
logger.info("-"*70)

if results:
    best = results[0]
    worst = results[-1]

    logger.info(f"Best:  {best['description']:15s} - Test Accuracy: {best['test_accuracy']:.4f}")
    logger.info(f"Worst: {worst['description']:15s} - Test Accuracy: {worst['test_accuracy']:.4f}")
    logger.info(f"Range: {(best['test_accuracy'] - worst['test_accuracy'])*100:.2f}% difference")

    # Compare top performers
    logger.info("")
    logger.info("Top 3 Performers:")
    for i, r in enumerate(results[:3], 1):
        logger.info(f"  {i}. {r['description']}: {r['test_accuracy']:.4f}")

    # Find best generalization
    results_by_gap = sorted(results, key=lambda x: x['generalization_gap'])
    logger.info("")
    logger.info("Best Generalization (smallest val-test gap):")
    for r in results_by_gap[:3]:
        logger.info(f"  {r['description']}: Gap={r['generalization_gap']:.4f}")

# Save comprehensive results
output_file = 'comprehensive_activation_results.json'
with open(output_file, 'w') as f:
    json.dump({
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': results,
        'best_activation': results[0] if results else None,
        'summary': {
            'total_tested': len(results),
            'best_test_accuracy': results[0]['test_accuracy'] if results else None,
            'worst_test_accuracy': results[-1]['test_accuracy'] if results else None,
        }
    }, f, indent=2)

logger.info("")
logger.info(f"Complete results saved to {output_file}")
logger.info("="*70)