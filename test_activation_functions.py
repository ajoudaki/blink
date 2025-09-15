#!/usr/bin/env python3
"""
Test different activation functions to find the best one for test accuracy.
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
                best_epoch = int(line.split(':')[1].strip())

        if test_acc:
            logger.info(f"  ✓ Val: {val_acc:.4f}, Test: {test_acc:.4f}, Best Epoch: {best_epoch}")
            return {
                'activation': activation_name,
                'description': description,
                'val_accuracy': val_acc,
                'test_accuracy': test_acc,
                'val_loss': val_loss,
                'test_loss': test_loss,
                'best_epoch': best_epoch
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


# Test different activation functions
logger.info("="*70)
logger.info("ACTIVATION FUNCTION SWEEP")
logger.info("="*70)
logger.info("Testing different activation functions with early stopping")
logger.info("Base config: 5-layer [384,256,128,64,32], dropout=0.15, wd=0.001")
logger.info("")

activations_to_test = [
    ('relu', 'ReLU (baseline)'),
    ('gelu', 'GELU (current best)'),
    ('tanh', 'Tanh'),
    ('leaky_relu', 'LeakyReLU'),
    ('none', 'None (linear only)'),
    ('silu', 'SiLU/Swish'),
    ('elu', 'ELU'),
]

results = []

for activation, description in activations_to_test:
    result = test_activation(activation, description)
    if result:
        results.append(result)
    time.sleep(2)  # Brief pause between runs

# Sort by test accuracy
results.sort(key=lambda x: x['test_accuracy'], reverse=True)

# Report results
logger.info("")
logger.info("="*70)
logger.info("RESULTS SUMMARY (sorted by test accuracy)")
logger.info("="*70)

for i, r in enumerate(results, 1):
    logger.info(f"{i}. {r['description']:20s}: Val={r['val_accuracy']:.4f}, Test={r['test_accuracy']:.4f}, "
                f"Gap={abs(r['val_accuracy']-r['test_accuracy']):.4f}, Epoch={r['best_epoch']}")

if results:
    best = results[0]
    logger.info("")
    logger.info("-"*70)
    logger.info("BEST ACTIVATION FUNCTION:")
    logger.info(f"  {best['description']}: Test Accuracy = {best['test_accuracy']:.4f}")

    # Compare with GELU baseline
    gelu_result = next((r for r in results if r['activation'] == 'gelu'), None)
    if gelu_result and best['activation'] != 'gelu':
        improvement = (best['test_accuracy'] - gelu_result['test_accuracy']) * 100
        if improvement > 0:
            logger.info(f"  ✓ Improvement over GELU: +{improvement:.2f}%")
        else:
            logger.info(f"  ✗ No improvement over GELU")

# Save results
with open('activation_sweep_results.json', 'w') as f:
    json.dump(results, f, indent=2)
logger.info("")
logger.info(f"Results saved to activation_sweep_results.json")
logger.info("="*70)