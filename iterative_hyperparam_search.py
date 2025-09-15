#!/usr/bin/env python3
"""
Iterative hyperparameter search - greedy approach.
Start with best config found so far and optimize one variable at a time.
"""

import subprocess
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_config(name, args, timeout=120):
    """Run a configuration and return accuracy."""
    try:
        cmd = f"python train.py {args}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)

        # Parse output
        output_lines = result.stdout.split('\n')
        avg_acc = None
        avg_loss = None

        for line in output_lines:
            if 'Average Accuracy:' in line:
                avg_acc = float(line.split(':')[1].strip())
            elif 'Average CE Loss:' in line:
                avg_loss = float(line.split(':')[1].strip())

        if avg_acc:
            logger.info(f"  {name}: Acc={avg_acc:.4f}, Loss={avg_loss:.3f}")
            return avg_acc, avg_loss
        else:
            logger.info(f"  {name}: Failed to parse")
            return None, None

    except subprocess.TimeoutExpired:
        logger.info(f"  {name}: Timeout")
        return None, None
    except Exception as e:
        logger.info(f"  {name}: Error - {e}")
        return None, None


# Start with the best config found so far
base_config = '--config-name=unified_best task_type=comparison model.hidden_dims=[384,256,128,64,32]'
best_acc = 0.720
best_config = base_config
best_name = "DeepNarrow baseline"

logger.info("="*70)
logger.info("ITERATIVE HYPERPARAMETER OPTIMIZATION")
logger.info("="*70)
logger.info(f"Starting config: {best_name}")
logger.info(f"Starting accuracy: {best_acc:.4f}")

# 1. LEARNING RATE (highest impact)
logger.info("\n" + "="*70)
logger.info("STEP 1: OPTIMIZING LEARNING RATE")
logger.info("-"*40)

lr_values = [0.0005, 0.001, 0.002, 0.005, 0.01]
lr_results = []

for lr in lr_values:
    epochs = 50 if lr <= 0.005 else 30  # Fewer epochs for very high LR
    config = f"{best_config} training.epochs={epochs} +optimizer.lr={lr}"
    if lr > 0.002:
        config += " +optimizer.grad_clip=1.0"  # Add gradient clipping for high LR

    acc, loss = run_config(f"LR={lr}", config)
    if acc:
        lr_results.append((lr, acc, loss))

if lr_results:
    lr_results.sort(key=lambda x: x[1], reverse=True)
    best_lr, best_lr_acc, best_lr_loss = lr_results[0]

    if best_lr_acc > best_acc:
        logger.info(f"✓ IMPROVED! LR={best_lr} achieved {best_lr_acc:.4f} (was {best_acc:.4f})")
        best_acc = best_lr_acc
        best_config = f"{best_config} +optimizer.lr={best_lr}"
        if best_lr > 0.002:
            best_config += " +optimizer.grad_clip=1.0"
        best_name = f"LR={best_lr}"
    else:
        logger.info(f"✗ No improvement. Best LR={best_lr} got {best_lr_acc:.4f} vs current {best_acc:.4f}")

# 2. DROPOUT (important for generalization)
logger.info("\n" + "="*70)
logger.info("STEP 2: OPTIMIZING DROPOUT")
logger.info("-"*40)

dropout_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
dropout_results = []

for dropout in dropout_values:
    config = f"{best_config} training.epochs=50 model.dropout={dropout}"
    acc, loss = run_config(f"Dropout={dropout}", config)
    if acc:
        dropout_results.append((dropout, acc, loss))

if dropout_results:
    dropout_results.sort(key=lambda x: x[1], reverse=True)
    best_dropout, best_dropout_acc, best_dropout_loss = dropout_results[0]

    if best_dropout_acc > best_acc:
        logger.info(f"✓ IMPROVED! Dropout={best_dropout} achieved {best_dropout_acc:.4f} (was {best_acc:.4f})")
        best_acc = best_dropout_acc
        best_config = f"{best_config} model.dropout={best_dropout}"
        best_name = f"{best_name}+Dropout={best_dropout}"
    else:
        logger.info(f"✗ No improvement. Best dropout={best_dropout} got {best_dropout_acc:.4f} vs current {best_acc:.4f}")

# 3. WEIGHT DECAY
logger.info("\n" + "="*70)
logger.info("STEP 3: OPTIMIZING WEIGHT DECAY")
logger.info("-"*40)

wd_values = [0.001, 0.01, 0.05, 0.1]
wd_results = []

for wd in wd_values:
    config = f"{best_config} training.epochs=50 +optimizer.weight_decay={wd}"
    acc, loss = run_config(f"WD={wd}", config)
    if acc:
        wd_results.append((wd, acc, loss))

if wd_results:
    wd_results.sort(key=lambda x: x[1], reverse=True)
    best_wd, best_wd_acc, best_wd_loss = wd_results[0]

    if best_wd_acc > best_acc:
        logger.info(f"✓ IMPROVED! WD={best_wd} achieved {best_wd_acc:.4f} (was {best_acc:.4f})")
        best_acc = best_wd_acc
        best_config = f"{best_config} +optimizer.weight_decay={best_wd}"
        best_name = f"{best_name}+WD={best_wd}"
    else:
        logger.info(f"✗ No improvement. Best WD={best_wd} got {best_wd_acc:.4f} vs current {best_acc:.4f}")

# 4. EPOCHS (check if we need more training)
logger.info("\n" + "="*70)
logger.info("STEP 4: OPTIMIZING TRAINING EPOCHS")
logger.info("-"*40)

epoch_values = [50, 75, 100]
epoch_results = []

for epochs in epoch_values:
    config = f"{best_config} training.epochs={epochs}"
    acc, loss = run_config(f"Epochs={epochs}", config, timeout=180)
    if acc:
        epoch_results.append((epochs, acc, loss))

if epoch_results:
    epoch_results.sort(key=lambda x: x[1], reverse=True)
    best_epochs, best_epochs_acc, best_epochs_loss = epoch_results[0]

    if best_epochs_acc > best_acc:
        logger.info(f"✓ IMPROVED! Epochs={best_epochs} achieved {best_epochs_acc:.4f} (was {best_acc:.4f})")
        best_acc = best_epochs_acc
        best_config = best_config.replace("training.epochs=50", f"training.epochs={best_epochs}")
        best_name = f"{best_name}+Epochs={best_epochs}"
    else:
        logger.info(f"✗ No improvement. Best epochs={best_epochs} got {best_epochs_acc:.4f} vs current {best_acc:.4f}")

# FINAL RESULTS
logger.info("\n" + "="*70)
logger.info("FINAL OPTIMIZED CONFIGURATION")
logger.info("="*70)
logger.info(f"Best Accuracy: {best_acc:.4f}")
logger.info(f"Best Config: {best_config}")
logger.info(f"Description: {best_name}")

# Compare to original baseline
logger.info("\n" + "-"*40)
if best_acc > 0.720:
    improvement = (best_acc - 0.720) * 100
    logger.info(f"✓ IMPROVEMENT: +{improvement:.1f}% over baseline (72.0%)")
else:
    logger.info("✗ No improvement over baseline (72.0%)")

logger.info("="*70)

# Save results
results = {
    'best_accuracy': best_acc,
    'best_config': best_config,
    'best_name': best_name,
    'baseline_accuracy': 0.720
}

with open('iterative_search_results.json', 'w') as f:
    json.dump(results, f, indent=2)