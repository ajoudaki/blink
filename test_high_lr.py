#!/usr/bin/env python3
"""
Test very high learning rates with stability measures.
"""

import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_config(lr, epochs=30):
    """Run config with high LR and stability measures."""
    # Best config so far
    base = "--config-name=unified_best task_type=comparison model.hidden_dims=[384,256,128,64,32] model.dropout=0.15"

    # Add LR and stability measures
    config = f"{base} training.epochs={epochs} +optimizer.lr={lr} +optimizer.weight_decay=0.001"

    # Add gradient clipping and scheduler for stability
    if lr >= 0.01:
        config += " +optimizer.grad_clip=1.0"
    if lr >= 0.02:
        config += " +optimizer.grad_clip=0.5"  # More aggressive clipping

    try:
        cmd = f"python train.py {config}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)

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
            logger.info(f"  LR={lr}: Acc={avg_acc:.4f}, Loss={avg_loss:.3f}")
            return avg_acc, avg_loss
        else:
            # Check for training failure
            if 'nan' in result.stdout.lower() or 'inf' in result.stdout.lower():
                logger.info(f"  LR={lr}: Training unstable (NaN/Inf)")
            else:
                logger.info(f"  LR={lr}: Failed to parse results")
            return None, None

    except subprocess.TimeoutExpired:
        logger.info(f"  LR={lr}: Timeout")
        return None, None
    except Exception as e:
        logger.info(f"  LR={lr}: Error - {e}")
        return None, None


logger.info("="*70)
logger.info("TESTING VERY HIGH LEARNING RATES")
logger.info("="*70)
logger.info("Current best: 73.4% (with LR=0.001, dropout=0.15, WD=0.001)")
logger.info("-"*40)

# Test progressively higher learning rates
high_lrs = [0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
results = []

for lr in high_lrs:
    # Use fewer epochs for very high LR to avoid overfitting
    epochs = 30 if lr <= 0.01 else 20 if lr <= 0.05 else 10

    logger.info(f"\nTesting LR={lr} with {epochs} epochs...")
    acc, loss = run_config(lr, epochs)

    if acc:
        results.append((lr, acc, loss, epochs))
        if acc >= 0.734:
            logger.info(f"  ✓ Good result! Matches or beats current best.")
    else:
        logger.info(f"  ✗ Training failed or unstable")
        # Stop if training becomes unstable
        if lr >= 0.02:
            logger.info("  Stopping - training becoming unstable at high LRs")
            break

# Report results
if results:
    logger.info("\n" + "="*70)
    logger.info("RESULTS SUMMARY")
    logger.info("-"*40)

    results.sort(key=lambda x: x[1], reverse=True)

    for lr, acc, loss, epochs in results:
        status = "✓ BEST" if acc >= 0.734 else ""
        logger.info(f"LR={lr:6.3f}: Acc={acc:.4f}, Loss={loss:.3f} ({epochs} epochs) {status}")

    best_lr, best_acc, best_loss, best_epochs = results[0]

    logger.info("\n" + "="*70)
    if best_acc > 0.734:
        logger.info(f"✓ NEW BEST: LR={best_lr} achieved {best_acc:.4f}")
        logger.info(f"Improvement: +{(best_acc - 0.734)*100:.1f}%")
        logger.info(f"Final config:")
        logger.info(f"  --config-name=unified_best task_type=comparison")
        logger.info(f"  model.hidden_dims=[384,256,128,64,32]")
        logger.info(f"  model.dropout=0.15")
        logger.info(f"  training.epochs={best_epochs}")
        logger.info(f"  +optimizer.lr={best_lr}")
        logger.info(f"  +optimizer.weight_decay=0.001")
        if best_lr >= 0.01:
            logger.info(f"  +optimizer.grad_clip=1.0")
    else:
        logger.info(f"✗ No improvement. Best was LR={best_lr} with {best_acc:.4f}")
        logger.info(f"Current best remains: 73.4% (LR=0.001)")

    logger.info("="*70)