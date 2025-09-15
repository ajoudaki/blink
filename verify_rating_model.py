#!/usr/bin/env python3
"""
Verify rating task performance with best architecture.
Run multiple times and collect all metrics.
"""

import subprocess
import logging
import json
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_rating_model(run_number):
    """Run rating model and extract all metrics."""

    cmd = "python train.py --config-name=unified_best_rating"

    try:
        logger.info(f"  Starting run {run_number}...")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)

        # Parse output for all metrics
        output_lines = result.stdout.split('\n')

        metrics = {
            'attractive': {},
            'smart': {},
            'trustworthy': {},
            'average': {}
        }

        current_target = None

        for line in output_lines:
            # Identify target sections
            if 'ATTRACTIVE:' in line:
                current_target = 'attractive'
            elif 'SMART:' in line:
                current_target = 'smart'
            elif 'TRUSTWORTHY:' in line:
                current_target = 'trustworthy'

            # Extract metrics for current target
            if current_target and current_target != 'average':
                if 'Accuracy:' in line:
                    metrics[current_target]['accuracy'] = float(line.split(':')[1].strip())
                elif 'CE Loss:' in line:
                    metrics[current_target]['ce_loss'] = float(line.split(':')[1].strip())
                elif 'MAE:' in line:
                    metrics[current_target]['mae'] = float(line.split(':')[1].strip())
                elif 'RMSE:' in line:
                    metrics[current_target]['rmse'] = float(line.split(':')[1].strip())

            # Extract average metrics
            if 'Average Accuracy:' in line:
                metrics['average']['accuracy'] = float(line.split(':')[1].strip())
            elif 'Average CE Loss:' in line:
                metrics['average']['ce_loss'] = float(line.split(':')[1].strip())
            elif 'Average MAE:' in line:
                metrics['average']['mae'] = float(line.split(':')[1].strip())
            elif 'Average RMSE:' in line:
                metrics['average']['rmse'] = float(line.split(':')[1].strip())

        # Verify we got all metrics
        if metrics['average'].get('accuracy'):
            return metrics
        else:
            return None

    except subprocess.TimeoutExpired:
        logger.error(f"  Run {run_number} timed out")
        return None
    except Exception as e:
        logger.error(f"  Run {run_number} failed: {e}")
        return None


def main():
    """Run verification for rating task."""

    logger.info("="*70)
    logger.info("RATING TASK - MODEL VERIFICATION")
    logger.info("="*70)
    logger.info("Config: unified_best_rating.yaml")
    logger.info("Architecture: [384, 256, 128, 64, 32] (5-layer deep narrow)")
    logger.info("Dropout: 0.15, Weight Decay: 0.001, LR: 0.001, Epochs: 50")
    logger.info("-"*70)

    # Run model multiple times
    n_runs = 3
    all_results = []

    for i in range(1, n_runs + 1):
        logger.info(f"\nRun {i}/{n_runs}:")
        metrics = run_rating_model(i)

        if metrics:
            all_results.append(metrics)
            logger.info(f"  ✓ Average Metrics:")
            logger.info(f"    Accuracy:  {metrics['average']['accuracy']:.3f}")
            logger.info(f"    CE Loss:   {metrics['average']['ce_loss']:.3f}")
            logger.info(f"    MAE:       {metrics['average']['mae']:.3f}")
            logger.info(f"    RMSE:      {metrics['average']['rmse']:.3f}")
        else:
            logger.error(f"  ✗ Failed to get results")

    if all_results:
        logger.info("\n" + "="*70)
        logger.info("VERIFICATION RESULTS - RATING TASK")
        logger.info("="*70)

        # Calculate statistics for each metric
        metrics_stats = {}

        for metric_name in ['accuracy', 'ce_loss', 'mae', 'rmse']:
            values = [r['average'][metric_name] for r in all_results]
            metrics_stats[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }

        logger.info(f"\n{len(all_results)} successful runs completed")
        logger.info("\nAverage Metrics Across Runs:")
        logger.info(f"  Accuracy:  {metrics_stats['accuracy']['mean']:.3f} ± {metrics_stats['accuracy']['std']:.3f}")
        logger.info(f"  CE Loss:   {metrics_stats['ce_loss']['mean']:.3f} ± {metrics_stats['ce_loss']['std']:.3f}")
        logger.info(f"  MAE:       {metrics_stats['mae']['mean']:.3f} ± {metrics_stats['mae']['std']:.3f}")
        logger.info(f"  RMSE:      {metrics_stats['rmse']['mean']:.3f} ± {metrics_stats['rmse']['std']:.3f}")

        # Per-target statistics
        logger.info("\nPer-Target Accuracy (mean ± std):")
        for target in ['attractive', 'smart', 'trustworthy']:
            target_accs = [r[target]['accuracy'] for r in all_results]
            target_maes = [r[target]['mae'] for r in all_results]
            logger.info(f"  {target.capitalize():12s}: Acc={np.mean(target_accs):.3f}±{np.std(target_accs):.3f}, MAE={np.mean(target_maes):.3f}±{np.std(target_maes):.3f}")

        # Compare to comparison task
        logger.info("\n" + "-"*70)
        logger.info("COMPARISON TO COMPARISON TASK:")
        logger.info(f"  Comparison Task Accuracy: 72.5%")
        logger.info(f"  Rating Task Accuracy:     {metrics_stats['accuracy']['mean']*100:.1f}%")

        diff = metrics_stats['accuracy']['mean'] - 0.565  # Baseline rating accuracy
        if diff > 0:
            logger.info(f"  Improvement over baseline: +{diff*100:.1f}%")

        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"rating_model_verification_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump({
                'config': 'unified_best_rating.yaml',
                'architecture': [384, 256, 128, 64, 32],
                'runs': all_results,
                'summary': {
                    'accuracy': metrics_stats['accuracy'],
                    'ce_loss': metrics_stats['ce_loss'],
                    'mae': metrics_stats['mae'],
                    'rmse': metrics_stats['rmse']
                }
            }, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

        logger.info(f"\n✓ Results saved to {results_file}")
        logger.info("="*70)

    else:
        logger.error("\n✗ No successful runs completed")


if __name__ == "__main__":
    main()