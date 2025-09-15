#!/usr/bin/env python3
"""
Final comprehensive CLIP model comparison with full training
"""

import os
import shutil
import subprocess
import json
import time
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_clip_model(model_name, embedding_file, description):
    """Test model with full training"""

    logger.info(f"\n{'='*70}")
    logger.info(f"Testing {model_name}: {description}")
    logger.info(f"Embedding file: {embedding_file}")
    logger.info('='*70)

    if not os.path.exists(embedding_file):
        logger.error(f"Embedding file not found: {embedding_file}")
        return None

    # Backup current cache
    main_cache = "artifacts/cache/clip_embeddings_cache.pkl"
    backup_file = f"{main_cache}.backup_{int(time.time())}"

    try:
        if os.path.exists(main_cache):
            shutil.copy2(main_cache, backup_file)

        # Use this model's embeddings
        shutil.copy2(embedding_file, main_cache)

        # Run full training with optimal config
        cmd = "python train.py --config-name=unified_optimal_final"

        logger.info("Running full training (200 epochs)...")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=900)

        if result.returncode != 0:
            logger.error(f"Training failed: {result.stderr}")
            return None

        # Parse results from output
        output_lines = result.stdout.split('\n')
        metrics = {
            'model': model_name,
            'description': description,
            'attractive_acc': None,
            'smart_acc': None,
            'trustworthy_acc': None,
            'average_acc': None,
            'attractive_loss': None,
            'smart_loss': None,
            'trustworthy_loss': None,
            'average_loss': None
        }

        # Parse the training results section
        for line in output_lines:
            if 'ATTRACTIVE:' in line:
                # Look for accuracy in next few lines
                idx = output_lines.index(line)
                for i in range(idx, min(idx+5, len(output_lines))):
                    if 'Accuracy:' in output_lines[i]:
                        try:
                            metrics['attractive_acc'] = float(output_lines[i].split(':')[1].strip())
                        except:
                            pass
                    elif 'CE Loss:' in output_lines[i]:
                        try:
                            metrics['attractive_loss'] = float(output_lines[i].split(':')[1].strip())
                        except:
                            pass
            elif 'SMART:' in line:
                idx = output_lines.index(line)
                for i in range(idx, min(idx+5, len(output_lines))):
                    if 'Accuracy:' in output_lines[i]:
                        try:
                            metrics['smart_acc'] = float(output_lines[i].split(':')[1].strip())
                        except:
                            pass
                    elif 'CE Loss:' in output_lines[i]:
                        try:
                            metrics['smart_loss'] = float(output_lines[i].split(':')[1].strip())
                        except:
                            pass
            elif 'TRUSTWORTHY:' in line:
                idx = output_lines.index(line)
                for i in range(idx, min(idx+5, len(output_lines))):
                    if 'Accuracy:' in output_lines[i]:
                        try:
                            metrics['trustworthy_acc'] = float(output_lines[i].split(':')[1].strip())
                        except:
                            pass
                    elif 'CE Loss:' in output_lines[i]:
                        try:
                            metrics['trustworthy_loss'] = float(output_lines[i].split(':')[1].strip())
                        except:
                            pass
            elif 'Average Accuracy:' in line:
                try:
                    metrics['average_acc'] = float(line.split(':')[1].strip())
                except:
                    pass
            elif 'Average CE Loss:' in line:
                try:
                    metrics['average_loss'] = float(line.split(':')[1].strip())
                except:
                    pass

        if metrics['average_acc'] is not None:
            logger.info(f"✓ Results:")
            logger.info(f"  Average Accuracy: {metrics['average_acc']:.4f}")
            logger.info(f"  Average Loss: {metrics['average_loss']:.4f}")
            logger.info(f"  Attractive: {metrics['attractive_acc']:.4f} acc, {metrics['attractive_loss']:.4f} loss")
            logger.info(f"  Smart: {metrics['smart_acc']:.4f} acc, {metrics['smart_loss']:.4f} loss")
            logger.info(f"  Trustworthy: {metrics['trustworthy_acc']:.4f} acc, {metrics['trustworthy_loss']:.4f} loss")
            return metrics
        else:
            logger.error("Failed to parse results")
            return None

    except subprocess.TimeoutExpired:
        logger.error("Training timed out after 15 minutes")
        return None
    except Exception as e:
        logger.error(f"Error: {e}")
        return None
    finally:
        # Restore original cache
        if os.path.exists(backup_file):
            shutil.copy2(backup_file, main_cache)
            os.remove(backup_file)

def main():
    """Run comprehensive CLIP model comparison"""

    logger.info("=" * 80)
    logger.info("FINAL CLIP MODEL COMPARISON - FULL TRAINING")
    logger.info("=" * 80)
    logger.info("Configuration: ReLU, LR=0.003, Dropout=0.1, WD=0.0, 200 epochs")
    logger.info("")

    # Models to test
    models_to_test = [
        ("ViT-B/32", "artifacts/cache/clip_embeddings_cache_backup.pkl", "Baseline (512D)"),
        ("ViT-B/16", "artifacts/cache/clip_embeddings_ViT-B_16_fixed.pkl", "Higher resolution (512D)"),
        ("ViT-L/14", "artifacts/cache/clip_embeddings_ViT-L_14_fixed.pkl", "Larger model (768D)"),
    ]

    results = []
    start_time = time.time()

    for model_name, embedding_file, description in models_to_test:
        result = test_clip_model(model_name, embedding_file, description)
        if result:
            results.append(result)
        time.sleep(5)  # Brief pause between tests

    # Generate comprehensive report
    if results:
        logger.info("")
        logger.info("=" * 80)
        logger.info("FINAL COMPARISON RESULTS")
        logger.info("=" * 80)

        # Sort by average accuracy
        results.sort(key=lambda x: x['average_acc'], reverse=True)

        # Header
        logger.info(f"\n{'Rank':<4} {'Model':<12} {'Avg Acc':<10} {'Attractive':<12} {'Smart':<12} {'Trustworthy':<12}")
        logger.info("-" * 80)

        # Results table
        for i, r in enumerate(results, 1):
            logger.info(f"{i:<4} {r['model']:<12} {r['average_acc']:<10.4f} "
                       f"{r['attractive_acc']:<12.4f} {r['smart_acc']:<12.4f} {r['trustworthy_acc']:<12.4f}")

        # Analysis
        best = results[0]
        baseline = next((r for r in results if r['model'] == 'ViT-B/32'), None)

        logger.info("")
        logger.info("DETAILED ANALYSIS:")
        logger.info(f"Best performer: {best['model']} with {best['average_acc']:.4f} average accuracy")

        if baseline:
            if best['model'] != 'ViT-B/32':
                improvement = (best['average_acc'] - baseline['average_acc']) * 100
                logger.info(f"Improvement over baseline: +{improvement:.2f}%")
                logger.info(f"  {baseline['average_acc']:.4f} → {best['average_acc']:.4f}")

            # Per-attribute comparison
            logger.info("")
            logger.info("Per-attribute improvements over ViT-B/32:")
            for result in results:
                if result['model'] != 'ViT-B/32':
                    attr_impr = [
                        (result['attractive_acc'] - baseline['attractive_acc']) * 100,
                        (result['smart_acc'] - baseline['smart_acc']) * 100,
                        (result['trustworthy_acc'] - baseline['trustworthy_acc']) * 100
                    ]
                    logger.info(f"  {result['model']}: Attractive +{attr_impr[0]:.2f}%, "
                               f"Smart +{attr_impr[1]:.2f}%, Trustworthy +{attr_impr[2]:.2f}%")

        # Performance ranking
        logger.info("")
        logger.info("Performance ranking:")
        for i, r in enumerate(results, 1):
            logger.info(f"  {i}. {r['model']} ({r['description']}): {r['average_acc']:.4f}")

        # Save detailed results
        output_file = 'final_clip_comparison_results.json'
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_time_minutes': (time.time() - start_time) / 60,
                'configuration': 'ReLU, LR=0.003, Dropout=0.1, WD=0.0, 200 epochs',
                'results': results,
                'best_model': best,
                'baseline_model': baseline,
                'summary': {
                    'best_performer': best['model'],
                    'best_accuracy': best['average_acc'],
                    'baseline_accuracy': baseline['average_acc'] if baseline else None,
                    'improvement_percent': ((best['average_acc'] - baseline['average_acc']) * 100) if baseline and best['model'] != 'ViT-B/32' else 0
                }
            }, f, indent=2)

        logger.info(f"\nDetailed results saved to {output_file}")

        # Final recommendation
        if baseline and best['model'] != 'ViT-B/32':
            improvement = (best['average_acc'] - baseline['average_acc']) * 100
            if improvement > 2.0:
                logger.info(f"\n✓ STRONG RECOMMENDATION: Upgrade to {best['model']} (+{improvement:.2f}% accuracy)")
            elif improvement > 0.5:
                logger.info(f"\n~ MODERATE RECOMMENDATION: Consider {best['model']} (+{improvement:.2f}% accuracy)")
            else:
                logger.info(f"\n✗ RECOMMENDATION: Stay with ViT-B/32 (minimal gain: +{improvement:.2f}%)")
        else:
            logger.info(f"\n= RECOMMENDATION: ViT-B/32 remains the best choice")

        total_time = (time.time() - start_time) / 60
        logger.info(f"\nTotal comparison time: {total_time:.1f} minutes")

    else:
        logger.error("No successful results obtained")

    logger.info("\n" + "=" * 80)

if __name__ == "__main__":
    main()