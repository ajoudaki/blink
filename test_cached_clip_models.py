#!/usr/bin/env python3
"""
Test best configuration with pre-cached CLIP model embeddings.
"""

import os
import shutil
import pickle
import subprocess
import logging
import json
import time
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_clip_model_embeddings(model_name, embedding_file, description):
    """Test model with specific CLIP embeddings."""

    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {model_name}: {description}")
    logger.info(f"Embedding file: {embedding_file}")
    logger.info('='*60)

    # Check if embedding file exists
    if not os.path.exists(embedding_file):
        logger.error(f"Embedding file not found: {embedding_file}")
        return None

    # Check embedding info
    with open(embedding_file, 'rb') as f:
        embeddings = pickle.load(f)

    num_embeddings = len(embeddings)
    if embeddings:
        sample_embedding = next(iter(embeddings.values()))
        embedding_dim = sample_embedding.shape[0]
    else:
        embedding_dim = "unknown"

    logger.info(f"Found {num_embeddings} embeddings with dimension {embedding_dim}")

    # Backup current cache and use this model's embeddings
    main_cache = "artifacts/cache/clip_embeddings_cache.pkl"
    backup_file = f"{main_cache}.backup_{int(time.time())}"

    try:
        # Backup current cache if it exists
        if os.path.exists(main_cache):
            shutil.copy2(main_cache, backup_file)

        # Copy this model's embeddings to main cache
        shutil.copy2(embedding_file, main_cache)

        # Run training with our best config
        cmd = "python train_with_early_stopping.py --config-name=unified_optimal_final"

        logger.info("Running training...")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            logger.error(f"Training failed with return code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return None

        # Parse results
        output_lines = result.stdout.split('\n')
        metrics = {
            'model': model_name,
            'description': description,
            'embedding_dim': embedding_dim,
            'num_embeddings': num_embeddings,
            'val_acc': None,
            'test_acc': None,
            'val_loss': None,
            'test_loss': None,
            'best_epoch': None
        }

        # Parse output for metrics
        for i, line in enumerate(output_lines):
            if 'Average Val Accuracy:' in line:
                try:
                    metrics['val_acc'] = float(line.split(':')[1].strip())
                except:
                    pass
            elif 'Average Test Accuracy:' in line:
                try:
                    metrics['test_acc'] = float(line.split(':')[1].strip())
                except:
                    pass
            elif 'Average Val CE Loss:' in line:
                try:
                    metrics['val_loss'] = float(line.split(':')[1].strip())
                except:
                    pass
            elif 'Average Test CE Loss:' in line:
                try:
                    metrics['test_loss'] = float(line.split(':')[1].strip())
                except:
                    pass
            elif 'Best Epoch:' in line:
                try:
                    metrics['best_epoch'] = int(line.split(':')[1].strip())
                except:
                    pass

        if metrics['val_acc'] and metrics['test_acc']:
            logger.info(f"✓ Results:")
            logger.info(f"  Validation: {metrics['val_acc']:.4f} accuracy, {metrics['val_loss']:.4f} loss")
            logger.info(f"  Test: {metrics['test_acc']:.4f} accuracy, {metrics['test_loss']:.4f} loss")
            logger.info(f"  Best epoch: {metrics['best_epoch']}")
            return metrics
        else:
            logger.error("Failed to parse validation/test metrics")
            return None

    except subprocess.TimeoutExpired:
        logger.error("Training timed out after 10 minutes")
        return None
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return None
    finally:
        # Restore original cache
        if os.path.exists(backup_file):
            shutil.copy2(backup_file, main_cache)
            os.remove(backup_file)


def main():
    """Test all cached CLIP model embeddings."""

    logger.info("="*70)
    logger.info("CLIP MODEL EMBEDDING COMPARISON")
    logger.info("="*70)
    logger.info("Testing best configuration with different CLIP embeddings")
    logger.info("Config: ReLU, LR=0.003, Dropout=0.1, WD=0.0, Early Stopping")
    logger.info("")

    # Define models to test
    models_to_test = [
        ("ViT-B/32", "artifacts/cache/clip_embeddings_ViT-B_32.pkl", "Current baseline (512D)"),
        ("ViT-B/16", "artifacts/cache/clip_embeddings_ViT-B_16.pkl", "Higher resolution patches (512D)"),
        ("ViT-L/14", "artifacts/cache/clip_embeddings_ViT-L_14.pkl", "Larger model (768D)"),
    ]

    results = []

    for model_name, embedding_file, description in models_to_test:
        result = test_clip_model_embeddings(model_name, embedding_file, description)
        if result:
            results.append(result)
        time.sleep(5)  # Brief pause between tests

    # Report comparison
    if len(results) > 0:
        logger.info("")
        logger.info("="*70)
        logger.info("COMPARISON RESULTS")
        logger.info("="*70)

        # Sort by test accuracy
        results.sort(key=lambda x: x['test_acc'], reverse=True)

        logger.info(f"\n{'Rank':<4} {'Model':<12} {'Dim':<6} {'Val Acc':<10} {'Test Acc':<10} {'Val Loss':<10} {'Epoch':<8}")
        logger.info("-"*70)

        for i, r in enumerate(results, 1):
            logger.info(f"{i:<4} {r['model']:<12} {r['embedding_dim']:<6} "
                       f"{r['val_acc']:<10.4f} {r['test_acc']:<10.4f} "
                       f"{r['val_loss']:<10.4f} {r['best_epoch']:<8}")

        # Analysis
        best = results[0]
        baseline = next((r for r in results if r['model'] == 'ViT-B/32'), None)

        logger.info("")
        logger.info("ANALYSIS:")
        logger.info(f"Best model: {best['model']} with {best['test_acc']:.4f} test accuracy")

        if baseline and best['model'] != 'ViT-B/32':
            val_improvement = (best['val_acc'] - baseline['val_acc']) * 100
            test_improvement = (best['test_acc'] - baseline['test_acc']) * 100
            logger.info(f"Improvement over baseline:")
            logger.info(f"  Validation: +{val_improvement:.2f}% ({baseline['val_acc']:.4f} → {best['val_acc']:.4f})")
            logger.info(f"  Test: +{test_improvement:.2f}% ({baseline['test_acc']:.4f} → {best['test_acc']:.4f})")

        # Find best per metric
        best_val = max(results, key=lambda x: x['val_acc'])
        best_test = max(results, key=lambda x: x['test_acc'])

        if best_val['model'] != best_test['model']:
            logger.info(f"\nNote: Best validation ({best_val['model']}) differs from best test ({best_test['model']})")

        # Save results
        output_file = 'clip_models_final_comparison.json'
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'config': 'ReLU, LR=0.003, Dropout=0.1, WD=0.0',
                'results': results,
                'best_model': best,
                'baseline_model': baseline
            }, f, indent=2)

        logger.info(f"\nResults saved to {output_file}")

        # Determine if CLIP upgrade is worth it
        if baseline and best['model'] != 'ViT-B/32':
            test_gain = (best['test_acc'] - baseline['test_acc']) * 100
            if test_gain > 1.0:
                logger.info(f"\n✓ Recommendation: Upgrade to {best['model']} (+{test_gain:.2f}% test accuracy)")
            elif test_gain > 0.3:
                logger.info(f"\n~ Recommendation: Consider {best['model']} (+{test_gain:.2f}% test accuracy)")
            else:
                logger.info(f"\n✗ Recommendation: Stay with ViT-B/32 (minimal gain: +{test_gain:.2f}%)")

    else:
        logger.error("No successful results obtained")

    logger.info("\n" + "="*70)


if __name__ == "__main__":
    main()