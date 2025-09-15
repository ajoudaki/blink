#!/usr/bin/env python3
"""
Test weight decay effect across multiple seeds for all CLIP models
"""

import os
import shutil
import subprocess
import json
import numpy as np
import time
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_with_seed(model_name, embedding_file, seed, wd=0.01):
    """Test a model with specific seed and weight decay"""

    # Backup current cache
    main_cache = "artifacts/cache/clip_embeddings_cache.pkl"
    backup_file = f"{main_cache}.backup_{int(time.time())}"

    try:
        if os.path.exists(main_cache):
            shutil.copy2(main_cache, backup_file)

        # Use this model's embeddings
        shutil.copy2(embedding_file, main_cache)

        # Run training with specific seed and weight decay
        cmd = f"python train_with_early_stopping.py --config-name=unified_optimal_with_wd seed={seed} training.weight_decay={wd}"

        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            logger.error(f"Training failed for {model_name} seed {seed}")
            return None

        # Parse results
        output_lines = result.stdout.split('\n')

        # Look for test results
        test_acc = None
        val_acc = None
        attractive_test = None
        smart_test = None
        trustworthy_test = None

        in_test_section = False
        in_val_section = False

        for i, line in enumerate(output_lines):
            if 'TEST SET RESULTS:' in line:
                in_test_section = True
                in_val_section = False
            elif 'VALIDATION SET RESULTS:' in line:
                in_val_section = True
                in_test_section = False
            elif 'Average Test Accuracy:' in line:
                try:
                    test_acc = float(line.split(':')[1].strip())
                except:
                    pass
            elif 'Average Val Accuracy:' in line:
                try:
                    val_acc = float(line.split(':')[1].strip())
                except:
                    pass
            elif in_test_section and 'ATTRACTIVE:' in line:
                # Look for accuracy in next few lines
                for j in range(i, min(i+3, len(output_lines))):
                    if 'Accuracy:' in output_lines[j]:
                        try:
                            attractive_test = float(output_lines[j].split(':')[1].strip())
                            break
                        except:
                            pass
            elif in_test_section and 'SMART:' in line:
                for j in range(i, min(i+3, len(output_lines))):
                    if 'Accuracy:' in output_lines[j]:
                        try:
                            smart_test = float(output_lines[j].split(':')[1].strip())
                            break
                        except:
                            pass
            elif in_test_section and 'TRUSTWORTHY:' in line:
                for j in range(i, min(i+3, len(output_lines))):
                    if 'Accuracy:' in output_lines[j]:
                        try:
                            trustworthy_test = float(output_lines[j].split(':')[1].strip())
                            break
                        except:
                            pass

        if test_acc is not None:
            return {
                'model': model_name,
                'seed': seed,
                'val_acc': val_acc,
                'test_acc': test_acc,
                'attractive_test': attractive_test,
                'smart_test': smart_test,
                'trustworthy_test': trustworthy_test
            }
        else:
            logger.error(f"Failed to parse results for {model_name} seed {seed}")
            return None

    except subprocess.TimeoutExpired:
        logger.error(f"Training timed out for {model_name} seed {seed}")
        return None
    except Exception as e:
        logger.error(f"Error for {model_name} seed {seed}: {e}")
        return None
    finally:
        # Restore original cache
        if os.path.exists(backup_file):
            shutil.copy2(backup_file, main_cache)
            os.remove(backup_file)

def compute_stats(results):
    """Compute mean and std for a list of results"""
    if not results:
        return None

    val_accs = [r['val_acc'] for r in results if r['val_acc'] is not None]
    test_accs = [r['test_acc'] for r in results if r['test_acc'] is not None]
    attractive = [r['attractive_test'] for r in results if r['attractive_test'] is not None]
    smart = [r['smart_test'] for r in results if r['smart_test'] is not None]
    trustworthy = [r['trustworthy_test'] for r in results if r['trustworthy_test'] is not None]

    return {
        'val_mean': np.mean(val_accs) if val_accs else None,
        'val_std': np.std(val_accs) if val_accs else None,
        'test_mean': np.mean(test_accs) if test_accs else None,
        'test_std': np.std(test_accs) if test_accs else None,
        'attractive_mean': np.mean(attractive) if attractive else None,
        'attractive_std': np.std(attractive) if attractive else None,
        'smart_mean': np.mean(smart) if smart else None,
        'smart_std': np.std(smart) if smart else None,
        'trustworthy_mean': np.mean(trustworthy) if trustworthy else None,
        'trustworthy_std': np.std(trustworthy) if trustworthy else None,
        'n_samples': len(test_accs)
    }

def main():
    """Test all models with multiple seeds"""

    logger.info("=" * 80)
    logger.info("WEIGHT DECAY ANALYSIS WITH MULTIPLE SEEDS")
    logger.info("Configuration: WD=0.01, Seeds=[42, 123, 456]")
    logger.info("=" * 80)

    # Models to test
    models = [
        ("ViT-B/32", "artifacts/cache/clip_embeddings_cache_backup.pkl", "Baseline (512D)"),
        ("ViT-B/16", "artifacts/cache/clip_embeddings_ViT-B_16_fixed.pkl", "Higher res (512D)"),
        ("ViT-L/14", "artifacts/cache/clip_embeddings_ViT-L_14_fixed.pkl", "Larger (768D)"),
    ]

    # Seeds to test
    seeds = [42, 123, 456]

    # Store all results
    all_results = {}

    for model_name, embedding_file, description in models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {model_name}: {description}")
        logger.info('='*60)

        if not os.path.exists(embedding_file):
            logger.error(f"Embedding file not found: {embedding_file}")
            continue

        model_results = []

        for seed in seeds:
            logger.info(f"  Testing seed {seed}...")
            result = test_model_with_seed(model_name, embedding_file, seed, wd=0.01)

            if result:
                model_results.append(result)
                logger.info(f"    ✓ Val: {result['val_acc']:.4f}, Test: {result['test_acc']:.4f}")
            else:
                logger.error(f"    ✗ Failed")

            time.sleep(2)  # Brief pause between runs

        # Compute statistics
        stats = compute_stats(model_results)
        if stats:
            all_results[model_name] = {
                'raw_results': model_results,
                'stats': stats,
                'description': description
            }

            logger.info(f"\n  Summary for {model_name}:")
            logger.info(f"    Validation: {stats['val_mean']:.4f} ± {stats['val_std']:.4f}")
            logger.info(f"    Test: {stats['test_mean']:.4f} ± {stats['test_std']:.4f}")

    # Generate final comparison table
    if all_results:
        logger.info("\n" + "=" * 80)
        logger.info("FINAL RESULTS WITH WEIGHT DECAY = 0.01")
        logger.info("=" * 80)

        # Main performance table
        logger.info("\n📊 Overall Performance (Test Accuracy)")
        logger.info("-" * 60)
        logger.info(f"{'Model':<12} {'Mean ± Std':<15} {'Min':<8} {'Max':<8} {'N':<5}")
        logger.info("-" * 60)

        for model_name in ["ViT-B/32", "ViT-B/16", "ViT-L/14"]:
            if model_name in all_results:
                stats = all_results[model_name]['stats']
                raw = all_results[model_name]['raw_results']
                test_accs = [r['test_acc'] for r in raw if r['test_acc'] is not None]

                mean_std = f"{stats['test_mean']:.3f} ± {stats['test_std']:.3f}"
                min_val = f"{min(test_accs):.3f}" if test_accs else "N/A"
                max_val = f"{max(test_accs):.3f}" if test_accs else "N/A"

                logger.info(f"{model_name:<12} {mean_std:<15} {min_val:<8} {max_val:<8} {stats['n_samples']:<5}")

        # Per-attribute table
        logger.info("\n📊 Per-Attribute Test Performance")
        logger.info("-" * 80)
        logger.info(f"{'Model':<12} {'Attractive':<20} {'Smart':<20} {'Trustworthy':<20}")
        logger.info("-" * 80)

        for model_name in ["ViT-B/32", "ViT-B/16", "ViT-L/14"]:
            if model_name in all_results:
                stats = all_results[model_name]['stats']

                attr_str = f"{stats['attractive_mean']:.3f} ± {stats['attractive_std']:.3f}"
                smart_str = f"{stats['smart_mean']:.3f} ± {stats['smart_std']:.3f}"
                trust_str = f"{stats['trustworthy_mean']:.3f} ± {stats['trustworthy_std']:.3f}"

                logger.info(f"{model_name:<12} {attr_str:<20} {smart_str:<20} {trust_str:<20}")

        # Ranking
        logger.info("\n🏆 Ranking by Mean Test Accuracy")
        logger.info("-" * 40)

        ranked = sorted(
            [(name, data['stats']['test_mean']) for name, data in all_results.items()],
            key=lambda x: x[1],
            reverse=True
        )

        for i, (model_name, mean_acc) in enumerate(ranked, 1):
            stats = all_results[model_name]['stats']
            logger.info(f"{i}. {model_name}: {mean_acc:.3f} ± {stats['test_std']:.3f}")

        # Save results
        output_file = 'weight_decay_multiseed_results.json'
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'config': 'WD=0.01, Seeds=[42, 123, 456]',
                'results': all_results
            }, f, indent=2)

        logger.info(f"\n💾 Results saved to {output_file}")

        # Statistical significance note
        logger.info("\n📈 Statistical Analysis:")
        if len(all_results) >= 2:
            # Find best and second best
            best = ranked[0]
            second = ranked[1] if len(ranked) > 1 else None

            if second:
                best_stats = all_results[best[0]]['stats']
                second_stats = all_results[second[0]]['stats']

                # Simple overlap check (not a proper statistical test)
                best_lower = best_stats['test_mean'] - best_stats['test_std']
                second_upper = second_stats['test_mean'] + second_stats['test_std']

                if best_lower > second_upper:
                    logger.info(f"✓ {best[0]} is significantly better (non-overlapping std)")
                else:
                    logger.info(f"⚠ {best[0]} and {second[0]} have overlapping performance ranges")

    logger.info("\n" + "=" * 80)

if __name__ == "__main__":
    main()