#!/usr/bin/env python3
"""
Simple test of CLIP models using the working train.py
"""

import os
import shutil
import pickle
import subprocess
import logging
import json
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_clip_model(model_name, embedding_file, description):
    """Test with specific CLIP embeddings using regular train.py"""

    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {model_name}: {description}")
    logger.info('='*60)

    # Check embedding info
    with open(embedding_file, 'rb') as f:
        embeddings = pickle.load(f)

    embedding_dim = next(iter(embeddings.values())).shape[0]
    logger.info(f"Embeddings: {len(embeddings)} items, {embedding_dim}D")

    # Use regular train.py with optimal config but shorter training
    # Create a temporary config file
    temp_config = f"""task_type: comparison

model:
  hidden_dims: [384, 256, 128, 64, 32]
  activation: relu
  dropout: 0.1
  batch_norm: false
  layer_norm: false
  use_user_encoding: false
  use_batchnorm: true
  use_layernorm: false
  use_user_embedding: true
  user_embedding_dim: 32
  use_glu: false

data:
  targets: ["attractive", "smart", "trustworthy"]
  embeddings_cache_file: "clip_embeddings_cache.pkl"
  force_recompute_embeddings: false

training:
  epochs: 50
  learning_rate: 0.003
  validation_split: 0.2
  augment_swapped_pairs: true
  optimizer: adamw
  weight_decay: 0.0
  batch_size: 128

gpu:
  device_id: 1

seed: 42"""

    # Write temp config
    with open('configs/temp_test.yaml', 'w') as f:
        f.write(temp_config)

    # Backup current cache and use this model's embeddings
    main_cache = "artifacts/cache/clip_embeddings_cache.pkl"
    backup_file = f"{main_cache}.backup_temp"

    try:
        if os.path.exists(main_cache):
            shutil.copy2(main_cache, backup_file)

        shutil.copy2(embedding_file, main_cache)

        # Run training
        cmd = "python train.py --config-name=temp_test"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            logger.error(f"Training failed: {result.stderr}")
            return None

        # Parse results
        output_lines = result.stdout.split('\n')
        avg_acc = None

        for line in output_lines:
            if 'Average Accuracy:' in line:
                try:
                    avg_acc = float(line.split(':')[1].strip())
                    break
                except:
                    pass

        if avg_acc:
            logger.info(f"✓ Average Accuracy: {avg_acc:.4f}")
            return {
                'model': model_name,
                'description': description,
                'embedding_dim': embedding_dim,
                'accuracy': avg_acc
            }
        else:
            logger.error("Failed to parse accuracy")
            return None

    except Exception as e:
        logger.error(f"Error: {e}")
        return None
    finally:
        # Restore original cache
        if os.path.exists(backup_file):
            shutil.copy2(backup_file, main_cache)
            os.remove(backup_file)

        # Clean up temp config
        if os.path.exists('configs/temp_test.yaml'):
            os.remove('configs/temp_test.yaml')


def main():
    """Test all CLIP models"""

    logger.info("="*70)
    logger.info("CLIP MODEL COMPARISON (Simple Test)")
    logger.info("="*70)

    # Models to test
    models = [
        ("ViT-B/32", "artifacts/cache/clip_embeddings_ViT-B_32.pkl", "Baseline"),
        ("ViT-B/16", "artifacts/cache/clip_embeddings_ViT-B_16.pkl", "Higher res"),
        ("ViT-L/14", "artifacts/cache/clip_embeddings_ViT-L_14.pkl", "Larger model"),
    ]

    results = []

    for model_name, embedding_file, description in models:
        if os.path.exists(embedding_file):
            result = test_clip_model(model_name, embedding_file, description)
            if result:
                results.append(result)
        else:
            logger.warning(f"Embedding file not found: {embedding_file}")

        time.sleep(3)

    # Report results
    if results:
        logger.info("")
        logger.info("="*70)
        logger.info("RESULTS SUMMARY")
        logger.info("="*70)

        results.sort(key=lambda x: x['accuracy'], reverse=True)

        logger.info(f"{'Model':<12} {'Dim':<6} {'Accuracy':<10} {'Description'}")
        logger.info("-"*50)

        for r in results:
            logger.info(f"{r['model']:<12} {r['embedding_dim']:<6} {r['accuracy']:<10.4f} {r['description']}")

        # Best vs baseline
        best = results[0]
        baseline = next((r for r in results if r['model'] == 'ViT-B/32'), None)

        logger.info("")
        if baseline and best['model'] != 'ViT-B/32':
            improvement = (best['accuracy'] - baseline['accuracy']) * 100
            logger.info(f"Best: {best['model']} with {best['accuracy']:.4f} accuracy")
            logger.info(f"Improvement over baseline: +{improvement:.2f}%")
        else:
            logger.info(f"Best: {best['model']} with {best['accuracy']:.4f} accuracy")

        # Save results
        with open('clip_simple_comparison.json', 'w') as f:
            json.dump({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'results': results
            }, f, indent=2)

    logger.info("="*70)


if __name__ == "__main__":
    main()