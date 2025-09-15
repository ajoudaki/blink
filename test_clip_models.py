#!/usr/bin/env python3
"""
Test different CLIP model variants to see if more powerful models improve performance.
"""

import os
import subprocess
import logging
import json
import time
import torch
import clip
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_clip_model(model_name, description):
    """Test a specific CLIP model by modifying train.py temporarily."""

    # First, modify train.py to use the new model
    try:
        # Read the current train.py
        with open('train.py', 'r') as f:
            original_content = f.read()

        # Replace the CLIP model
        modified_content = original_content.replace(
            'model, preprocess = clip.load("ViT-B/32", device=device)',
            f'model, preprocess = clip.load("{model_name}", device=device)'
        )

        # Also update the embedding dimension if needed
        model_dims = {
            "ViT-B/32": 512,
            "ViT-B/16": 512,
            "ViT-L/14": 768,
            "ViT-L/14@336px": 768,
            "RN50": 1024,
            "RN101": 512,
            "RN50x4": 640,
            "RN50x16": 768,
            "RN50x64": 1024
        }

        if model_name in model_dims:
            dim = model_dims[model_name]
            modified_content = modified_content.replace(
                'input_dim = 512  # CLIP embedding size',
                f'input_dim = {dim}  # CLIP embedding size for {model_name}'
            )

        # Write modified version
        with open('train_temp.py', 'w') as f:
            f.write(modified_content)

        # Clear the cache to force re-extraction with new model
        cache_file = 'artifacts/cache/clip_embeddings_cache.pkl'
        if os.path.exists(cache_file):
            os.rename(cache_file, f'{cache_file}.backup_{model_name.replace("/", "_")}')

        logger.info(f"Testing {description} ({model_name})...")

        # Run training with the modified script
        cmd = "python train_temp.py --config-name=unified_optimal_final"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)

        # Parse results
        output_lines = result.stdout.split('\n')
        metrics = {
            'model': model_name,
            'description': description,
            'val_acc': None,
            'test_acc': None,
            'embedding_dim': model_dims.get(model_name, 'unknown')
        }

        for line in output_lines:
            if 'Average Accuracy:' in line:
                try:
                    acc = float(line.split(':')[1].strip())
                    if 'test' in result.stdout[max(0, result.stdout.find(line)-200):result.stdout.find(line)].lower():
                        metrics['test_acc'] = acc
                    else:
                        metrics['val_acc'] = acc
                except:
                    pass

        # Restore cache
        if os.path.exists(f'{cache_file}.backup_{model_name.replace("/", "_")}'):
            os.rename(f'{cache_file}.backup_{model_name.replace("/", "_")}', cache_file)

        if metrics['val_acc']:
            logger.info(f"  ✓ {model_name}: Val={metrics['val_acc']:.4f}, Dim={metrics['embedding_dim']}")
            return metrics
        else:
            logger.error(f"  ✗ Failed to get results for {model_name}")
            return None

    except Exception as e:
        logger.error(f"  ✗ Error testing {model_name}: {e}")
        return None
    finally:
        # Clean up temp file
        if os.path.exists('train_temp.py'):
            os.remove('train_temp.py')
        # Restore original train.py
        with open('train.py', 'w') as f:
            f.write(original_content)


def check_available_models():
    """Check which CLIP models are available."""
    available = []
    models_to_check = [
        ("ViT-B/32", "Vision Transformer Base/32 (current)"),
        ("ViT-B/16", "Vision Transformer Base/16"),
        ("ViT-L/14", "Vision Transformer Large/14"),
        ("ViT-L/14@336px", "Vision Transformer Large/14 @336px"),
        ("RN50", "ResNet-50"),
        ("RN101", "ResNet-101"),
        ("RN50x4", "ResNet-50x4"),
        ("RN50x16", "ResNet-50x16"),
        ("RN50x64", "ResNet-50x64")
    ]

    logger.info("Checking available CLIP models...")
    for model_name, description in models_to_check:
        if model_name in clip.available_models():
            available.append((model_name, description))
            logger.info(f"  ✓ {model_name} available")
        else:
            logger.info(f"  ✗ {model_name} not available")

    return available


# Main execution
logger.info("="*70)
logger.info("CLIP MODEL COMPARISON")
logger.info("="*70)
logger.info("Testing different CLIP model variants")
logger.info("")

# Check available models
available_models = check_available_models()

if not available_models:
    logger.error("No CLIP models available!")
    exit(1)

logger.info(f"\nFound {len(available_models)} available models")
logger.info("")

# Models to test in order of increasing power
models_to_test = [
    ("ViT-B/32", "ViT-Base/32 (current baseline)"),
    ("ViT-B/16", "ViT-Base/16 (2x resolution)"),
    ("ViT-L/14", "ViT-Large/14 (more parameters)"),
    ("ViT-L/14@336px", "ViT-Large/14 @336px (highest resolution)"),
]

# Filter to only available models
models_to_test = [(m, d) for m, d in models_to_test if any(am == m for am, _ in available_models)]

logger.info(f"Testing {len(models_to_test)} CLIP models...")
logger.info("Note: This will take significant time as embeddings need to be re-extracted")
logger.info("")

results = []

for model_name, description in models_to_test:
    result = test_clip_model(model_name, description)
    if result:
        results.append(result)
    time.sleep(5)  # Brief pause between models

# Report results
if results:
    logger.info("")
    logger.info("="*70)
    logger.info("RESULTS SUMMARY")
    logger.info("="*70)

    # Sort by validation accuracy
    results.sort(key=lambda x: x['val_acc'] if x['val_acc'] else 0, reverse=True)

    logger.info(f"{'Model':<20} {'Embedding Dim':<15} {'Val Acc':<10}")
    logger.info("-"*50)

    for r in results:
        logger.info(f"{r['model']:<20} {r['embedding_dim']:<15} {r['val_acc']:<10.4f}")

    # Compare with baseline
    baseline = next((r for r in results if r['model'] == 'ViT-B/32'), None)
    if baseline:
        logger.info("")
        logger.info("Improvements over baseline (ViT-B/32):")
        for r in results:
            if r['model'] != 'ViT-B/32' and r['val_acc']:
                improvement = (r['val_acc'] - baseline['val_acc']) * 100
                if improvement > 0:
                    logger.info(f"  {r['model']}: +{improvement:.2f}%")
                else:
                    logger.info(f"  {r['model']}: {improvement:.2f}%")

    # Save results
    with open('clip_model_comparison.json', 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': results
        }, f, indent=2)

    logger.info("")
    logger.info(f"Results saved to clip_model_comparison.json")

logger.info("="*70)