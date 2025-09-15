#!/usr/bin/env python3
"""
Test different CLIP models with proper path mapping to local images.
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import clip
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json
import time
from datetime import datetime

# CLIP model dimensions
CLIP_DIMENSIONS = {
    "ViT-B/32": 512,
    "ViT-B/16": 512,
    "ViT-L/14": 768,
    "ViT-L/14@336px": 768,
}


def fix_image_path(original_path):
    """Convert Excel paths to local paths."""
    # Extract just the filename
    filename = os.path.basename(original_path)

    # Check multiple possible locations
    possible_paths = [
        f"data/ffhq/src/{filename}",
        f"data/ffhq2/src/{filename}",
        f"data/ffqh/src/{filename}",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    # If not found, return None
    return None


def extract_embeddings_for_model(model_name, device='cuda'):
    """Extract embeddings using specified CLIP model."""

    print(f"\n{'='*60}")
    print(f"Extracting embeddings with {model_name}")
    print('='*60)

    # Load the data to get image paths
    print("Loading image paths from Excel files...")
    compare_labels = pd.read_excel('data/big_compare_label.xlsx')
    compare_data = pd.read_excel('data/big_compare_data.xlsx')
    df = compare_data.merge(compare_labels, left_on='_id', right_on='item_id', how='inner')

    # Get all unique image paths
    all_paths = list(df['im1_path'].unique()) + list(df['im2_path'].unique())
    all_paths = list(set(all_paths))
    print(f"Found {len(all_paths)} unique image paths in Excel")

    # Fix paths and filter to existing files
    path_mapping = {}
    valid_paths = []

    for orig_path in all_paths:
        fixed_path = fix_image_path(orig_path)
        if fixed_path:
            path_mapping[orig_path] = fixed_path
            valid_paths.append(fixed_path)

    valid_paths = list(set(valid_paths))  # Remove duplicates
    print(f"Found {len(valid_paths)} valid local image files")

    # Cache file for this model
    cache_file = f"artifacts/cache/clip_embeddings_{model_name.replace('/', '_').replace('@', '_')}.pkl"
    os.makedirs("artifacts/cache", exist_ok=True)

    # Check if we already have embeddings
    if os.path.exists(cache_file):
        print(f"Loading cached embeddings from {cache_file}")
        with open(cache_file, 'rb') as f:
            embeddings_local = pickle.load(f)
        print(f"Loaded {len(embeddings_local)} cached embeddings")
    else:
        embeddings_local = {}

    # Extract missing embeddings
    missing_paths = [p for p in valid_paths if p not in embeddings_local]

    if missing_paths:
        print(f"Extracting {len(missing_paths)} new embeddings...")

        # Load CLIP model
        model, preprocess = clip.load(model_name, device=device)

        for path in tqdm(missing_paths, desc="Extracting"):
            try:
                image = Image.open(path).convert('RGB')
                image_input = preprocess(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    image_features = model.encode_image(image_input)
                    image_features = image_features / image_features.norm(dim=1, keepdim=True)
                    embeddings_local[path] = image_features.cpu().numpy().squeeze()
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue

        # Save updated cache
        with open(cache_file, 'wb') as f:
            pickle.dump(embeddings_local, f)
        print(f"Saved {len(embeddings_local)} embeddings to cache")

    # Map back to original paths
    embeddings = {}
    for orig_path, local_path in path_mapping.items():
        if local_path in embeddings_local:
            embeddings[orig_path] = embeddings_local[local_path]

    print(f"Mapped {len(embeddings)} embeddings to original paths")

    # Get embedding dimension
    if embeddings:
        embedding_dim = next(iter(embeddings.values())).shape[0]
        print(f"Embedding dimension: {embedding_dim}")
    else:
        embedding_dim = CLIP_DIMENSIONS.get(model_name, "unknown")

    return embeddings, embedding_dim


def train_and_evaluate(embeddings, embedding_dim, model_name):
    """Train and evaluate model with given embeddings."""

    print(f"\nTraining with {model_name} embeddings...")

    # Simplified training for quick testing
    # We'll use the existing train.py infrastructure

    # Save embeddings with standard name for train.py
    temp_cache = "artifacts/cache/clip_embeddings_cache.pkl"
    original_cache = None

    # Backup original if exists
    if os.path.exists(temp_cache):
        original_cache = f"{temp_cache}.backup"
        os.rename(temp_cache, original_cache)

    # Save our embeddings
    with open(temp_cache, 'wb') as f:
        pickle.dump(embeddings, f)

    # Run training
    import subprocess
    cmd = "python train.py --config-name=unified_optimal_final training.epochs=30"

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)

        # Parse results
        output_lines = result.stdout.split('\n')
        val_acc = None

        for line in output_lines:
            if 'Average Accuracy:' in line:
                try:
                    val_acc = float(line.split(':')[1].strip())
                except:
                    pass

        # Restore original cache
        if original_cache:
            os.rename(original_cache, temp_cache)

        return val_acc

    except Exception as e:
        print(f"Error during training: {e}")

        # Restore original cache
        if original_cache and os.path.exists(original_cache):
            os.rename(original_cache, temp_cache)

        return None


def main():
    """Test different CLIP models."""

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Models to test
    models_to_test = [
        ("ViT-B/32", "Current baseline"),
        ("ViT-B/16", "Higher resolution patches"),
        ("ViT-L/14", "Larger model"),
    ]

    # Check which models are available
    available_models = []
    print("\nChecking available CLIP models...")
    for model_name, description in models_to_test:
        if model_name in clip.available_models():
            available_models.append((model_name, description))
            print(f"  ✓ {model_name}: {description}")
        else:
            print(f"  ✗ {model_name}: Not available")

    if not available_models:
        print("No CLIP models available!")
        return

    # Test each model
    results = []

    for model_name, description in available_models:
        print(f"\n{'='*70}")
        print(f"Testing {model_name}: {description}")
        print('='*70)

        # Extract embeddings
        embeddings, embedding_dim = extract_embeddings_for_model(model_name, device)

        if not embeddings:
            print(f"Failed to extract embeddings for {model_name}")
            continue

        # Train and evaluate
        val_acc = train_and_evaluate(embeddings, embedding_dim, model_name)

        if val_acc:
            results.append({
                'model': model_name,
                'description': description,
                'embedding_dim': embedding_dim,
                'validation_accuracy': val_acc,
                'num_embeddings': len(embeddings)
            })

            print(f"\nResults for {model_name}:")
            print(f"  Embedding dimension: {embedding_dim}")
            print(f"  Validation accuracy: {val_acc:.4f}")

    # Compare results
    if results:
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)

        print(f"\n{'Model':<15} {'Dim':<8} {'Val Acc':<12} {'Embeddings':<12}")
        print("-"*55)

        for r in results:
            print(f"{r['model']:<15} {r['embedding_dim']:<8} "
                  f"{r['validation_accuracy']:<12.4f} {r['num_embeddings']:<12}")

        # Find best model
        best = max(results, key=lambda x: x['validation_accuracy'])
        baseline = next((r for r in results if r['model'] == 'ViT-B/32'), None)

        if baseline and best['model'] != 'ViT-B/32':
            improvement = (best['validation_accuracy'] - baseline['validation_accuracy']) * 100
            print(f"\nBest model: {best['model']} with {best['validation_accuracy']:.4f} accuracy")
            print(f"Improvement over baseline: +{improvement:.2f}%")

        # Save results
        output_file = 'clip_model_comparison_final.json'
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'results': results
            }, f, indent=2)

        print(f"\nResults saved to {output_file}")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()