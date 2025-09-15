#!/usr/bin/env python3
"""
Shared CLIP embedding extractor with caching functionality.
Simple and reusable across both rating and comparison models.
"""

import os
import pickle
import torch
import clip
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_or_extract_embeddings(image_paths, cache_file='clip_embeddings_cache.pkl',
                               device='cuda', force_recompute=False):
    """
    Load embeddings from cache or extract them if not cached.

    Args:
        image_paths: List of image paths to process
        cache_file: Path to cache file
        device: Device for CLIP model
        force_recompute: Force recomputation even if cache exists

    Returns:
        Dictionary mapping image paths to embeddings
    """

    # Try to load from cache
    if os.path.exists(cache_file) and not force_recompute:
        print(f"Loading embeddings from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            embeddings = pickle.load(f)

        # Check if all requested images are in cache
        missing = set(image_paths) - set(embeddings.keys())
        if not missing:
            print(f"✓ All {len(image_paths)} embeddings loaded from cache")
            return {path: embeddings[path] for path in image_paths}
        else:
            print(f"⚠ {len(missing)} images not in cache, extracting...")
    else:
        embeddings = {}
        missing = set(image_paths)

    # Extract missing embeddings
    if missing:
        # Load CLIP model
        print("Loading CLIP model...")
        model, preprocess = clip.load("ViT-B/32", device=device)

        # Image directories to search
        image_dirs = ['data/ffhq/src/', 'data/ffhq2/src/', 'data/ffqh/src/']

        for img_path in tqdm(missing, desc="Extracting new embeddings"):
            embedding = extract_single_embedding(
                img_path, model, preprocess, device, image_dirs
            )
            embeddings[img_path] = embedding

        # Save updated cache
        print(f"Saving {len(embeddings)} embeddings to cache...")
        with open(cache_file, 'wb') as f:
            pickle.dump(embeddings, f)

    return {path: embeddings[path] for path in image_paths}


def extract_single_embedding(img_path, model, preprocess, device, image_dirs):
    """Extract embedding for a single image."""

    img_filename = img_path.split('/')[-1]
    base_name = img_filename.rsplit('.', 1)[0]
    extensions = ['.webp', '.jpg', '.png', '.jpeg']

    # Try to find the image
    for directory in image_dirs:
        for ext in extensions:
            full_path = directory + base_name + ext
            if os.path.exists(full_path):
                try:
                    image = Image.open(full_path).convert('RGB')
                    image_input = preprocess(image).unsqueeze(0).to(device)

                    with torch.no_grad():
                        image_features = model.encode_image(image_input)
                        return image_features.cpu().numpy().squeeze()
                except Exception:
                    pass

    # Return zero embedding if image not found
    return np.zeros(512)