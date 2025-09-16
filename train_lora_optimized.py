#!/usr/bin/env python3
"""
Optimized LoRA fine-tuning with GPU-based transformations.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from PIL import Image
import clip
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# LoRA Implementation
# ============================================================================

class LoRALayer(nn.Module):
    """LoRA adapter layer for efficient fine-tuning."""

    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA parameters
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x):
        # Apply LoRA: x @ A.T @ B.T * scaling
        # Ensure same dtype as input
        lora_A = self.lora_A.to(x.dtype)
        lora_B = self.lora_B.to(x.dtype)
        return x @ lora_A.T @ lora_B.T * self.scaling


def add_lora_to_clip(model, rank=4, alpha=1.0, device='cuda'):
    """Add LoRA adapters to CLIP model."""
    lora_layers = {}

    # Add LoRA to vision transformer
    for name, module in model.visual.named_modules():
        if isinstance(module, nn.Linear):
            # Add LoRA adapter
            lora = LoRALayer(module.in_features, module.out_features, rank, alpha).to(device)
            lora_layers[f"visual.{name}"] = lora

            # Monkey patch the forward method
            original_forward = module.forward
            def new_forward(x, orig=original_forward, adapter=lora):
                return orig(x) + adapter(x)
            module.forward = new_forward

    # Add LoRA to text transformer
    for name, module in model.transformer.named_modules():
        if isinstance(module, nn.Linear):
            # Add LoRA adapter
            lora = LoRALayer(module.in_features, module.out_features, rank, alpha).to(device)
            lora_layers[f"text.{name}"] = lora

            # Monkey patch the forward method
            original_forward = module.forward
            def new_forward(x, orig=original_forward, adapter=lora):
                return orig(x) + adapter(x)
            module.forward = new_forward

    return lora_layers


# ============================================================================
# GPU-based Augmentations
# ============================================================================

class GPUAugmentation(nn.Module):
    """GPU-based image augmentations."""

    def __init__(self, enable=True):
        super().__init__()
        self.enable = enable

    def forward(self, x):
        """Apply augmentations on GPU tensors."""
        if not self.enable or not self.training:
            return x

        batch_size = x.shape[0]
        device = x.device

        # Random horizontal flip (50% chance per image)
        flip_mask = torch.rand(batch_size, device=device) > 0.5
        x[flip_mask] = torch.flip(x[flip_mask], dims=[-1])

        # Random grayscale (20% chance per image)
        gray_mask = torch.rand(batch_size, device=device) < 0.2
        if gray_mask.any():
            # Convert to grayscale for selected images
            gray_images = 0.2989 * x[gray_mask, 0] + 0.5870 * x[gray_mask, 1] + 0.1140 * x[gray_mask, 2]
            x[gray_mask, 0] = gray_images
            x[gray_mask, 1] = gray_images
            x[gray_mask, 2] = gray_images

        # Color jitter (brightness, contrast, saturation)
        # Apply different random values per image
        for i in range(batch_size):
            # Brightness adjustment (±20%)
            brightness_factor = 0.8 + torch.rand(1, device=device).item() * 0.4
            x[i] = x[i] * brightness_factor

            # Contrast adjustment (±20%)
            if torch.rand(1, device=device) < 0.5:
                mean = x[i].mean(dim=[1, 2], keepdim=True)
                contrast_factor = 0.8 + torch.rand(1, device=device).item() * 0.4
                x[i] = (x[i] - mean) * contrast_factor + mean

        # Clamp values to valid range
        x = torch.clamp(x, -2.5, 2.5)  # Approximate range for normalized images

        return x


# ============================================================================
# Optimized Dataset
# ============================================================================

class OptimizedComparisonDataset(Dataset):
    """Optimized dataset with caching and faster loading."""

    def __init__(self, data_file, label_file, preprocess, split='train',
                 val_split=0.1, test_split=0.1, seed=42, single_user=None,
                 cache_images=False, device='cuda'):

        # Load data
        compare_data = pd.read_excel(data_file)
        compare_labels = pd.read_excel(label_file)
        df = compare_data.merge(compare_labels, left_on='_id', right_on='item_id', how='inner')

        # Filter to single user if specified
        if single_user is not None:
            df = df[df['user_id'] == single_user].reset_index(drop=True)
            print(f"Filtered to user {single_user}: {len(df)} samples")

        # Convert labels to binary (2 means first image is preferred)
        self.targets = ['attractive', 'smart', 'trustworthy']
        for target in self.targets:
            df[f'{target}_label'] = (df[target] == 2).astype(int)

        # Split data
        np.random.seed(seed)
        indices = np.arange(len(df))
        np.random.shuffle(indices)

        n_test = int(len(df) * test_split)
        n_val = int(len(df) * val_split)

        if split == 'test':
            indices = indices[:n_test]
        elif split == 'val':
            indices = indices[n_test:n_test + n_val]
        else:  # train
            indices = indices[n_test + n_val:]

        self.df = df.iloc[indices].reset_index(drop=True)
        self.preprocess = preprocess
        self.split = split
        self.cache_images = cache_images and (split == 'train')
        self.device = device

        # Cache for preprocessed images
        self.image_cache = {}

        # Pre-load images if caching is enabled
        if self.cache_images:
            print(f"Pre-loading {len(self.df)} images to cache...")
            for idx in tqdm(range(len(self.df))):
                row = self.df.iloc[idx]
                im1_path = self.fix_image_path(row['im1_path'])
                im2_path = self.fix_image_path(row['im2_path'])

                if im1_path not in self.image_cache:
                    img = Image.open(im1_path).convert('RGB')
                    self.image_cache[im1_path] = self.preprocess(img)

                if im2_path not in self.image_cache:
                    img = Image.open(im2_path).convert('RGB')
                    self.image_cache[im2_path] = self.preprocess(img)
            print(f"Cached {len(self.image_cache)} unique images")

    def __len__(self):
        # Each sample appears once per epoch (but with 3 targets)
        return len(self.df) * len(self.targets)

    def __getitem__(self, idx):
        # Get sample and target
        sample_idx = idx // len(self.targets)
        target_idx = idx % len(self.targets)
        target = self.targets[target_idx]

        row = self.df.iloc[sample_idx]

        # Fix image paths
        im1_path = self.fix_image_path(row['im1_path'])
        im2_path = self.fix_image_path(row['im2_path'])

        # Load images (from cache if available)
        if self.cache_images and im1_path in self.image_cache:
            image1 = self.image_cache[im1_path].clone()
            image2 = self.image_cache[im2_path].clone()
        else:
            # Load and preprocess images
            image1_pil = Image.open(im1_path).convert('RGB')
            image2_pil = Image.open(im2_path).convert('RGB')

            # Apply CLIP preprocessing
            image1 = self.preprocess(image1_pil)
            image2 = self.preprocess(image2_pil)

        return {
            'image1': image1,
            'image2': image2,
            'label': row[f'{target}_label'],
            'target': target
        }

    def fix_image_path(self, original_path):
        """Convert Excel paths to local paths - matching original logic."""
        filename = os.path.basename(original_path)

        # Try multiple directory variations (typos and variations)
        possible_paths = [
            f"data/ffhq/src/{filename}",
            f"data/ffhq2/src/{filename}",
            f"data/ffqh/src/{filename}",  # Note: typo in original
            f"pics_small/{filename}",
        ]

        # Also try with leading zeros if it's a webp file
        if filename.endswith('.webp'):
            try:
                num = filename.replace('.webp', '')
                padded_filename = f"{int(num):05d}.webp"
                possible_paths.extend([
                    f"data/ffhq/src/{padded_filename}",
                    f"data/ffhq2/src/{padded_filename}",
                    f"data/ffqh/src/{padded_filename}",
                ])
            except:
                pass

        for path in possible_paths:
            if os.path.exists(path):
                return path

        # Return original if nothing found (let it fail naturally)
        return original_path


# ============================================================================
# Training
# ============================================================================

def compute_similarity_probs(image1_features, image2_features, text_features, temperature):
    """
    Compute probabilities using CLIP similarity with temperature scaling.
    Returns probability that image1 is chosen over image2 for the given text.
    """
    # Normalize features
    image1_features = F.normalize(image1_features, dim=-1)
    image2_features = F.normalize(image2_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    # Compute similarities
    sim1 = (image1_features * text_features).sum(dim=-1) / temperature
    sim2 = (image2_features * text_features).sum(dim=-1) / temperature

    # Stack similarities and apply softmax
    sims = torch.stack([sim2, sim1], dim=-1)  # [batch, 2]
    probs = F.softmax(sims, dim=-1)

    return probs


def train_epoch(model, dataloader, lora_params, optimizer, cfg, device, gpu_augment=None):
    """Train for one epoch with GPU augmentations."""
    model.train()
    if gpu_augment:
        gpu_augment.training = True

    total_loss = 0
    correct = 0
    total = 0

    # Text templates for augmentation
    templates = cfg.get('text_templates', ["{}", "a {} person"])

    # Pre-encode target texts with all templates
    target_texts = {}
    for template in templates:
        target_texts[f'attractive_{template}'] = template.format("attractive")
        target_texts[f'smart_{template}'] = template.format("smart")
        target_texts[f'trustworthy_{template}'] = template.format("trustworthy")

    text_tokens = {}
    for key, text in target_texts.items():
        text_tokens[key] = clip.tokenize([text]).to(device)

    pbar = tqdm(dataloader, desc="Training")
    for i, batch in enumerate(pbar):
        # Move to device
        image1 = batch['image1'].to(device, non_blocking=True)
        image2 = batch['image2'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        targets = batch['target']

        # Apply GPU augmentations
        if gpu_augment:
            image1 = gpu_augment(image1)
            image2 = gpu_augment(image2)

        # Get image features
        image1_features = model.encode_image(image1)
        image2_features = model.encode_image(image2)

        # Randomly select a template for this batch
        template_idx = torch.randint(0, len(templates), (1,)).item()
        template = templates[template_idx]

        # Get text features for this batch (with gradients)
        text_features_list = []
        for t in targets:
            key = f"{t}_{template}"
            text_feat = model.encode_text(text_tokens[key])
            text_features_list.append(text_feat)
        batch_text_features = torch.cat(text_features_list, dim=0)

        # Compute probabilities
        probs = compute_similarity_probs(
            image1_features, image2_features,
            batch_text_features, cfg.lora.temperature
        )

        # Compute loss
        loss = F.cross_entropy(probs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        batch_loss = loss.item()
        total_loss += batch_loss
        predictions = (probs[:, 1] > 0.5).long()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        # Update progress bar with running average
        avg_loss = total_loss / (i + 1)
        pbar.set_postfix({'CE_loss': batch_loss, 'avg_loss': avg_loss})

    accuracy = correct / total if total > 0 else 0
    avg_loss = total_loss / len(dataloader)

    return avg_loss, accuracy


def evaluate(model, dataloader, cfg, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0

    # Per-target metrics
    target_correct = {'attractive': 0, 'smart': 0, 'trustworthy': 0}
    target_total = {'attractive': 0, 'smart': 0, 'trustworthy': 0}

    # Use the base template for evaluation (consistent evaluation)
    eval_template = cfg.get('text_templates', ["{}", "a {} person"])[0]

    # Pre-encode target texts
    target_texts = {
        'attractive': eval_template.format("attractive"),
        'smart': eval_template.format("smart"),
        'trustworthy': eval_template.format("trustworthy")
    }

    text_tokens = {}
    for target, text in target_texts.items():
        text_tokens[target] = clip.tokenize([text]).to(device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            image1 = batch['image1'].to(device, non_blocking=True)
            image2 = batch['image2'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            targets = batch['target']

            # Get features
            image1_features = model.encode_image(image1)
            image2_features = model.encode_image(image2)

            # Get text features for this batch
            text_features_list = []
            for t in targets:
                text_feat = model.encode_text(text_tokens[t])
                text_features_list.append(text_feat)
            batch_text_features = torch.cat(text_features_list, dim=0)

            # Compute probabilities
            probs = compute_similarity_probs(
                image1_features, image2_features,
                batch_text_features, cfg.lora.temperature
            )

            # Loss
            loss = F.cross_entropy(probs, labels)
            total_loss += loss.item()

            # Predictions
            predictions = (probs[:, 1] > 0.5).long()

            # Update per-target metrics
            for i, t in enumerate(targets):
                target_correct[t] += (predictions[i] == labels[i]).item()
                target_total[t] += 1

    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    total_correct = sum(target_correct.values())
    total_samples = sum(target_total.values())
    accuracy = total_correct / total_samples if total_samples > 0 else 0

    target_accuracies = {}
    for target in target_correct:
        if target_total[target] > 0:
            target_accuracies[target] = target_correct[target] / target_total[target]
        else:
            target_accuracies[target] = 0.0

    return avg_loss, accuracy, target_accuracies


# ============================================================================
# Main Training Loop
# ============================================================================

@hydra.main(config_path="configs", config_name="lora_single_user", version_base=None)
def main(cfg: DictConfig):
    # Set device
    device_id = cfg.get('gpu', {}).get('device_id', 0)
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Enable mixed precision for faster training
    use_amp = cfg.get('use_mixed_precision', True)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Set seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load CLIP model
    print(f"CLIP Model: {cfg.clip_model}")
    print(f"LoRA rank: {cfg.lora.rank}, alpha: {cfg.lora.alpha}")

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = cfg.clip_model.replace("/", "_")
    run_dir = Path(f"runs/lora_{model_name}_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # Load CLIP
    print(f"Loading CLIP model: {cfg.clip_model}")
    model, preprocess = clip.load(cfg.clip_model, device=device)

    # Add LoRA adapters
    lora_layers = add_lora_to_clip(model, rank=cfg.lora.rank, alpha=cfg.lora.alpha, device=device)
    print(f"Added {len(lora_layers)} LoRA adapter layers")

    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False

    # Enable gradients for LoRA only
    for lora in lora_layers.values():
        for param in lora.parameters():
            param.requires_grad = True

    # Create GPU augmentation module
    gpu_augment = GPUAugmentation(enable=cfg.get('enable_image_augmentation', True)).to(device)

    # Create datasets
    single_user = cfg.data.get('single_user', None)
    cache_images = cfg.get('cache_images', False)

    train_dataset = OptimizedComparisonDataset(
        'data/big_compare_data.xlsx',
        'data/big_compare_label.xlsx',
        preprocess, split='train',
        val_split=cfg.data.val_split,
        test_split=cfg.data.test_split,
        seed=cfg.seed,
        single_user=single_user,
        cache_images=cache_images,
        device=device
    )

    val_dataset = OptimizedComparisonDataset(
        'data/big_compare_data.xlsx',
        'data/big_compare_label.xlsx',
        preprocess, split='val',
        val_split=cfg.data.val_split,
        test_split=cfg.data.test_split,
        seed=cfg.seed,
        single_user=single_user,
        cache_images=False,  # No caching for validation
        device=device
    )

    test_dataset = OptimizedComparisonDataset(
        'data/big_compare_data.xlsx',
        'data/big_compare_label.xlsx',
        preprocess, split='test',
        val_split=cfg.data.val_split,
        test_split=cfg.data.test_split,
        seed=cfg.seed,
        single_user=single_user,
        cache_images=False,  # No caching for test
        device=device
    )

    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Create dataloaders with optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,  # Pin memory for faster transfer
        persistent_workers=True if cfg.data.num_workers > 0 else False,  # Keep workers alive
        prefetch_factor=2 if cfg.data.num_workers > 0 else None  # Prefetch batches
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.eval_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        persistent_workers=True if cfg.data.num_workers > 0 else False,
        prefetch_factor=2 if cfg.data.num_workers > 0 else None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.eval_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        [p for lora in lora_layers.values() for p in lora.parameters()],
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay
    )

    # Training loop
    print("\nStarting training...")
    best_val_acc = 0
    patience_counter = 0

    for epoch in range(cfg.training.max_epochs):
        print(f"\nEpoch {epoch+1}/{cfg.training.max_epochs}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, lora_layers, optimizer, cfg, device, gpu_augment
        )

        # Evaluate
        val_loss, val_acc, val_target_accs = evaluate(model, val_loader, cfg, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Val Accuracies - Attractive: {val_target_accs['attractive']:.4f}, "
              f"Smart: {val_target_accs['smart']:.4f}, "
              f"Trustworthy: {val_target_accs['trustworthy']:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'lora_state_dict': {k: v.state_dict() for k, v in lora_layers.items()},
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'cfg': cfg
            }, run_dir / 'best_model.pt')
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= cfg.training.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Final test evaluation
    print("\nFinal test evaluation:")
    test_loss, test_acc, test_target_accs = evaluate(model, test_loader, cfg, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    print(f"Test Accuracies - Attractive: {test_target_accs['attractive']:.4f}, "
          f"Smart: {test_target_accs['smart']:.4f}, "
          f"Trustworthy: {test_target_accs['trustworthy']:.4f}")

    # Save final results
    results = {
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_target_accs': test_target_accs,
        'best_val_acc': best_val_acc,
        'config': dict(cfg)
    }

    with open(run_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {run_dir}")


if __name__ == "__main__":
    main()