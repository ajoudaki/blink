#!/usr/bin/env python3
"""
LoRA fine-tuning for CLIP using text-image similarity.
Clean implementation without decoder networks.
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
# Dataset
# ============================================================================

class ComparisonDataset(Dataset):
    """Dataset for comparison task using CLIP."""

    def __init__(self, data_file, label_file, preprocess, split='train',
                 val_split=0.1, test_split=0.1, seed=42, single_user=None):

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

    def __len__(self):
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

        # Load and preprocess images
        image1 = self.preprocess(Image.open(im1_path).convert('RGB'))
        image2 = self.preprocess(Image.open(im2_path).convert('RGB'))

        # Get label for this target
        label = row[f'{target}_label']

        return {
            'image1': image1,
            'image2': image2,
            'target': target,
            'label': torch.tensor(label, dtype=torch.long)
        }

    def fix_image_path(self, original_path):
        """Convert Excel paths to local paths."""
        filename = os.path.basename(original_path)
        possible_paths = [
            f"data/ffhq/src/{filename}",
            f"data/ffhq2/src/{filename}",
            f"data/ffqh/src/{filename}",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Image not found: {original_path}")


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


def train_epoch(model, dataloader, lora_params, optimizer, cfg, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # Pre-encode target texts (they don't change)
    target_texts = {
        'attractive': "an attractive person",
        'smart': "a smart person",
        'trustworthy': "a trustworthy person"
    }

    text_tokens = {}
    for target, text in target_texts.items():
        text_tokens[target] = clip.tokenize([text]).to(device)

    pbar = tqdm(dataloader, desc="Training")
    for i, batch in enumerate(pbar):
        # Move to device
        image1 = batch['image1'].to(device)
        image2 = batch['image2'].to(device)
        labels = batch['label'].to(device)
        targets = batch['target']

        # Get image features
        image1_features = model.encode_image(image1)
        image2_features = model.encode_image(image2)

        # Get text features for this batch (with gradients)
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

        # Compute loss
        loss = F.cross_entropy(probs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        batch_loss = loss.item()
        total_loss += batch_loss
        preds = probs.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Update progress bar with running average
        running_avg_loss = total_loss / (i + 1)
        pbar.set_postfix({'CE_loss': f'{batch_loss:.4f}', 'avg_loss': f'{running_avg_loss:.4f}'})

    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, cfg, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0

    # Per-target metrics
    target_correct = {'attractive': 0, 'smart': 0, 'trustworthy': 0}
    target_total = {'attractive': 0, 'smart': 0, 'trustworthy': 0}

    # Pre-encode target texts
    target_texts = {
        'attractive': "an attractive person",
        'smart': "a smart person",
        'trustworthy': "a trustworthy person"
    }

    text_tokens = {}
    for target, text in target_texts.items():
        text_tokens[target] = clip.tokenize([text]).to(device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            image1 = batch['image1'].to(device)
            image2 = batch['image2'].to(device)
            labels = batch['label'].to(device)
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

            # Per-target accuracy
            preds = probs.argmax(dim=-1)
            for i, target in enumerate(targets):
                target_correct[target] += (preds[i] == labels[i]).item()
                target_total[target] += 1

    # Compute accuracies
    accuracies = {t: target_correct[t] / max(target_total[t], 1)
                  for t in target_texts.keys()}
    avg_accuracy = np.mean(list(accuracies.values()))

    return total_loss / len(dataloader), avg_accuracy, accuracies


# ============================================================================
# Main
# ============================================================================

@hydra.main(version_base=None, config_path="configs", config_name="lora_config")
def main(cfg: DictConfig):
    """Main training function for LoRA fine-tuning."""

    # Setup
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu.device_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"CLIP Model: {cfg.clip_model}")
    print(f"LoRA rank: {cfg.lora.rank}, alpha: {cfg.lora.alpha}")

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / f"lora_{cfg.clip_model.replace('/', '_')}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # Load CLIP model
    print(f"Loading CLIP model: {cfg.clip_model}")
    model, preprocess = clip.load(cfg.clip_model, device=device)

    # Freeze original parameters
    for param in model.parameters():
        param.requires_grad = False

    # Add LoRA adapters
    lora_layers = add_lora_to_clip(model, cfg.lora.rank, cfg.lora.alpha, device)
    print(f"Added {len(lora_layers)} LoRA adapter layers")

    # Create datasets
    single_user = cfg.data.get('single_user', None)

    train_dataset = ComparisonDataset(
        'data/big_compare_data.xlsx',
        'data/big_compare_label.xlsx',
        preprocess, split='train',
        val_split=cfg.data.val_split,
        test_split=cfg.data.test_split,
        seed=cfg.seed,
        single_user=single_user
    )

    val_dataset = ComparisonDataset(
        'data/big_compare_data.xlsx',
        'data/big_compare_label.xlsx',
        preprocess, split='val',
        val_split=cfg.data.val_split,
        test_split=cfg.data.test_split,
        seed=cfg.seed,
        single_user=single_user
    )

    test_dataset = ComparisonDataset(
        'data/big_compare_data.xlsx',
        'data/big_compare_label.xlsx',
        preprocess, split='test',
        val_split=cfg.data.val_split,
        test_split=cfg.data.test_split,
        seed=cfg.seed,
        single_user=single_user
    )

    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.training.batch_size,
        shuffle=True, num_workers=cfg.data.num_workers
    )

    eval_batch_size = cfg.training.get('eval_batch_size', cfg.training.batch_size * 2)

    val_loader = DataLoader(
        val_dataset, batch_size=eval_batch_size,
        shuffle=False, num_workers=cfg.data.num_workers
    )

    test_loader = DataLoader(
        test_dataset, batch_size=eval_batch_size,
        shuffle=False, num_workers=cfg.data.num_workers
    )

    # Setup optimizer (only LoRA parameters)
    lora_params = []
    for layer in lora_layers.values():
        lora_params.extend([layer.lora_A, layer.lora_B])

    optimizer = torch.optim.AdamW(
        lora_params,
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay
    )

    # Training loop with early stopping
    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    print("\nStarting training...")
    for epoch in range(cfg.training.max_epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, lora_params, optimizer, cfg, device
        )

        # Validate
        val_loss, val_acc, val_accs = evaluate(model, val_loader, cfg, device)

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print progress
        print(f"\nEpoch {epoch+1}/{cfg.training.max_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Val Accuracies - Attractive: {val_accs['attractive']:.4f}, "
              f"Smart: {val_accs['smart']:.4f}, Trustworthy: {val_accs['trustworthy']:.4f}")

        # Early stopping
        if val_acc > best_val_acc + cfg.training.min_delta:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0

            # Save best model (only LoRA parameters)
            torch.save({
                'epoch': epoch,
                'lora_state': {k: v.state_dict() for k, v in lora_layers.items()},
                'val_acc': val_acc,
                'val_accs': val_accs
            }, run_dir / 'best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= cfg.training.patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Load best model
    checkpoint = torch.load(run_dir / 'best_model.pth')
    for name, layer in lora_layers.items():
        layer.load_state_dict(checkpoint['lora_state'][name])

    # Test evaluation
    test_loss, test_acc, test_accs = evaluate(model, test_loader, cfg, device)

    # Print final results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Best Epoch: {best_epoch + 1}")
    print(f"Best Val Acc: {best_val_acc:.4f}")
    print(f"\nTest Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Attractive: {test_accs['attractive']:.4f}")
    print(f"  Smart: {test_accs['smart']:.4f}")
    print(f"  Trustworthy: {test_accs['trustworthy']:.4f}")

    # Save results
    results = {
        'best_epoch': best_epoch + 1,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_accs': test_accs,
        'history': history,
        'config': dict(cfg)
    }

    with open(run_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {run_dir}")


if __name__ == "__main__":
    main()