#!/usr/bin/env python3
"""LoRA fine-tuning with labeler-specific tokens for CLIP model."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import clip
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import random
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer for efficient fine-tuning."""

    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Initialize LoRA parameters
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x):
        # Convert to same dtype as input
        lora_A = self.lora_A.to(x.dtype)
        lora_B = self.lora_B.to(x.dtype)
        return x @ lora_A.T @ lora_B.T * self.scaling


def add_lora_to_clip(model, rank=4, alpha=1.0, device='cuda'):
    """Add LoRA adapters to CLIP model."""
    lora_layers = []

    # Add LoRA to vision and text transformers' MLP layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'mlp' in name:
            # Add adapter to MLP layers
            in_features = module.in_features
            out_features = module.out_features

            # Create LoRA adapter
            adapter = LoRALayer(in_features, out_features, rank, alpha).to(device)
            lora_layers.append(adapter)

            # Monkey-patch the forward method
            orig = module.forward
            def new_forward(x, orig=orig, adapter=adapter):
                return orig(x) + adapter(x)
            module.forward = new_forward

    return lora_layers


class ComparisonDataset(Dataset):
    """Dataset for pairwise comparisons with labeler-specific tokens."""

    def __init__(self, cfg, split='train', transform=None, seed=42):
        self.cfg = cfg
        self.split = split
        self.transform = transform
        self.device = f"cuda:{cfg.gpu.device_id}" if torch.cuda.is_available() else "cpu"

        # Load data
        df_labels = pd.read_excel('data/big_compare_label.xlsx')
        df_data = pd.read_excel('data/big_compare_data.xlsx')

        # Merge dataframes
        df = df_data.merge(df_labels, left_on='_id', right_on='item_id', how='inner')

        # Get top 2 users by label count
        user_counts = df['user_id'].value_counts()
        self.top_users = user_counts.head(2).index.tolist()
        print(f"Top 2 users: {self.top_users}")
        print(f"User 1: {self.top_users[0]} with {user_counts[self.top_users[0]]} labels")
        print(f"User 2: {self.top_users[1]} with {user_counts[self.top_users[1]]} labels")

        # Filter to top 2 users
        df = df[df['user_id'].isin(self.top_users)]
        print(f"Filtered to top 2 users: {len(df)} samples")

        # Create user ID mapping for special tokens
        self.user_to_token = {
            self.top_users[0]: "<user1>",
            self.top_users[1]: "<user2>"
        }

        # Convert to samples with labeler tokens
        self.samples = []
        for _, row in df.iterrows():
            user_token = self.user_to_token[row['user_id']]

            for attr in ['attractive', 'smart', 'trustworthy']:
                value = row[attr]
                if value in [1, 2]:
                    # Note: 2 means first image (im1) is preferred
                    winner = row['im1_path'] if value == 2 else row['im2_path']
                    loser = row['im2_path'] if value == 2 else row['im1_path']

                    self.samples.append({
                        'winner': winner,
                        'loser': loser,
                        'attribute': attr,
                        'user_token': user_token,
                        'userID': row['user_id']
                    })

        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        random.shuffle(self.samples)

        # Split data
        n_samples = len(self.samples)
        n_val = int(n_samples * cfg.data.val_split)
        n_test = int(n_samples * cfg.data.test_split)

        if split == 'val':
            self.samples = self.samples[:n_val]
        elif split == 'test':
            self.samples = self.samples[n_val:n_val+n_test]
        else:  # train
            self.samples = self.samples[n_val+n_test:]

        # Text templates with labeler tokens
        self.text_templates = cfg.text_templates

        # CLIP preprocessing
        self.preprocess = transform

    def fix_image_path(self, original_path):
        """Fix image path to match actual file locations."""
        filename = os.path.basename(original_path)

        # Try different possible paths
        possible_paths = [
            f"data/ffhq/src/{filename}",
            f"data/ffhq2/src/{filename}",
            f"data/ffqh/src/{filename}",  # Note: typo in original
            f"pics_small/{filename}",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        return original_path

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load images
        winner_path = self.fix_image_path(sample['winner'])
        loser_path = self.fix_image_path(sample['loser'])

        winner_img = Image.open(winner_path).convert('RGB')
        loser_img = Image.open(loser_path).convert('RGB')

        if self.preprocess:
            winner_img = self.preprocess(winner_img)
            loser_img = self.preprocess(loser_img)

        # Select random template
        template = random.choice(self.text_templates)

        # Create text with labeler token
        if '{}' in template:
            text = template.format(sample['attribute'])
        else:
            text = template

        # Add labeler token at the beginning
        text_with_token = f"{sample['user_token']} {text}"

        return {
            'winner_img': winner_img,
            'loser_img': loser_img,
            'text': text_with_token,
            'attribute': sample['attribute'],
            'userID': sample['userID']
        }

    def __len__(self):
        return len(self.samples)


def train_epoch(model, train_loader, lora_layers, optimizer, temperature, device, clip_model, tokenizer):
    """Train for one epoch with labeler tokens."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # Create progress bar
    pbar = tqdm(train_loader, desc='Training')
    running_loss = 0

    for batch_idx, batch in enumerate(pbar):
        # Move to device
        winner_imgs = batch['winner_img'].to(device)
        loser_imgs = batch['loser_img'].to(device)
        texts = batch['text']

        # Encode images
        with torch.cuda.amp.autocast():
            winner_features = model.encode_image(winner_imgs)
            loser_features = model.encode_image(loser_imgs)

            # Encode text with labeler tokens
            text_tokens = clip.tokenize(texts, truncate=True).to(device)
            text_features = model.encode_text(text_tokens)

            # Normalize features
            winner_features = F.normalize(winner_features, dim=-1)
            loser_features = F.normalize(loser_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

            # Compute similarities with temperature scaling
            winner_sim = (winner_features @ text_features.T).diag() / temperature
            loser_sim = (loser_features @ text_features.T).diag() / temperature

            # Create logits for cross-entropy loss
            logits = torch.stack([winner_sim, loser_sim], dim=1)
            labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device)

            # Compute loss
            loss = F.cross_entropy(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        _, predicted = logits.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # Update running loss for display
        if batch_idx == 0:
            running_loss = loss.item()
        else:
            running_loss = 0.9 * running_loss + 0.1 * loss.item()

        # Update progress bar
        pbar.set_postfix({
            'CE_loss': f'{running_loss:.4f}',
            'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
        })

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, val_loader, temperature, device, clip_model, tokenizer):
    """Evaluate model with labeler tokens."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    # Track per-attribute and per-user accuracy
    attr_correct = {'attractive': 0, 'smart': 0, 'trustworthy': 0}
    attr_total = {'attractive': 0, 'smart': 0, 'trustworthy': 0}

    user_correct = {}
    user_total = {}

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating'):
            winner_imgs = batch['winner_img'].to(device)
            loser_imgs = batch['loser_img'].to(device)
            texts = batch['text']
            attributes = batch['attribute']
            userIDs = batch['userID']

            # Encode
            with torch.cuda.amp.autocast():
                winner_features = model.encode_image(winner_imgs)
                loser_features = model.encode_image(loser_imgs)

                text_tokens = clip.tokenize(texts, truncate=True).to(device)
                text_features = model.encode_text(text_tokens)

                # Normalize
                winner_features = F.normalize(winner_features, dim=-1)
                loser_features = F.normalize(loser_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)

                # Compute similarities
                winner_sim = (winner_features @ text_features.T).diag() / temperature
                loser_sim = (loser_features @ text_features.T).diag() / temperature

                # Create logits
                logits = torch.stack([winner_sim, loser_sim], dim=1)
                labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device)

                # Compute loss
                loss = F.cross_entropy(logits, labels)

            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct_mask = (predicted == labels)
            correct += correct_mask.sum().item()
            total += labels.size(0)

            # Track per-attribute accuracy
            for i, attr in enumerate(attributes):
                attr_correct[attr] += correct_mask[i].item()
                attr_total[attr] += 1

            # Track per-user accuracy
            for i, user in enumerate(userIDs):
                if user not in user_correct:
                    user_correct[user] = 0
                    user_total[user] = 0
                user_correct[user] += correct_mask[i].item()
                user_total[user] += 1

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total

    # Compute per-attribute accuracy
    attr_acc = {attr: attr_correct[attr]/attr_total[attr]
                for attr in attr_correct if attr_total[attr] > 0}

    # Compute per-user accuracy
    user_acc = {user: user_correct[user]/user_total[user]
                for user in user_correct if user_total[user] > 0}

    return avg_loss, accuracy, attr_acc, user_acc


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    # Set device
    device = f"cuda:{cfg.gpu.device_id}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Set random seeds
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = cfg.clip_model.replace("/", "_")
    run_dir = Path(f"runs/lora_labeler_tokens_{model_name}_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # Load CLIP model
    print(f"Loading CLIP model: {cfg.clip_model}")
    model, preprocess = clip.load(cfg.clip_model, device=device)

    # Add LoRA adapters
    lora_layers = add_lora_to_clip(model, rank=cfg.lora.rank, alpha=cfg.lora.alpha, device=device)
    print(f"Added {len(lora_layers)} LoRA adapter layers")
    print(f"LoRA rank: {cfg.lora.rank}, alpha: {cfg.lora.alpha}")

    # Freeze original CLIP parameters
    for param in model.parameters():
        param.requires_grad = False

    # Enable gradients for LoRA parameters
    for lora in lora_layers:
        for param in lora.parameters():
            param.requires_grad = True

    # Create datasets
    train_dataset = ComparisonDataset(cfg, split='train', transform=preprocess, seed=cfg.seed)
    val_dataset = ComparisonDataset(cfg, split='val', transform=preprocess, seed=cfg.seed)
    test_dataset = ComparisonDataset(cfg, split='test', transform=preprocess, seed=cfg.seed)

    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.eval_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.eval_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True
    )

    # Create optimizer for LoRA parameters only
    lora_params = []
    for lora in lora_layers:
        lora_params.extend(lora.parameters())

    optimizer = torch.optim.AdamW(
        lora_params,
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay
    )

    # Training loop
    print("\nStarting training with labeler-specific tokens...")
    best_val_acc = 0
    patience_counter = 0

    # Get tokenizer for text encoding
    tokenizer = clip.tokenize

    for epoch in range(cfg.training.max_epochs):
        print(f"\nEpoch {epoch+1}/{cfg.training.max_epochs}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, lora_layers, optimizer,
            cfg.lora.temperature, device, clip, tokenizer
        )

        # Evaluate
        val_loss, val_acc, val_attr_acc, val_user_acc = evaluate(
            model, val_loader, cfg.lora.temperature, device, clip, tokenizer
        )

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Val Accuracies - Attractive: {val_attr_acc.get('attractive', 0):.4f}, "
              f"Smart: {val_attr_acc.get('smart', 0):.4f}, "
              f"Trustworthy: {val_attr_acc.get('trustworthy', 0):.4f}")

        # Print per-user accuracy
        for user, acc in val_user_acc.items():
            token = train_dataset.user_to_token.get(user, user)
            print(f"Val Accuracy for {token} ({user[:8]}...): {acc:.4f}")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            # Save best model
            checkpoint = {
                'epoch': epoch,
                'lora_state_dict': {f'lora_{i}': lora.state_dict()
                                   for i, lora in enumerate(lora_layers)},
                'val_acc': val_acc,
                'val_loss': val_loss,
                'cfg': cfg
            }
            torch.save(checkpoint, run_dir / 'best_model.pt')
            print(f"Saved best model with val_acc: {val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= cfg.training.patience:
                print(f"Early stopping after {epoch+1} epochs")
                break

    # Load best model and evaluate on test set
    print("\nEvaluating best model on test set...")
    checkpoint = torch.load(run_dir / 'best_model.pt')
    for i, lora in enumerate(lora_layers):
        lora.load_state_dict(checkpoint['lora_state_dict'][f'lora_{i}'])

    test_loss, test_acc, test_attr_acc, test_user_acc = evaluate(
        model, test_loader, cfg.lora.temperature, device, clip, tokenizer
    )

    print(f"\nFinal Test Results:")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    print(f"Test Accuracies - Attractive: {test_attr_acc.get('attractive', 0):.4f}, "
          f"Smart: {test_attr_acc.get('smart', 0):.4f}, "
          f"Trustworthy: {test_attr_acc.get('trustworthy', 0):.4f}")

    # Print per-user test accuracy
    for user, acc in test_user_acc.items():
        token = train_dataset.user_to_token.get(user, user)
        print(f"Test Accuracy for {token} ({user[:8]}...): {acc:.4f}")

    # Save results
    results = {
        'best_epoch': checkpoint['epoch'],
        'val_acc': val_acc,
        'val_loss': val_loss,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'test_attr_acc': test_attr_acc,
        'test_user_acc': test_user_acc,
        'val_user_acc': val_user_acc,
        'config': dict(cfg)
    }

    import json
    with open(run_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {run_dir}")


if __name__ == "__main__":
    main()