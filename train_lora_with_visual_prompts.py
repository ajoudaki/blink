#!/usr/bin/env python3
"""LoRA fine-tuning with labeler-specific tokens and visual prompts for CLIP model."""

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


class VisualPromptTokens(nn.Module):
    """Learnable visual prompt token for each user (single token like CLS)."""

    def __init__(self, num_users, embed_dim=768):
        super().__init__()
        self.num_users = num_users
        self.embed_dim = embed_dim
        # Initialize single visual token per user (like CLS token)
        self.visual_tokens = nn.Parameter(torch.randn(num_users, 1, embed_dim) * 0.02)

    def forward(self, user_indices):
        """Get visual token for batch of user indices."""
        return self.visual_tokens[user_indices]  # [batch, 1, embed_dim]


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


def modify_vit_for_visual_prompts(model, visual_prompts):
    """Modify ViT forward pass to include visual prompt tokens."""

    # Store original encode_image method
    orig_encode_image = model.encode_image

    def new_encode_image(images, user_indices=None):
        """Modified image encoding with user visual prompts."""
        dtype = model.dtype

        # Process images through conv1 patch embedding
        x = model.visual.conv1(images.type(dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [batch, embed_dim, grid**2]
        x = x.permute(0, 2, 1)  # [batch, grid**2, embed_dim]

        # Add class token
        x = torch.cat([model.visual.class_embedding.to(dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=dtype, device=x.device), x], dim=1)

        # Add positional embedding to image tokens
        x = x + model.visual.positional_embedding.to(dtype)

        # Insert user visual prompt token if provided (single token per user)
        if user_indices is not None:
            batch_size = x.shape[0]
            # Get visual token for this batch of users (single token like CLS)
            user_token = visual_prompts(user_indices).to(dtype)  # [batch, 1, embed_dim]

            # Split x into CLS token and patch embeddings
            cls_tokens = x[:, :1, :]  # [batch, 1, embed_dim]
            patch_tokens = x[:, 1:, :]  # [batch, num_patches, embed_dim]

            # Concatenate: [CLS], user_token, patch_tokens
            x = torch.cat([cls_tokens, user_token, patch_tokens], dim=1)

        # Apply layer norm pre
        x = model.visual.ln_pre(x)

        # Pass through transformer
        x = x.permute(1, 0, 2)  # [seq_len, batch, embed_dim]
        x = model.visual.transformer(x)
        x = x.permute(1, 0, 2)  # [batch, seq_len, embed_dim]

        # Extract CLS token output
        x = model.visual.ln_post(x[:, 0, :])

        # Project if needed
        if model.visual.proj is not None:
            x = x @ model.visual.proj

        return x

    # Replace the encode_image method
    model.encode_image = new_encode_image
    return model


class ComparisonDataset(Dataset):
    """Dataset for pairwise comparisons with labeler-specific tokens and visual prompts."""

    def __init__(self, cfg, split='train', transform=None, seed=42):
        self.cfg = cfg
        self.split = split
        self.preprocess = transform
        self.device = f"cuda:{cfg.gpu.device_id}" if torch.cuda.is_available() else "cpu"

        # Load data
        df_labels = pd.read_excel('data/big_compare_label.xlsx')
        df_data = pd.read_excel('data/big_compare_data.xlsx')

        # Merge dataframes
        df = df_data.merge(df_labels, left_on='_id', right_on='item_id', how='inner')

        # Get all users with sufficient data
        user_counts = df['user_id'].value_counts()

        # Filter users with at least 100 samples (configurable)
        min_samples = getattr(cfg, 'min_samples_per_user', 100)
        self.top_users = [user_id for user_id, count in user_counts.items() if count >= min_samples]

        # Optionally limit to top N users
        max_users = getattr(cfg, 'max_users', None)
        if max_users:
            self.top_users = self.top_users[:max_users]

        print(f"Selected {len(self.top_users)} users with >= {min_samples} samples:")
        for i, user_id in enumerate(self.top_users):
            print(f"User {i+1}: {user_id} with {user_counts[user_id]} labels")

        # Filter to selected users
        df = df[df['user_id'].isin(self.top_users)]

        # Create user ID to token mapping for text
        self.user_to_token = {user_id: f"<user{i+1}>"
                             for i, user_id in enumerate(self.top_users)}

        # Create user ID to index mapping for visual prompts
        self.user_to_idx = {user_id: i for i, user_id in enumerate(self.top_users)}

        # Shuffle with seed
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

        # Split data
        n = len(df)
        if split == 'train':
            start_idx = 0
            end_idx = int(n * 0.7)
        elif split == 'val':
            start_idx = int(n * 0.7)
            end_idx = int(n * 0.85)
        else:  # test
            start_idx = int(n * 0.85)
            end_idx = n

        df = df.iloc[start_idx:end_idx]

        # Process into samples (using comparison data format)
        self.samples = []

        # Iterate through each user's data
        for user_id in self.top_users:
            user_df = df[df['user_id'] == user_id]

            for _, row in user_df.iterrows():
                # Get the comparison pair ID
                item_id = row['item_id']

                # Find matching comparison data
                data_row = df_data[df_data['_id'] == item_id]
                if len(data_row) == 0:
                    continue

                data_row = data_row.iloc[0]

                # Extract image paths
                im1_path = data_row['im1_path']
                im2_path = data_row['im2_path']

                # For each attribute with a strong preference (>3)
                for attribute in ['attractive', 'smart', 'trustworthy']:
                    score = row[attribute]
                    if score > 3:  # Positive preference for first image
                        sample = {
                            'winner': im1_path,
                            'loser': im2_path,
                            'attribute': attribute,
                            'userID': user_id,
                            'user_token': self.user_to_token[user_id],
                            'user_idx': self.user_to_idx[user_id]
                        }
                        self.samples.append(sample)
                    elif score < 3:  # Negative preference (second image wins)
                        sample = {
                            'winner': im2_path,
                            'loser': im1_path,
                            'attribute': attribute,
                            'userID': user_id,
                            'user_token': self.user_to_token[user_id],
                            'user_idx': self.user_to_idx[user_id]
                        }
                        self.samples.append(sample)

        # Shuffle samples
        random.shuffle(self.samples)

        # Text templates for augmentation
        self.text_templates = getattr(cfg.data, 'text_templates', ['{}'])

        print(f"Dataset {split}: {len(self.samples)} samples")

    def fix_image_path(self, path):
        """Fix various image path formats."""
        path = str(path)
        if 'ffhq\\src\\' in path:
            parts = path.split('ffhq\\src\\')
            filename = parts[-1].replace('\\', '').replace('.jpg', '')
        elif 'ffhq/src/' in path:
            parts = path.split('ffhq/src/')
            filename = parts[-1].replace('/', '').replace('.jpg', '')
        else:
            filename = path.replace('.jpg', '')
        return f'data/ffhq/src/{filename}.jpg'

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
            'userID': sample['userID'],
            'user_idx': sample['user_idx']  # Add user index for visual prompts
        }

    def __len__(self):
        return len(self.samples)


def train_epoch(model, train_loader, lora_layers, visual_prompts, optimizer, temperature, device):
    """Train for one epoch with labeler tokens and visual prompts."""
    model.train()
    visual_prompts.train()
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
        user_indices = batch['user_idx'].to(device)

        # Encode images with user visual prompts
        with torch.cuda.amp.autocast():
            winner_features = model.encode_image(winner_imgs, user_indices)
            loser_features = model.encode_image(loser_imgs, user_indices)

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


def evaluate(model, val_loader, visual_prompts, temperature, device):
    """Evaluate model with labeler tokens and visual prompts."""
    model.eval()
    visual_prompts.eval()
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
            user_indices = batch['user_idx'].to(device)

            # Encode with user visual prompts
            with torch.cuda.amp.autocast():
                winner_features = model.encode_image(winner_imgs, user_indices)
                loser_features = model.encode_image(loser_imgs, user_indices)

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


@hydra.main(config_path="configs", config_name="lora_all_users_tokens", version_base=None)
def main(cfg: DictConfig):
    device = f"cuda:{cfg.gpu.device_id}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Set random seeds
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = cfg.clip_model.replace("/", "_")
    run_dir = Path(f"runs/lora_visual_prompts_{model_name}_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # Load CLIP model
    print(f"Loading CLIP model: {cfg.clip_model}")
    model, preprocess = clip.load(cfg.clip_model, device=device)

    # Create datasets first to get number of users
    train_dataset = ComparisonDataset(cfg, split='train', transform=preprocess, seed=cfg.seed)
    num_users = len(train_dataset.top_users)

    # Get visual embedding dimension
    embed_dim = model.visual.transformer.width

    # Create visual prompt tokens
    # Single visual token per user (like CLS token)
    visual_prompts = VisualPromptTokens(num_users, embed_dim).to(device)
    print(f"Created visual prompt tokens: {num_users} users × 1 token × {embed_dim} dim")

    # Modify ViT to use visual prompts
    model = modify_vit_for_visual_prompts(model, visual_prompts)

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

    # Enable gradients for visual prompts
    for param in visual_prompts.parameters():
        param.requires_grad = True

    # Create remaining datasets
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

    # Create optimizer for LoRA parameters and visual prompts
    lora_params = []
    for lora in lora_layers:
        lora_params.extend(lora.parameters())

    # Add visual prompt parameters
    visual_params = list(visual_prompts.parameters())

    optimizer = torch.optim.AdamW(
        lora_params + visual_params,
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay
    )

    # Training loop
    print("\nStarting training with labeler-specific tokens and visual prompts...")
    best_val_acc = 0
    patience_counter = 0

    for epoch in range(cfg.training.max_epochs):
        print(f"\n=== Epoch {epoch+1}/{cfg.training.max_epochs} ===")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, lora_layers, visual_prompts, optimizer,
            cfg.lora.temperature, device
        )

        # Evaluate
        if (epoch + 1) % cfg.training.eval_every == 0:
            val_loss, val_acc, attr_acc, user_acc = evaluate(
                model, val_loader, visual_prompts,
                cfg.lora.temperature, device
            )

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Print per-attribute accuracy
            print(f"Val Accuracies - Attractive: {attr_acc.get('attractive', 0):.4f}, "
                  f"Smart: {attr_acc.get('smart', 0):.4f}, "
                  f"Trustworthy: {attr_acc.get('trustworthy', 0):.4f}")

            # Print per-user accuracy
            for user_id in train_dataset.top_users[:6]:  # Show first 6 users
                user_token = train_dataset.user_to_token[user_id]
                acc = user_acc.get(user_id, 0)
                print(f"Val Accuracy for {user_token} ({user_id[:12]}...): {acc:.4f}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'lora_state_dict': [lora.state_dict() for lora in lora_layers],
                    'visual_prompts_state_dict': visual_prompts.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, run_dir / 'best_model.pt')
                print(f"Saved best model with val_acc: {val_acc:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= cfg.training.patience:
                print(f"Early stopping after {patience_counter} epochs without improvement")
                break

    # Final test evaluation
    print("\n=== Final Test Evaluation ===")
    test_loss, test_acc, test_attr_acc, test_user_acc = evaluate(
        model, test_loader, visual_prompts,
        cfg.lora.temperature, device
    )

    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    print(f"Test Accuracies - Attractive: {test_attr_acc.get('attractive', 0):.4f}, "
          f"Smart: {test_attr_acc.get('smart', 0):.4f}, "
          f"Trustworthy: {test_attr_acc.get('trustworthy', 0):.4f}")

    # Print final per-user test accuracy
    for user_id in train_dataset.top_users[:6]:  # Show first 6 users
        user_token = train_dataset.user_to_token[user_id]
        acc = test_user_acc.get(user_id, 0)
        print(f"Test Accuracy for {user_token} ({user_id[:12]}...): {acc:.4f}")

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Final test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()