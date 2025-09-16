import os
import torch
import torch.nn as nn
import clip
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from datetime import datetime
import json
import pandas as pd
from collections import defaultdict


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer for fine-tuning."""

    def __init__(self, in_dim, out_dim, rank=4, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA parameters: A (down) and B (up) projection
        self.lora_A = nn.Parameter(torch.randn(in_dim, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_dim))

    def forward(self, x, weight):
        # Original forward pass + LoRA adaptation
        original = x @ weight
        lora = x @ self.lora_A @ self.lora_B * self.scaling
        return original + lora


class VisualPromptTokens(nn.Module):
    """Learnable visual prompt token for each user (single token like CLS)."""

    def __init__(self, num_users, embed_dim=768):
        super().__init__()
        # Initialize single visual token per user (like CLS token)
        self.visual_tokens = nn.Parameter(torch.randn(num_users, 1, embed_dim) * 0.02)

    def forward(self, user_idx):
        """Get the visual token for a specific user."""
        return self.visual_tokens[user_idx]


def add_lora_to_clip(model, rank=4, alpha=1.0, device='cuda'):
    """Add LoRA adapters to CLIP model."""
    lora_layers = []

    # Add LoRA to text encoder transformer blocks
    for block in model.transformer.resblocks:
        # Add LoRA to self-attention
        in_proj = block.attn.in_proj_weight
        in_dim, out_dim = in_proj.shape[1], in_proj.shape[0]
        lora = LoRALayer(in_dim, out_dim, rank, alpha).to(device)
        lora_layers.append(lora)

        # Replace forward method for attention
        orig_attn_forward = block.attn.forward
        def new_attn_forward(x, lora_layer=lora, orig_forward=orig_attn_forward, in_proj_weight=in_proj):
            # Modify the attention computation to include LoRA
            orig_result = orig_forward(x)
            return orig_result
        block.attn.forward = new_attn_forward

        # Add LoRA to MLP
        for fc in [block.mlp.c_fc, block.mlp.c_proj]:
            in_dim, out_dim = fc.weight.shape[1], fc.weight.shape[0]
            lora = LoRALayer(in_dim, out_dim, rank, alpha).to(device)
            lora_layers.append(lora)

            # Replace forward method
            orig_fc_forward = fc.forward
            weight = fc.weight
            def new_fc_forward(x, lora_layer=lora, orig_forward=orig_fc_forward, w=weight):
                return lora_layer(x, w)
            fc.forward = new_fc_forward

    # Add LoRA to visual encoder transformer blocks
    for block in model.visual.transformer.resblocks:
        # Similar LoRA additions for visual transformer
        for fc in [block.mlp.c_fc, block.mlp.c_proj]:
            in_dim, out_dim = fc.weight.shape[1], fc.weight.shape[0]
            lora = LoRALayer(in_dim, out_dim, rank, alpha).to(device)
            lora_layers.append(lora)

            orig_fc_forward = fc.forward
            weight = fc.weight
            def new_fc_forward(x, lora_layer=lora, orig_forward=orig_fc_forward, w=weight):
                return lora_layer(x, w)
            fc.forward = new_fc_forward

    return lora_layers


class ComparisonDataset(Dataset):
    """Dataset for pairwise comparisons with user-specific tokens."""

    def __init__(self, cfg, split='train', transform=None):
        self.cfg = cfg
        self.split = split
        self.transform = transform

        # Load data
        self.data_path = f'data/pairwise_comparisons_{split}.csv'
        self.df = pd.read_csv(self.data_path)

        # Filter for minimum samples per user
        user_counts = self.df['user'].value_counts()
        valid_users = user_counts[user_counts >= cfg.min_samples_per_user].index.tolist()
        self.df = self.df[self.df['user'].isin(valid_users)]

        # Select top N users if specified
        if cfg.max_users:
            top_users = user_counts[valid_users].head(cfg.max_users).index.tolist()
            self.df = self.df[self.df['user'].isin(top_users)]
        else:
            top_users = valid_users

        # Create user token mapping (same order across splits)
        self.top_users = sorted(top_users)  # Sort to ensure consistent ordering
        self.user_to_token = {user_id: f"<user{i+1}>"
                              for i, user_id in enumerate(self.top_users)}
        self.user_to_idx = {user_id: i for i, user_id in enumerate(self.top_users)}

        print(f"Selected {len(self.top_users)} users with >= {cfg.min_samples_per_user} samples:")
        for i, user_id in enumerate(self.top_users):
            user_samples = len(self.df[self.df['user'] == user_id])
            print(f"User {i+1}: {user_id} with {user_samples} labels")

        self.templates = cfg.text_templates
        self.augment_text = cfg.data.augment_text

        # Store samples with user tokens
        self.samples = []
        for _, row in self.df.iterrows():
            user_token = self.user_to_token[row['user']]
            user_idx = self.user_to_idx[row['user']]
            self.samples.append({
                'left_image': row['image_left_file_name'],
                'right_image': row['image_right_file_name'],
                'label': row['label'],
                'attribute': row['attribute'],
                'user': row['user'],
                'user_token': user_token,
                'user_idx': user_idx
            })

        print(f"Dataset {split}: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load images
        img_dir = 'data/images'
        left_path = os.path.join(img_dir, sample['left_image'])
        right_path = os.path.join(img_dir, sample['right_image'])

        from PIL import Image
        left_img = Image.open(left_path).convert('RGB')
        right_img = Image.open(right_path).convert('RGB')

        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)

        # Get text with user token
        if self.augment_text and self.split == 'train':
            template = np.random.choice(self.templates)
        else:
            template = self.templates[0]

        # Prepend user token to text
        text = template.format(sample['attribute'])
        text_with_token = f"{sample['user_token']} {text}"

        return {
            'left_image': left_img,
            'right_image': right_img,
            'text': text_with_token,
            'label': sample['label'],
            'attribute': sample['attribute'],
            'user': sample['user'],
            'user_idx': sample['user_idx']
        }


def train_model_with_early_stopping(model, visual_prompts, train_loader, val_loader, test_loader,
                                   cfg, lora_layers, device):
    """Train model with early stopping based on validation accuracy."""

    # Setup optimizer for LoRA and visual prompt parameters
    lora_params = []
    for layer in lora_layers:
        lora_params.extend([layer.lora_A, layer.lora_B])

    visual_params = list(visual_prompts.parameters())

    optimizer = torch.optim.AdamW([
        {'params': lora_params, 'lr': cfg.training.learning_rate},
        {'params': visual_params, 'lr': cfg.training.learning_rate}
    ], weight_decay=cfg.training.weight_decay)

    best_val_acc = 0
    patience_counter = 0
    early_stopping_patience = cfg.training.get('early_stopping_patience', 3)
    best_model_path = f"runs/best_model_visual_prompts.pt"

    print(f"\nStarting training with early stopping (patience={early_stopping_patience})...")

    for epoch in range(cfg.training.max_epochs):
        print(f"\n=== Epoch {epoch+1}/{cfg.training.max_epochs} ===")

        # Training phase
        model.train()
        visual_prompts.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        train_bar = tqdm(train_loader, desc='Training')
        for batch_idx, batch in enumerate(train_bar):
            left_imgs = batch['left_image'].to(device)
            right_imgs = batch['right_image'].to(device)
            texts = batch['text']
            labels = batch['label'].to(device)
            user_indices = batch['user_idx'].to(device)

            # Tokenize text with user tokens
            text_tokens = clip.tokenize(texts, truncate=True).to(device)

            # Get visual prompts for users in batch
            batch_visual_prompts = visual_prompts(user_indices)

            # Encode images with visual prompts
            left_features = encode_image_with_prompts(model, left_imgs, batch_visual_prompts)
            right_features = encode_image_with_prompts(model, right_imgs, batch_visual_prompts)

            # Encode text (already contains user tokens)
            text_features = model.encode_text(text_tokens)

            # Normalize features
            left_features = left_features / left_features.norm(dim=-1, keepdim=True)
            right_features = right_features / right_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Compute similarities
            temp = cfg.lora.temperature
            left_sim = (left_features @ text_features.T).squeeze() / temp
            right_sim = (right_features @ text_features.T).squeeze() / temp

            # Compute loss
            logits = torch.stack([left_sim, right_sim], dim=1)
            loss = nn.CrossEntropyLoss()(logits, labels)

            # Update accuracy
            predictions = logits.argmax(dim=1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 10 == 0:
                acc = 100 * train_correct / train_total
                train_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.2f}%")

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Validation phase
        if (epoch + 1) % cfg.training.eval_every == 0:
            val_loss, val_acc, attr_acc, user_acc = evaluate_model(
                model, visual_prompts, val_loader, cfg, device
            )

            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Val Accuracies - Attractive: {attr_acc['attractive']:.4f}, "
                  f"Smart: {attr_acc['smart']:.4f}, Trustworthy: {attr_acc['trustworthy']:.4f}")

            print("\nPer-User Validation Accuracies:")
            for user_id, acc in user_acc.items():
                print(f"  User {user_id[:8]}...: {acc:.4f}")

            # Early stopping logic
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                print(f"Saved best model with val_acc: {val_acc:.4f}")

                # Save best model
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'visual_prompts_state_dict': visual_prompts.state_dict(),
                    'lora_states': [layer.state_dict() for layer in lora_layers],
                    'val_acc': val_acc,
                    'cfg': cfg
                }, best_model_path)
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{early_stopping_patience}")

                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    print(f"Best validation accuracy: {best_val_acc:.4f}")
                    break

    # Load best model for final test evaluation
    print("\n" + "="*50)
    print("Loading best model for final test evaluation")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    visual_prompts.load_state_dict(checkpoint['visual_prompts_state_dict'])
    for i, layer in enumerate(lora_layers):
        layer.load_state_dict(checkpoint['lora_states'][i])

    # Final test evaluation
    print("Final Test Evaluation")
    test_loss, test_acc, test_attr_acc, test_user_acc = evaluate_model(
        model, visual_prompts, test_loader, cfg, device
    )

    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    print(f"Test Accuracies - Attractive: {test_attr_acc['attractive']:.4f}, "
          f"Smart: {test_attr_acc['smart']:.4f}, Trustworthy: {test_attr_acc['trustworthy']:.4f}")

    print("\nPer-User Test Accuracies:")
    for user_id, acc in test_user_acc.items():
        short_id = user_id[:12] + "..." if len(user_id) > 12 else user_id
        print(f"  User {short_id}: {acc:.4f}")

    return best_val_acc, test_acc


def encode_image_with_prompts(model, images, visual_prompts):
    """Encode images with user-specific visual prompts inserted after CLS token."""
    # Get image features up to the transformer
    x = model.visual.conv1(images)
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)

    # Add class embedding and positional encoding
    cls_token = model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
    x = torch.cat([cls_token, x], dim=1)
    x = x + model.visual.positional_embedding.to(x.dtype)

    # Insert visual prompts after CLS token (before patch tokens)
    # x shape: [batch_size, 1 + num_patches, embed_dim]
    # visual_prompts shape: [batch_size, 1, embed_dim]
    x = torch.cat([x[:, :1, :], visual_prompts, x[:, 1:, :]], dim=1)

    # Pass through transformer
    x = model.visual.ln_pre(x)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = model.visual.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD

    # Take the CLS token output (which is now at position 0)
    x = model.visual.ln_post(x[:, 0, :])

    if model.visual.proj is not None:
        x = x @ model.visual.proj

    return x


def evaluate_model(model, visual_prompts, dataloader, cfg, device):
    """Evaluate model on a dataset."""
    model.eval()
    visual_prompts.eval()

    total_loss = 0
    correct = 0
    total = 0

    # Track per-attribute and per-user accuracy
    attr_correct = defaultdict(int)
    attr_total = defaultdict(int)
    user_correct = defaultdict(int)
    user_total = defaultdict(int)

    with torch.no_grad():
        eval_bar = tqdm(dataloader, desc='Evaluating')
        for batch in eval_bar:
            left_imgs = batch['left_image'].to(device)
            right_imgs = batch['right_image'].to(device)
            texts = batch['text']
            labels = batch['label'].to(device)
            attributes = batch['attribute']
            users = batch['user']
            user_indices = batch['user_idx'].to(device)

            # Tokenize text
            text_tokens = clip.tokenize(texts, truncate=True).to(device)

            # Get visual prompts
            batch_visual_prompts = visual_prompts(user_indices)

            # Encode with visual prompts
            left_features = encode_image_with_prompts(model, left_imgs, batch_visual_prompts)
            right_features = encode_image_with_prompts(model, right_imgs, batch_visual_prompts)
            text_features = model.encode_text(text_tokens)

            # Normalize
            left_features = left_features / left_features.norm(dim=-1, keepdim=True)
            right_features = right_features / right_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Compute similarities
            temp = cfg.lora.temperature
            left_sim = (left_features @ text_features.T).squeeze() / temp
            right_sim = (right_features @ text_features.T).squeeze() / temp

            # Compute loss and accuracy
            logits = torch.stack([left_sim, right_sim], dim=1)
            loss = nn.CrossEntropyLoss()(logits, labels)
            total_loss += loss.item()

            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Track per-attribute accuracy
            for i, attr in enumerate(attributes):
                attr_correct[attr] += (predictions[i] == labels[i]).item()
                attr_total[attr] += 1

            # Track per-user accuracy
            for i, user in enumerate(users):
                user_correct[user] += (predictions[i] == labels[i]).item()
                user_total[user] += 1

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    # Calculate per-attribute accuracy
    attr_acc = {}
    for attr in attr_total:
        attr_acc[attr] = attr_correct[attr] / attr_total[attr] if attr_total[attr] > 0 else 0

    # Calculate per-user accuracy
    user_acc = {}
    for user in user_total:
        user_acc[user] = user_correct[user] / user_total[user] if user_total[user] > 0 else 0

    # Sort users for consistent display
    user_acc = dict(sorted(user_acc.items(),
                           key=lambda x: dataloader.dataset.user_to_idx.get(x[0], float('inf'))))

    return avg_loss, accuracy, attr_acc, user_acc


@hydra.main(config_path="configs", config_name="lora_all_users_tokens", version_base=None)
def main(cfg: DictConfig):
    device = torch.device(f"cuda:{cfg.gpu.device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"runs/lora_visual_prompts_{cfg.clip_model.replace('/', '_')}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # Load CLIP model
    print(f"Loading CLIP model: {cfg.clip_model}")
    model, preprocess = clip.load(cfg.clip_model, device=device)

    # Create datasets
    train_dataset = ComparisonDataset(cfg, split='train', transform=preprocess)
    val_dataset = ComparisonDataset(cfg, split='val', transform=preprocess)
    test_dataset = ComparisonDataset(cfg, split='test', transform=preprocess)

    # Get number of users for visual prompts
    num_users = len(train_dataset.top_users)

    # Get visual embedding dimension
    embed_dim = model.visual.transformer.width

    # Create visual prompt tokens
    visual_prompts = VisualPromptTokens(num_users, embed_dim).to(device)
    print(f"Created visual prompt tokens: {num_users} users × 1 token × {embed_dim} dim")

    # Add LoRA adapters
    lora_layers = add_lora_to_clip(model, rank=cfg.lora.rank, alpha=cfg.lora.alpha, device=device)
    print(f"Added {len(lora_layers)} LoRA adapter layers")
    print(f"LoRA rank: {cfg.lora.rank}, alpha: {cfg.lora.alpha}")

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

    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Train model with early stopping
    print("\nStarting training with labeler-specific tokens and visual prompts...")
    best_val_acc, test_acc = train_model_with_early_stopping(
        model, visual_prompts, train_loader, val_loader, test_loader,
        cfg, lora_layers, device
    )

    print(f"\n{'='*50}")
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Final test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()