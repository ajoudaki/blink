import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import clip
from PIL import Image
from datetime import datetime
import random
from collections import defaultdict
import hydra
from omegaconf import DictConfig

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x):
        # Get LoRA matrices in the correct dtype
        lora_A = self.lora_A.to(x.dtype)
        lora_B = self.lora_B.to(x.dtype)

        # Compute low-rank update
        return x @ lora_A.T @ lora_B.T * self.scaling

def add_lora_to_clip(model, rank=4, alpha=1.0, device='cuda'):
    """Add LoRA adapters to CLIP model MLP layers"""
    lora_layers = []

    # Add LoRA to vision encoder MLP layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'mlp' in name:
            in_features = module.in_features
            out_features = module.out_features

            # Create LoRA adapter
            adapter = LoRALayer(in_features, out_features, rank, alpha).to(device)
            lora_layers.append(adapter)

            # Store original forward
            original_forward = module.forward

            # Create new forward with LoRA
            def new_forward(self, x, adapter=adapter, original=original_forward):
                original_output = original(x)
                lora_output = adapter(x)
                return original_output + lora_output

            # Replace forward method
            module.forward = new_forward.__get__(module, type(module))

    print(f"Added {len(lora_layers)} LoRA adapter layers")
    return lora_layers

class MultiUserDataset(Dataset):
    def __init__(self, data_file, image_dir, user_list, split='train', transform=None, text_templates=None):
        self.data = pd.read_excel(data_file)
        self.image_dir = image_dir
        self.transform = transform
        self.split = split
        self.text_templates = text_templates or ["{}"]

        # Filter to only include specified users with enough data
        self.data = self.data[self.data['user_id'].isin(user_list)]

        # Create user ID to token mapping
        self.user_to_token = {user_id: f"<user{i+1}>"
                             for i, user_id in enumerate(user_list)}

        # Create samples with user tokens
        self.samples = []
        for _, row in self.data.iterrows():
            img_path = f"{image_dir}/{row['item_id']}.jpg"
            user_token = self.user_to_token[row['user_id']]

            for target in ['attractive', 'smart', 'trustworthy']:
                if row[target] > 3:  # Positive label
                    self.samples.append({
                        'img_path': img_path,
                        'winner': target,
                        'user_id': row['user_id'],
                        'user_token': user_token
                    })

        # Shuffle and optionally duplicate with text augmentation
        random.shuffle(self.samples)

        if split == 'train' and len(self.text_templates) > 1:
            augmented_samples = []
            for sample in self.samples:
                for _ in range(len(self.text_templates)):
                    augmented_samples.append(sample.copy())
            self.samples = augmented_samples

        print(f"{split} dataset: {len(self.samples)} samples for {len(user_list)} users")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        try:
            img = Image.open(sample['img_path']).convert('RGB')
        except:
            # If image fails to load, use a blank image
            img = Image.new('RGB', (224, 224), (128, 128, 128))

        if self.transform:
            img = self.transform(img)

        # Select random template for this sample
        template = random.choice(self.text_templates)
        text = template.format(sample['winner'])

        # Add user token at the beginning
        text_with_token = f"{sample['user_token']} {text}"

        return img, text_with_token, sample['winner'], sample['user_id'], sample['user_token']

def train_epoch(model, train_loader, clip_model, optimizer, lora_layers, device, temperature=0.15):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # Set LoRA layers to training mode
    for layer in lora_layers:
        layer.train()

    progress_bar = tqdm(train_loader, desc='Training')
    for batch_idx, (imgs, texts, targets, user_ids, user_tokens) in enumerate(progress_bar):
        imgs = imgs.to(device)

        # Encode images with LoRA-adapted model
        with torch.cuda.amp.autocast():
            image_features = model.encode_image(imgs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Tokenize and encode text with user tokens
            text_tokens = clip.tokenize(texts, truncate=True).to(device)
            text_features = model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Compute similarity with temperature
            logits = (image_features @ text_features.T) / temperature

            # Create labels (diagonal is correct)
            labels = torch.arange(len(imgs)).to(device)

            # Cross-entropy loss (symmetric)
            loss_i2t = nn.CrossEntropyLoss()(logits, labels)
            loss_t2i = nn.CrossEntropyLoss()(logits.T, labels)
            loss = (loss_i2t + loss_t2i) / 2

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += len(imgs)

        # Update progress bar
        if batch_idx % 10 == 0:
            progress_bar.set_postfix({
                'CE_loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
            })

    return total_loss / len(train_loader), correct / total

def evaluate(model, val_loader, device, temperature=0.15, user_list=None):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    # Per-attribute and per-user tracking
    attribute_correct = defaultdict(int)
    attribute_total = defaultdict(int)
    user_correct = defaultdict(int)
    user_total = defaultdict(int)

    with torch.no_grad():
        for imgs, texts, targets, user_ids, user_tokens in tqdm(val_loader, desc='Evaluating'):
            imgs = imgs.to(device)

            with torch.cuda.amp.autocast():
                # Encode images
                image_features = model.encode_image(imgs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # Tokenize and encode text with user tokens
                text_tokens = clip.tokenize(texts, truncate=True).to(device)
                text_features = model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # Compute similarity
                logits = (image_features @ text_features.T) / temperature

                # Create labels
                labels = torch.arange(len(imgs)).to(device)

                # Compute loss
                loss_i2t = nn.CrossEntropyLoss()(logits, labels)
                loss_t2i = nn.CrossEntropyLoss()(logits.T, labels)
                loss = (loss_i2t + loss_t2i) / 2

            total_loss += loss.item()

            # Compute accuracy
            pred = logits.argmax(dim=1)
            batch_correct = (pred == labels)
            correct += batch_correct.sum().item()
            total += len(imgs)

            # Track per-attribute accuracy
            for i, (target, is_correct) in enumerate(zip(targets, batch_correct)):
                attribute_correct[target] += is_correct.item()
                attribute_total[target] += 1

            # Track per-user accuracy
            for i, (user_id, user_token, is_correct) in enumerate(zip(user_ids, user_tokens, batch_correct)):
                user_correct[user_token] += is_correct.item()
                user_total[user_token] += 1

    # Compute per-attribute accuracy
    attr_accs = {attr: attribute_correct[attr] / attribute_total[attr]
                 for attr in attribute_total}

    # Compute per-user accuracy
    user_accs = {token: user_correct[token] / user_total[token]
                 for token in user_total}

    return total_loss / len(val_loader), correct / total, attr_accs, user_accs

@hydra.main(config_path="configs", config_name="lora_all_users", version_base=None)
def main(cfg: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"runs/lora_all_users_{cfg.clip_model.replace('/', '_')}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # Load CLIP model
    print(f"Loading CLIP model: {cfg.clip_model}")
    model, preprocess = clip.load(cfg.clip_model, device=device)

    # Add LoRA adapters
    lora_layers = add_lora_to_clip(model, rank=cfg.lora.rank, alpha=cfg.lora.alpha, device=device)
    print(f"LoRA rank: {cfg.lora.rank}, alpha: {cfg.lora.alpha}")

    # Get all users with sufficient data (excluding user with only 3 samples)
    all_users_df = pd.read_excel(cfg.data.train_file)
    user_counts = all_users_df['user_id'].value_counts()

    # Filter users with at least min_samples
    min_samples = cfg.data.get('min_samples_per_user', 100)
    valid_users = [user_id for user_id, count in user_counts.items()
                  if count >= min_samples]

    print(f"\nUsing {len(valid_users)} users with >= {min_samples} samples:")
    for i, user_id in enumerate(valid_users):
        count = user_counts[user_id]
        print(f"User {i+1} ({user_id}): {count} samples")

    # Create datasets
    train_dataset = MultiUserDataset(
        cfg.data.train_file,
        cfg.data.image_dir,
        valid_users,
        split='train',
        transform=preprocess,
        text_templates=cfg.text_templates
    )

    val_dataset = MultiUserDataset(
        cfg.data.val_file,
        cfg.data.image_dir,
        valid_users,
        split='val',
        transform=preprocess,
        text_templates=[cfg.text_templates[0]]  # No augmentation for validation
    )

    test_dataset = MultiUserDataset(
        cfg.data.test_file,
        cfg.data.image_dir,
        valid_users,
        split='test',
        transform=preprocess,
        text_templates=[cfg.text_templates[0]]
    )

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

    # Setup optimizer (only optimize LoRA parameters)
    lora_params = []
    for layer in lora_layers:
        lora_params.extend([layer.lora_A, layer.lora_B])

    optimizer = optim.AdamW(lora_params, lr=cfg.training.lr)

    # Training loop
    best_val_acc = 0
    print("\nStarting training with user-specific tokens for all users...")

    for epoch in range(cfg.training.max_epochs):
        print(f"\nEpoch {epoch+1}/{cfg.training.max_epochs}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, model, optimizer, lora_layers,
            device, temperature=cfg.lora.temperature
        )

        # Validate
        val_loss, val_acc, attr_accs, user_accs = evaluate(
            model, val_loader, device,
            temperature=cfg.lora.temperature,
            user_list=valid_users
        )

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Print per-attribute accuracy
        print(f"Val Accuracies - Attractive: {attr_accs.get('attractive', 0):.4f}, "
              f"Smart: {attr_accs.get('smart', 0):.4f}, "
              f"Trustworthy: {attr_accs.get('trustworthy', 0):.4f}")

        # Print per-user accuracy
        print("\nPer-User Validation Accuracies:")
        for i, user_id in enumerate(valid_users):
            token = f"<user{i+1}>"
            acc = user_accs.get(token, 0)
            print(f"  {token} ({user_id[:12]}...): {acc:.4f}")

        # Calculate average user accuracy
        avg_user_acc = np.mean(list(user_accs.values()))
        print(f"Average User Accuracy: {avg_user_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'lora_state': [layer.state_dict() for layer in lora_layers],
                'val_acc': val_acc,
                'user_accs': user_accs,
                'attr_accs': attr_accs,
                'user_mapping': train_dataset.user_to_token
            }
            torch.save(checkpoint, f"{run_dir}/best_model.pt")
            print(f"Saved best model with val_acc: {val_acc:.4f}")

    # Test evaluation
    print("\n" + "="*50)
    print("Final Test Evaluation")
    test_loss, test_acc, test_attr_accs, test_user_accs = evaluate(
        model, test_loader, device,
        temperature=cfg.lora.temperature,
        user_list=valid_users
    )

    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    print(f"Test Accuracies - Attractive: {test_attr_accs.get('attractive', 0):.4f}, "
          f"Smart: {test_attr_accs.get('smart', 0):.4f}, "
          f"Trustworthy: {test_attr_accs.get('trustworthy', 0):.4f}")

    print("\nPer-User Test Accuracies:")
    for i, user_id in enumerate(valid_users):
        token = f"<user{i+1}>"
        acc = test_user_accs.get(token, 0)
        count = user_counts[user_id]
        print(f"  {token} ({user_id[:12]}..., {count} samples): {acc:.4f}")

    avg_test_user_acc = np.mean(list(test_user_accs.values()))
    print(f"Average Test User Accuracy: {avg_test_user_acc:.4f}")

if __name__ == "__main__":
    main()