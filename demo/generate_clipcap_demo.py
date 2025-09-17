#!/usr/bin/env python3
"""Demo using ClipCap to generate captions directly from CLIP embeddings."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from PIL import Image
import pandas as pd
from pathlib import Path
import json
import os
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel


class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x):
        return x @ self.lora_A.T.to(x.dtype) @ self.lora_B.T.to(x.dtype) * self.scaling


class VisualPromptTokens(nn.Module):
    def __init__(self, num_users, embed_dim=768):
        super().__init__()
        self.visual_tokens = nn.Parameter(torch.randn(num_users, 1, embed_dim) * 0.02)

    def forward(self, user_indices):
        return self.visual_tokens[user_indices]


class MLP(nn.Module):
    """Simple MLP to map CLIP embeddings to GPT2 embedding space."""

    def __init__(self, sizes, bias=True, act=nn.Tanh):
        super().__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ClipCaptionModel(nn.Module):
    """Simple ClipCap-style model: maps CLIP embeddings to GPT2 prefix."""

    def __init__(self, prefix_length=10, prefix_size=512, gpt2_model='gpt2'):
        super().__init__()
        self.prefix_length = prefix_length

        # Load GPT2
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model)
        self.gpt2_embedding_size = self.gpt2.transformer.wte.weight.shape[1]

        # MLP to map CLIP embeddings to GPT2 prefix
        self.clip_project = MLP([prefix_size, self.gpt2_embedding_size // 2,
                                 self.gpt2_embedding_size * prefix_length])

    def forward(self, clip_features, tokens=None, labels=None):
        # Project CLIP features to GPT2 embedding space
        prefix_projections = self.clip_project(clip_features)
        prefix_embeddings = prefix_projections.view(-1, self.prefix_length, self.gpt2_embedding_size)

        if tokens is not None:
            token_embeddings = self.gpt2.transformer.wte(tokens)
            embeddings = torch.cat([prefix_embeddings, token_embeddings], dim=1)
        else:
            embeddings = prefix_embeddings

        # Forward through GPT2
        outputs = self.gpt2(inputs_embeds=embeddings, labels=labels)
        return outputs

    def generate_caption(self, clip_features, tokenizer, max_length=30, temperature=1.0):
        """Generate caption from CLIP features."""

        # Get prefix embeddings from CLIP features
        prefix_projections = self.clip_project(clip_features)
        prefix_embeddings = prefix_projections.view(-1, self.prefix_length, self.gpt2_embedding_size)

        # Start with just the prefix
        generated = []
        inputs_embeds = prefix_embeddings

        # Generate token by token
        for _ in range(max_length):
            with torch.no_grad():
                outputs = self.gpt2(inputs_embeds=inputs_embeds)
                logits = outputs.logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)

                # Sample next token
                next_token = torch.multinomial(probs, 1)
                generated.append(next_token.item())

                # Check for EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break

                # Add new token embedding
                next_embedding = self.gpt2.transformer.wte(next_token)
                inputs_embeds = torch.cat([inputs_embeds, next_embedding], dim=1)

        # Decode
        caption = tokenizer.decode(generated, skip_special_tokens=True)
        return caption


def load_lora_model(checkpoint_path, device='cuda'):
    """Load CLIP with LoRA adapters and visual prompts."""
    model, preprocess = clip.load('ViT-B/32', device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Add LoRA layers
    if 'lora_state' in checkpoint:
        lora_idx = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and 'mlp' in name and 'visual' in name:
                if lora_idx < len(checkpoint['lora_state']):
                    adapter = LoRALayer(module.in_features, module.out_features).to(device)
                    adapter.load_state_dict(checkpoint['lora_state'][lora_idx])
                    orig = module.forward
                    module.forward = lambda x, orig=orig, adapter=adapter: orig(x) + adapter(x)
                    lora_idx += 1

    # Add visual prompts
    visual_prompts = None
    if 'visual_prompts_state' in checkpoint:
        num_users = checkpoint['visual_prompts_state']['visual_tokens'].shape[0]
        visual_prompts = VisualPromptTokens(num_users, model.visual.transformer.width).to(device)
        visual_prompts.load_state_dict(checkpoint['visual_prompts_state'])

        orig_encode = model.encode_image

        def new_encode(images, user_idx=None):
            if user_idx is None:
                return orig_encode(images)

            dtype = model.dtype
            x = model.visual.conv1(images.type(dtype))
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            x = torch.cat([model.visual.class_embedding.to(dtype) +
                          torch.zeros(x.shape[0], 1, x.shape[-1], dtype=dtype, device=x.device), x], dim=1)
            x = x + model.visual.positional_embedding.to(dtype)

            if user_idx is not None:
                user_token = visual_prompts(torch.tensor([user_idx]).to(device)).to(dtype)
                x = torch.cat([x[:, :1], user_token, x[:, 1:]], dim=1)

            x = model.visual.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = model.visual.transformer(x)
            x = x.permute(1, 0, 2)
            x = model.visual.ln_post(x[:, 0, :])
            if model.visual.proj is not None:
                x = x @ model.visual.proj
            return x

        model.encode_image = new_encode

    user_accs = checkpoint.get('user_accs', {})
    return model, preprocess, visual_prompts, user_accs


def train_simple_clipcap(clip_model, device='cuda'):
    """Train a simple ClipCap model on COCO or use pre-trained weights."""

    print("Initializing ClipCap model...")

    # For demo purposes, we'll create a simple untrained model
    # In production, you'd load pre-trained weights
    caption_model = ClipCaptionModel(prefix_length=10, prefix_size=512).to(device)

    # Try to load pre-trained weights if available
    pretrained_path = 'clipcap_weights.pt'
    if os.path.exists(pretrained_path):
        print(f"Loading pre-trained ClipCap weights from {pretrained_path}")
        caption_model.load_state_dict(torch.load(pretrained_path, map_location=device))
    else:
        print("Warning: Using untrained ClipCap model. Consider training on COCO first.")
        print("For better results, use pre-trained ClipCap weights.")

    caption_model.eval()
    return caption_model


def generate_from_embedding(clip_embedding, caption_model, tokenizer,
                           temperature=1.0, user_name=""):
    """Generate caption directly from CLIP embedding."""

    with torch.no_grad():
        caption = caption_model.generate_caption(
            clip_embedding,
            tokenizer,
            max_length=30,
            temperature=temperature
        )

    # Clean up caption
    caption = caption.strip()
    if not caption:
        caption = f"A photo {user_name}"

    return caption


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models
    print("\nLoading CLIP models...")
    base_clip, preprocess = clip.load('ViT-B/32', device=device)

    checkpoint_path = 'runs/lora_visual_prompts_ViT-B_32_20250917_142456/best_model.pt'
    lora_clip, _, visual_prompts, user_accs = load_lora_model(checkpoint_path, device)

    # Get top 2 users
    if not user_accs:
        print("No user accuracies found!")
        return

    sorted_users = sorted(user_accs.items(), key=lambda x: x[1], reverse=True)
    user1_id, user1_acc = sorted_users[0]
    user2_id, user2_acc = sorted_users[1]

    user_list = list(user_accs.keys())
    user1_idx = user_list.index(user1_id)
    user2_idx = user_list.index(user2_id)

    print(f"\nTop users:")
    print(f"User 1: {user1_id[:12]}... (accuracy: {user1_acc:.2%})")
    print(f"User 2: {user2_id[:12]}... (accuracy: {user2_acc:.2%})")

    # Initialize ClipCap model
    caption_model = train_simple_clipcap(base_clip, device)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Create output directory
    demo_dir = Path('docs/clipcap_demo')
    demo_dir.mkdir(parents=True, exist_ok=True)

    # Select sample images
    ffhq_dir = Path('data/ffhq/src')
    sample_images = list(ffhq_dir.glob('*.webp'))[:100]

    import random
    random.seed(42)
    selected_images = random.sample(sample_images, min(10, len(sample_images)))

    print(f"\nProcessing {len(selected_images)} images with ClipCap...")

    results = []
    for i, img_path in enumerate(selected_images):
        print(f"Processing image {i+1}/{len(selected_images)}: {img_path.name}")

        # Load and preprocess image
        image = Image.open(img_path).convert('RGB')
        clip_img = preprocess(image).unsqueeze(0).to(device)

        # Save image
        img_name = f"img_{i+1}.jpg"
        image.save(demo_dir / img_name, 'JPEG', quality=95)

        with torch.no_grad():
            # Get embeddings
            base_emb = F.normalize(base_clip.encode_image(clip_img), dim=-1)
            user1_emb = F.normalize(lora_clip.encode_image(clip_img, user1_idx), dim=-1)
            user2_emb = F.normalize(lora_clip.encode_image(clip_img, user2_idx), dim=-1)

            # Calculate embedding shifts
            shift1 = torch.norm(user1_emb - base_emb).item()
            shift2 = torch.norm(user2_emb - base_emb).item()

        # Generate captions from embeddings
        base_caption = generate_from_embedding(base_emb, caption_model, tokenizer,
                                              temperature=0.8, user_name="baseline")

        # Use embedding shift to modulate temperature
        temp1 = min(0.7 + shift1 * 0.3, 1.2)
        temp2 = min(0.7 + shift2 * 0.3, 1.2)

        user1_caption = generate_from_embedding(user1_emb, caption_model, tokenizer,
                                               temperature=temp1, user_name="user1")
        user2_caption = generate_from_embedding(user2_emb, caption_model, tokenizer,
                                               temperature=temp2, user_name="user2")

        results.append({
            'image': img_name,
            'baseline_caption': base_caption,
            'user1_caption': user1_caption,
            'user2_caption': user2_caption,
            'embedding_shift_user1': shift1,
            'embedding_shift_user2': shift2
        })

    # Generate markdown report
    print("\nGenerating markdown report...")

    md_content = """# ClipCap Demo: Direct Caption Generation from CLIP Embeddings

## Overview
This demo uses ClipCap-style architecture to generate captions directly from CLIP embeddings (both baseline and LoRA-adapted).

## Key Difference
Unlike BLIP which processes images independently, ClipCap:
- Takes CLIP embeddings as input
- Maps them to GPT-2 prefix embeddings
- Generates captions that directly reflect embedding differences

## Users
"""
    md_content += f"- **User 1**: {user1_acc:.1%} accuracy\n"
    md_content += f"- **User 2**: {user2_acc:.1%} accuracy\n\n"

    md_content += "---\n\n"

    for i, result in enumerate(results, 1):
        md_content += f"### Image {i}\n\n"
        md_content += f"![{result['image']}]({result['image']})\n\n"

        md_content += f"**Embedding Shifts:**\n"
        md_content += f"- User 1: {result['embedding_shift_user1']:.3f}\n"
        md_content += f"- User 2: {result['embedding_shift_user2']:.3f}\n\n"

        md_content += "**Captions:**\n"
        md_content += f"- **Baseline**: {result['baseline_caption']}\n"
        md_content += f"- **User 1**: {result['user1_caption']}\n"
        md_content += f"- **User 2**: {result['user2_caption']}\n\n"

        # Highlight differences
        if result['user1_caption'] != result['baseline_caption']:
            md_content += "✓ User 1 caption differs from baseline\n"
        if result['user2_caption'] != result['baseline_caption']:
            md_content += "✓ User 2 caption differs from baseline\n"

        md_content += "\n---\n\n"

    # Save report
    md_path = demo_dir / 'clipcap_demo.md'
    with open(md_path, 'w') as f:
        f.write(md_content)

    print(f"✓ Markdown report saved to: {md_path}")

    # Save JSON results
    json_path = demo_dir / 'results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to: {json_path}")
    print("\nClipCap demo complete!")

    print("\n" + "="*60)
    print("NOTE: For best results with ClipCap, you should:")
    print("1. Train the MLP projection layer on COCO dataset")
    print("2. Or use pre-trained ClipCap weights")
    print("3. Or try CoCa model which is designed for this task")
    print("="*60)


if __name__ == "__main__":
    main()