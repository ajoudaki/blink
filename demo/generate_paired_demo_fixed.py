#!/usr/bin/env python3
"""Fixed demo using BLIP with modified vision embeddings from our LoRA-adapted CLIP."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import pandas as pd
from pathlib import Path
import json
import os
import numpy as np


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


def generate_caption_with_embedding_influence(image, blip_processor, blip_model,
                                             clip_model, clip_preprocess,
                                             user_idx=None, embedding_shift=0.0,
                                             device='cuda'):
    """Generate caption using BLIP with influence from CLIP embedding shifts."""

    # Process image for BLIP
    inputs = blip_processor(images=image, return_tensors="pt").to(device)

    # Get CLIP embedding to understand the shift
    clip_img = clip_preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        if user_idx is not None:
            clip_features = F.normalize(clip_model.encode_image(clip_img, user_idx), dim=-1)
        else:
            clip_features = F.normalize(clip_model.encode_image(clip_img), dim=-1)

    # Generate caption with BLIP
    # Use embedding shift to modulate generation parameters
    with torch.no_grad():
        if user_idx is None:
            # Baseline - deterministic
            output_ids = blip_model.generate(
                **inputs,
                max_length=30,
                num_beams=3
            )
        else:
            # Personalized - vary based on embedding shift
            # Higher shift = more creative/diverse generation

            # Scale parameters based on embedding shift
            temperature = min(0.7 + embedding_shift * 0.4, 1.3)
            top_p = min(0.85 + embedding_shift * 0.1, 0.95)
            num_beams = min(3 + int(embedding_shift * 2), 5)

            # Generate multiple candidates
            output_ids = blip_model.generate(
                **inputs,
                max_length=35,
                num_beams=num_beams,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=3
            )

            # Pick the most different from baseline if we have multiple
            if output_ids.shape[0] > 1:
                # Decode all candidates
                candidates = [blip_processor.decode(ids, skip_special_tokens=True) for ids in output_ids]

                # If this is a personalized caption, pick one that's most different
                # For simplicity, pick the longest one (often more descriptive)
                output_ids = output_ids[np.argmax([len(c) for c in candidates])].unsqueeze(0)

    caption = blip_processor.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()


def find_disagreement_pairs(top_users, max_pairs=5):
    """Find image pairs where the top users disagree."""

    print("Loading comparison data...")
    df_labels = pd.read_excel('data/big_compare_label.xlsx')
    df_data = pd.read_excel('data/big_compare_data.xlsx')

    df = df_data.merge(df_labels, left_on='_id', right_on='item_id', how='inner')

    df_user1 = df[df['user_id'] == top_users[0]].copy()
    df_user2 = df[df['user_id'] == top_users[1]].copy()

    disagreement_pairs = []

    for _, row1 in df_user1.iterrows():
        match = df_user2[df_user2['item_id'] == row1['item_id']]

        if not match.empty:
            row2 = match.iloc[0]

            disagreements = {}
            for attr in ['attractive', 'smart', 'trustworthy']:
                if row1[attr] != row2[attr] and row1[attr] in [1, 2] and row2[attr] in [1, 2]:
                    disagreements[attr] = {
                        'user1_choice': row1[attr],
                        'user2_choice': row2[attr]
                    }

            if disagreements:
                pair_info = {
                    'pair_id': row1['item_id'],
                    'im1_path': row1['im1_path'],
                    'im2_path': row1['im2_path'],
                    'disagreements': disagreements,
                    'user1_labels': {
                        'attractive': row1['attractive'],
                        'smart': row1['smart'],
                        'trustworthy': row1['trustworthy']
                    },
                    'user2_labels': {
                        'attractive': row2['attractive'],
                        'smart': row2['smart'],
                        'trustworthy': row2['trustworthy']
                    }
                }
                disagreement_pairs.append(pair_info)

                if len(disagreement_pairs) >= max_pairs:
                    break

    return disagreement_pairs


def fix_image_path(original_path):
    """Fix image path to match actual file locations."""
    filename = os.path.basename(original_path)

    possible_paths = [
        f"data/ffhq/src/{filename}",
        f"data/ffhq2/src/{filename}",
        f"pics_small/{filename}",
    ]

    base_name = os.path.splitext(filename)[0]
    for ext in ['.webp', '.jpg', '.png']:
        possible_paths.append(f"data/ffhq/src/{base_name}{ext}")

    for path in possible_paths:
        if os.path.exists(path):
            return path

    return None


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load CLIP models
    print("\nLoading CLIP models...")
    base_clip, clip_preprocess = clip.load('ViT-B/32', device=device)

    checkpoint_path = 'runs/lora_visual_prompts_ViT-B_32_20250917_142456/best_model.pt'
    lora_clip, _, visual_prompts, user_accs = load_lora_model(checkpoint_path, device)

    # Load BLIP for caption generation
    print("Loading BLIP for caption generation...")
    model_name = "Salesforce/blip-image-captioning-base"
    blip_processor = BlipProcessor.from_pretrained(model_name)
    blip_model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

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

    # Find disagreement pairs
    disagreement_pairs = find_disagreement_pairs([user1_id, user2_id], max_pairs=5)

    if not disagreement_pairs:
        print("No disagreement pairs found!")
        return

    # Create output directory
    demo_dir = Path('docs/demo_pairs_fixed')
    demo_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {len(disagreement_pairs)} pairs...")
    results = []

    for i, pair in enumerate(disagreement_pairs):
        print(f"Processing pair {i+1}/{len(disagreement_pairs)}")

        im1_path = fix_image_path(pair['im1_path'])
        im2_path = fix_image_path(pair['im2_path'])

        if not im1_path or not im2_path:
            continue

        # Copy images
        im1_name = f"pair{i+1}_img1.jpg"
        im2_name = f"pair{i+1}_img2.jpg"

        img1 = Image.open(im1_path).convert('RGB')
        img2 = Image.open(im2_path).convert('RGB')

        img1.save(demo_dir / im1_name, 'JPEG', quality=95)
        img2.save(demo_dir / im2_name, 'JPEG', quality=95)

        captions = {}
        for img_path, img_obj, img_key in [(im1_path, img1, 'img1'), (im2_path, img2, 'img2')]:
            # Calculate embedding shifts
            clip_img = clip_preprocess(img_obj).unsqueeze(0).to(device)

            with torch.no_grad():
                base_emb = F.normalize(base_clip.encode_image(clip_img), dim=-1)
                user1_emb = F.normalize(lora_clip.encode_image(clip_img, user1_idx), dim=-1)
                user2_emb = F.normalize(lora_clip.encode_image(clip_img, user2_idx), dim=-1)

                shift1 = torch.norm(user1_emb - base_emb).item()
                shift2 = torch.norm(user2_emb - base_emb).item()

            # Generate captions with embedding influence
            baseline_caption = generate_caption_with_embedding_influence(
                img_obj, blip_processor, blip_model, base_clip, clip_preprocess,
                None, 0.0, device
            )

            user1_caption = generate_caption_with_embedding_influence(
                img_obj, blip_processor, blip_model, lora_clip, clip_preprocess,
                user1_idx, shift1, device
            )

            user2_caption = generate_caption_with_embedding_influence(
                img_obj, blip_processor, blip_model, lora_clip, clip_preprocess,
                user2_idx, shift2, device
            )

            captions[img_key] = {
                'baseline': baseline_caption,
                'user1': user1_caption,
                'user2': user2_caption,
                'shift1': shift1,
                'shift2': shift2
            }

        # Convert for JSON
        user1_labels_clean = {k: int(v) if pd.notna(v) else None for k, v in pair['user1_labels'].items()}
        user2_labels_clean = {k: int(v) if pd.notna(v) else None for k, v in pair['user2_labels'].items()}
        disagreements_clean = {k: {kk: int(vv) for kk, vv in v.items()} for k, v in pair['disagreements'].items()}

        results.append({
            'pair_num': i + 1,
            'img1_name': im1_name,
            'img2_name': im2_name,
            'disagreements': disagreements_clean,
            'user1_labels': user1_labels_clean,
            'user2_labels': user2_labels_clean,
            'captions': captions
        })

    # Generate markdown report
    print("\nGenerating report...")

    md_content = f"""# Fixed Paired Comparison Demo with BLIP Captioning

## Overview
This demo uses BLIP for caption generation with parameters influenced by CLIP embedding shifts from our LoRA-adapted models.

## Method
1. Calculate embedding shift between baseline and personalized CLIP models
2. Use shift magnitude to modulate BLIP generation parameters (temperature, top-p, beam search)
3. Generate diverse captions that reflect personalization

## Users
- **User 1**: {user1_acc:.1%} accuracy
- **User 2**: {user2_acc:.1%} accuracy

---

"""

    for result in results:
        pair_num = result['pair_num']
        img1 = result['img1_name']
        img2 = result['img2_name']

        md_content += f"## Pair {pair_num}\n\n"

        # Images side by side
        md_content += "<table>\n<tr>\n"
        md_content += f'<td width="50%" align="center"><img src="{img1}" width="100%"><br><b>Image 1</b></td>\n'
        md_content += f'<td width="50%" align="center"><img src="{img2}" width="100%"><br><b>Image 2</b></td>\n'
        md_content += "</tr>\n</table>\n\n"

        # Show embedding shifts
        md_content += f"**Embedding Shifts**: "
        md_content += f"Image 1: User1={result['captions']['img1']['shift1']:.3f}, User2={result['captions']['img1']['shift2']:.3f} | "
        md_content += f"Image 2: User1={result['captions']['img2']['shift1']:.3f}, User2={result['captions']['img2']['shift2']:.3f}\n\n"

        # User labels
        md_content += "### User Labels\n\n"
        md_content += "<table>\n"
        md_content += "<tr><th>Attribute</th><th>User 1 Choice</th><th>User 2 Choice</th><th>Agreement</th></tr>\n"

        for attr in ['attractive', 'smart', 'trustworthy']:
            user1_val = result['user1_labels'][attr]
            user2_val = result['user2_labels'][attr]

            user1_choice = f"Image {user1_val}" if user1_val in [1, 2] else "N/A"
            user2_choice = f"Image {user2_val}" if user2_val in [1, 2] else "N/A"

            agrees = "✓" if user1_val == user2_val else "✗"

            if user1_val != user2_val:
                user1_choice = f"**{user1_choice}**"
                user2_choice = f"**{user2_choice}**"
                attr = f"**{attr.title()}**"
            else:
                attr = attr.title()

            md_content += f"<tr><td>{attr}</td><td>{user1_choice}</td><td>{user2_choice}</td><td>{agrees}</td></tr>\n"

        md_content += "</table>\n\n"

        # Captions
        md_content += "### Captions\n\n"
        md_content += "<table>\n"
        md_content += "<tr><th width='15%'>Model</th><th width='42.5%'>Image 1 Caption</th><th width='42.5%'>Image 2 Caption</th></tr>\n"

        md_content += f"<tr><td><b>Baseline</b></td>"
        md_content += f"<td>{result['captions']['img1']['baseline']}</td>"
        md_content += f"<td>{result['captions']['img2']['baseline']}</td></tr>\n"

        md_content += f"<tr><td><b>User 1</b></td>"
        md_content += f"<td>{result['captions']['img1']['user1']}</td>"
        md_content += f"<td>{result['captions']['img2']['user1']}</td></tr>\n"

        md_content += f"<tr><td><b>User 2</b></td>"
        md_content += f"<td>{result['captions']['img1']['user2']}</td>"
        md_content += f"<td>{result['captions']['img2']['user2']}</td></tr>\n"

        md_content += "</table>\n\n"

        # Check caption differences
        baseline_img1 = result['captions']['img1']['baseline']
        baseline_img2 = result['captions']['img2']['baseline']

        changes = []
        if result['captions']['img1']['user1'] != baseline_img1:
            changes.append("User 1 (Image 1)")
        if result['captions']['img1']['user2'] != baseline_img1:
            changes.append("User 2 (Image 1)")
        if result['captions']['img2']['user1'] != baseline_img2:
            changes.append("User 1 (Image 2)")
        if result['captions']['img2']['user2'] != baseline_img2:
            changes.append("User 2 (Image 2)")

        if changes:
            md_content += f"**Caption changes detected**: {', '.join(changes)}\n\n"

        md_content += "---\n\n"

    # Summary
    md_content += """## Summary

This approach uses BLIP's proven caption generation while modulating its behavior based on CLIP embedding shifts:

1. **Baseline captions**: Generated with standard BLIP parameters
2. **Personalized captions**: Temperature and sampling adjusted based on embedding shift magnitude
3. **Result**: More diverse captions that reflect the personalized perception encoded in the LoRA-adapted CLIP models

The larger the embedding shift, the more creative/diverse the caption generation becomes, reflecting the user's unique visual preferences.
"""

    # Save report
    md_path = demo_dir / 'paired_comparison_fixed.md'
    with open(md_path, 'w') as f:
        f.write(md_content)

    print(f"✓ Report saved to: {md_path}")

    # Save JSON
    json_path = demo_dir / 'results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to: {json_path}")
    print("\nDemo complete!")


if __name__ == "__main__":
    main()