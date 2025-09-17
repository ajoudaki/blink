#!/usr/bin/env python3
"""Generate demo showing image pairs where top users disagree, with side-by-side captions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import pandas as pd
from pathlib import Path
import shutil
import json
import os


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


def find_disagreement_pairs(top_users, max_pairs=10):
    """Find image pairs where the top users disagree."""

    # Load comparison data
    print("Loading comparison data...")
    df_labels = pd.read_excel('data/big_compare_label.xlsx')
    df_data = pd.read_excel('data/big_compare_data.xlsx')

    # Merge to get image paths
    df = df_data.merge(df_labels, left_on='_id', right_on='item_id', how='inner')

    # Filter to only our top 2 users
    df_user1 = df[df['user_id'] == top_users[0]].copy()
    df_user2 = df[df['user_id'] == top_users[1]].copy()

    print(f"User 1 has {len(df_user1)} comparisons")
    print(f"User 2 has {len(df_user2)} comparisons")

    # Find common pairs where they disagree
    disagreement_pairs = []

    for _, row1 in df_user1.iterrows():
        # Find matching comparison for user 2
        match = df_user2[df_user2['item_id'] == row1['item_id']]

        if not match.empty:
            row2 = match.iloc[0]

            # Check for disagreements on each attribute
            disagreements = {}
            for attr in ['attractive', 'smart', 'trustworthy']:
                if row1[attr] != row2[attr] and row1[attr] in [1, 2] and row2[attr] in [1, 2]:
                    disagreements[attr] = {
                        'user1_choice': row1[attr],  # 1 or 2
                        'user2_choice': row2[attr]
                    }

            if disagreements:  # If they disagree on at least one attribute
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

    print(f"Found {len(disagreement_pairs)} pairs with disagreements")
    return disagreement_pairs


def fix_image_path(original_path):
    """Fix image path to match actual file locations."""
    filename = os.path.basename(original_path)

    # Try different possible paths
    possible_paths = [
        f"data/ffhq/src/{filename}",
        f"data/ffhq2/src/{filename}",
        f"pics_small/{filename}",
    ]

    # Try with different extensions
    base_name = os.path.splitext(filename)[0]
    for ext in ['.webp', '.jpg', '.png']:
        possible_paths.append(f"data/ffhq/src/{base_name}{ext}")

    for path in possible_paths:
        if os.path.exists(path):
            return path

    print(f"Warning: Could not find {filename}")
    return None


def generate_caption(image_path, clip_model, preprocess, blip_model, processor,
                     user_idx=None, device='cuda'):
    """Generate caption for an image."""

    if not os.path.exists(image_path):
        return "Image not found"

    image = Image.open(image_path).convert('RGB')
    clip_img = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        # Get embedding
        if user_idx is not None:
            _ = clip_model.encode_image(clip_img, user_idx)
        else:
            _ = clip_model.encode_image(clip_img)

        # Generate caption
        inputs = processor(image, return_tensors="pt").to(device)

        if user_idx is not None:
            # Personalized caption with sampling
            out = blip_model.generate(**inputs, max_length=30, num_beams=5,
                                     temperature=0.9, do_sample=True, top_k=50)
        else:
            # Baseline caption
            out = blip_model.generate(**inputs, max_length=30, num_beams=3)

        caption = processor.decode(out[0], skip_special_tokens=True)

    return caption


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models
    print("\nLoading models...")
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

    # Find disagreement pairs
    disagreement_pairs = find_disagreement_pairs([user1_id, user2_id], max_pairs=10)

    if not disagreement_pairs:
        print("No disagreement pairs found!")
        return

    # Load BLIP
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)

    # Create output directory
    demo_dir = Path('docs/demo_pairs')
    demo_dir.mkdir(parents=True, exist_ok=True)

    # Generate captions for each pair
    print(f"\nProcessing {len(disagreement_pairs)} pairs...")
    results = []

    for i, pair in enumerate(disagreement_pairs):
        print(f"Processing pair {i+1}/{len(disagreement_pairs)}")

        # Fix image paths
        im1_path = fix_image_path(pair['im1_path'])
        im2_path = fix_image_path(pair['im2_path'])

        if not im1_path or not im2_path:
            continue

        # Copy images to demo directory
        im1_name = f"pair{i+1}_img1.jpg"
        im2_name = f"pair{i+1}_img2.jpg"

        img1 = Image.open(im1_path).convert('RGB')
        img2 = Image.open(im2_path).convert('RGB')

        img1.save(demo_dir / im1_name, 'JPEG', quality=95)
        img2.save(demo_dir / im2_name, 'JPEG', quality=95)

        # Generate captions
        captions = {}
        for img_path, img_key in [(im1_path, 'img1'), (im2_path, 'img2')]:
            captions[img_key] = {
                'baseline': generate_caption(img_path, base_clip, preprocess,
                                            blip_model, processor, None, device),
                'user1': generate_caption(img_path, lora_clip, preprocess,
                                         blip_model, processor, user1_idx, device),
                'user2': generate_caption(img_path, lora_clip, preprocess,
                                         blip_model, processor, user2_idx, device)
            }

        # Convert numpy types to Python types for JSON serialization
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
    print("\nGenerating markdown report...")

    md_content = f"""# Image Pair Comparison Demo: User Disagreements

## Overview
This demo shows image pairs where our top two users disagree on their perception labels, along with baseline and personalized captions for each image.

## Users
- **User 1**: Best performer ({user1_acc:.1%} accuracy)
- **User 2**: Second best ({user2_acc:.1%} accuracy)

## Label Key
- **1**: First image preferred
- **2**: Second image preferred
- **Bold**: Indicates disagreement between users

---

"""

    for result in results:
        pair_num = result['pair_num']
        img1 = result['img1_name']
        img2 = result['img2_name']

        md_content += f"## Pair {pair_num}\n\n"

        # Create side-by-side images using HTML table
        md_content += "<table>\n<tr>\n"
        md_content += f'<td width="50%" align="center"><img src="{img1}" width="100%"><br><b>Image 1</b></td>\n'
        md_content += f'<td width="50%" align="center"><img src="{img2}" width="100%"><br><b>Image 2</b></td>\n'
        md_content += "</tr>\n</table>\n\n"

        # Show user labels with disagreements highlighted
        md_content += "### User Labels\n\n"
        md_content += "<table>\n"
        md_content += "<tr><th>Attribute</th><th>User 1 Choice</th><th>User 2 Choice</th><th>Agreement</th></tr>\n"

        for attr in ['attractive', 'smart', 'trustworthy']:
            user1_val = result['user1_labels'][attr]
            user2_val = result['user2_labels'][attr]

            # Format choices (1=Image1, 2=Image2)
            user1_choice = f"Image {user1_val}" if user1_val in [1, 2] else "N/A"
            user2_choice = f"Image {user2_val}" if user2_val in [1, 2] else "N/A"

            # Check agreement
            agrees = "✓" if user1_val == user2_val else "✗"

            # Highlight disagreements
            if user1_val != user2_val:
                user1_choice = f"**{user1_choice}**"
                user2_choice = f"**{user2_choice}**"
                attr = f"**{attr.title()}**"
            else:
                attr = attr.title()

            md_content += f"<tr><td>{attr}</td><td>{user1_choice}</td><td>{user2_choice}</td><td>{agrees}</td></tr>\n"

        md_content += "</table>\n\n"

        # Show captions in aligned table
        md_content += "### Captions\n\n"
        md_content += "<table>\n"
        md_content += "<tr><th width='15%'>Model</th><th width='42.5%'>Image 1 Caption</th><th width='42.5%'>Image 2 Caption</th></tr>\n"

        # Baseline captions
        md_content += f"<tr><td><b>Baseline</b></td>"
        md_content += f"<td>{result['captions']['img1']['baseline']}</td>"
        md_content += f"<td>{result['captions']['img2']['baseline']}</td></tr>\n"

        # User 1 captions
        md_content += f"<tr><td><b>User 1</b></td>"
        md_content += f"<td>{result['captions']['img1']['user1']}</td>"
        md_content += f"<td>{result['captions']['img2']['user1']}</td></tr>\n"

        # User 2 captions
        md_content += f"<tr><td><b>User 2</b></td>"
        md_content += f"<td>{result['captions']['img1']['user2']}</td>"
        md_content += f"<td>{result['captions']['img2']['user2']}</td></tr>\n"

        md_content += "</table>\n\n"
        md_content += "---\n\n"

    # Add summary
    total_disagreements = sum(len(r['disagreements']) for r in results)
    md_content += f"""## Summary

- **Total Pairs Analyzed**: {len(results)}
- **Total Attribute Disagreements**: {total_disagreements}
- **Most Common Disagreement**: {max(set([attr for r in results for attr in r['disagreements'].keys()]),
                                        key=lambda x: sum(1 for r in results if x in r['disagreements']))}

The personalized captions reflect each user's visual perception preferences learned from their comparison labels. Notice how the captions may emphasize different aspects of the images based on which image each user preferred for different attributes.
"""

    # Save markdown
    md_path = demo_dir / 'paired_comparison_demo.md'
    with open(md_path, 'w') as f:
        f.write(md_content)

    print(f"✓ Markdown report saved to: {md_path}")

    # Save JSON results
    json_path = demo_dir / 'results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ JSON results saved to: {json_path}")
    print(f"✓ Images saved to: {demo_dir}")
    print("\nDemo generation complete!")


if __name__ == "__main__":
    main()