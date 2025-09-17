#!/usr/bin/env python3
"""Generate captions that reflect CLIP embedding differences using template-based generation."""

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
from sklearn.metrics.pairwise import cosine_similarity


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


def get_attribute_scores(clip_model, image_embedding, device='cuda'):
    """Get similarity scores for different attributes using CLIP text encoder."""

    # Define attribute prompts
    attribute_prompts = {
        'attractive': [
            "a photo of a very attractive person",
            "a photo of a beautiful person",
            "a photo of an unattractive person"
        ],
        'smart': [
            "a photo of an intelligent looking person",
            "a photo of a smart professional",
            "a photo of a simple person"
        ],
        'trustworthy': [
            "a photo of a trustworthy person",
            "a photo of an honest looking person",
            "a photo of an untrustworthy person"
        ],
        'emotion': [
            "a photo of a happy smiling person",
            "a photo of a serious person",
            "a photo of a sad person"
        ],
        'age': [
            "a photo of a young person",
            "a photo of a middle-aged person",
            "a photo of an elderly person"
        ]
    }

    scores = {}
    for attr, prompts in attribute_prompts.items():
        # Encode text prompts
        text_tokens = clip.tokenize(prompts).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)

        # Calculate similarities
        similarities = (image_embedding @ text_features.T).squeeze().cpu().numpy()
        scores[attr] = similarities

    return scores


def generate_embedding_aware_caption(clip_model, base_embedding, user_embedding,
                                    attribute_scores_base, attribute_scores_user, user_name=""):
    """Generate caption based on embedding differences and attribute scores."""

    # Calculate embedding shift
    shift = torch.norm(user_embedding - base_embedding).item()

    # Find which attributes changed most
    attr_changes = {}
    for attr in attribute_scores_base:
        base_score = attribute_scores_base[attr][0]  # First prompt is positive
        user_score = attribute_scores_user[attr][0]
        change = user_score - base_score
        attr_changes[attr] = change

    # Build caption based on strongest signals
    caption_parts = []

    # Determine primary descriptor
    if attr_changes['age'] > 0.1:
        caption_parts.append("a young")
    elif attr_changes['age'] < -0.1:
        caption_parts.append("an older")
    else:
        caption_parts.append("a")

    # Add emotional state if strong signal
    if attribute_scores_user['emotion'][0] > 0.3:
        caption_parts.append("happy")
    elif attribute_scores_user['emotion'][1] > 0.3:
        caption_parts.append("serious")

    # Add trustworthiness if it's a strong signal for this user
    if attr_changes['trustworthy'] > 0.15:
        caption_parts.append("trustworthy looking")
    elif attr_changes['trustworthy'] < -0.15:
        caption_parts.append("mysterious")

    # Add attractiveness if strong signal
    if attr_changes['attractive'] > 0.2:
        caption_parts.append("attractive")

    # Add intelligence if relevant
    if attr_changes['smart'] > 0.15:
        caption_parts.append("professional")

    caption_parts.append("person")

    # Add context based on embedding shift magnitude
    if shift > 1.2:
        caption_parts.append("with distinctive features")
    elif shift > 0.8:
        caption_parts.append("in the photo")

    caption = " ".join(caption_parts)

    # Clean up grammar
    caption = caption.replace("a attractive", "an attractive")
    caption = caption.replace("a older", "an older")

    return caption, attr_changes


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
    disagreement_pairs = find_disagreement_pairs([user1_id, user2_id], max_pairs=5)

    if not disagreement_pairs:
        print("No disagreement pairs found!")
        return

    # Create output directory
    demo_dir = Path('docs/embedding_aware_demo')
    demo_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {len(disagreement_pairs)} pairs with embedding-aware captions...")
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
        for img_path, img_key in [(im1_path, 'img1'), (im2_path, 'img2')]:
            image = Image.open(img_path).convert('RGB')
            clip_img = preprocess(image).unsqueeze(0).to(device)

            with torch.no_grad():
                # Get embeddings
                base_emb = F.normalize(base_clip.encode_image(clip_img), dim=-1)
                user1_emb = F.normalize(lora_clip.encode_image(clip_img, user1_idx), dim=-1)
                user2_emb = F.normalize(lora_clip.encode_image(clip_img, user2_idx), dim=-1)

                # Get attribute scores
                attr_base = get_attribute_scores(base_clip, base_emb, device)
                attr_user1 = get_attribute_scores(lora_clip, user1_emb, device)
                attr_user2 = get_attribute_scores(lora_clip, user2_emb, device)

            # Generate captions based on embeddings
            base_caption, _ = generate_embedding_aware_caption(
                base_clip, base_emb, base_emb, attr_base, attr_base, "baseline"
            )

            user1_caption, changes1 = generate_embedding_aware_caption(
                lora_clip, base_emb, user1_emb, attr_base, attr_user1, "user1"
            )

            user2_caption, changes2 = generate_embedding_aware_caption(
                lora_clip, base_emb, user2_emb, attr_base, attr_user2, "user2"
            )

            captions[img_key] = {
                'baseline': base_caption,
                'user1': user1_caption,
                'user2': user2_caption,
                'attr_changes_user1': changes1,
                'attr_changes_user2': changes2
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

    md_content = f"""# Embedding-Aware Caption Generation Demo

## Overview
This demo generates captions that directly reflect the differences in CLIP embeddings between baseline and user-personalized models.

## Method
1. Calculate attribute similarity scores using CLIP text encoder
2. Compare how these scores change with personalized embeddings
3. Generate captions that emphasize the attributes that changed most

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

        # Embedding-aware captions
        md_content += "### Embedding-Aware Captions\n\n"
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

        # Show attribute changes
        md_content += "### Attribute Score Changes (vs baseline)\n\n"
        md_content += "**Image 1:**\n"
        for attr in ['attractive', 'smart', 'trustworthy']:
            change1 = result['captions']['img1']['attr_changes_user1'].get(attr, 0)
            change2 = result['captions']['img1']['attr_changes_user2'].get(attr, 0)
            md_content += f"- {attr.title()}: User1={change1:+.2f}, User2={change2:+.2f}\n"

        md_content += "\n**Image 2:**\n"
        for attr in ['attractive', 'smart', 'trustworthy']:
            change1 = result['captions']['img2']['attr_changes_user1'].get(attr, 0)
            change2 = result['captions']['img2']['attr_changes_user2'].get(attr, 0)
            md_content += f"- {attr.title()}: User1={change1:+.2f}, User2={change2:+.2f}\n"

        md_content += "\n---\n\n"

    md_content += """## Summary

This approach generates captions that:
1. Directly reflect the CLIP embedding differences
2. Emphasize attributes that changed most for each user
3. Show personalized perception through caption variation

The captions now vary based on how each user's model perceives different attributes in the images.
"""

    # Save report
    md_path = demo_dir / 'embedding_aware_demo.md'
    with open(md_path, 'w') as f:
        f.write(md_content)

    print(f"✓ Report saved to: {md_path}")

    # Save JSON
    json_path = demo_dir / 'results.json'

    # Clean up attribute changes for JSON serialization
    for result in results:
        for img_key in ['img1', 'img2']:
            for user in ['attr_changes_user1', 'attr_changes_user2']:
                result['captions'][img_key][user] = {
                    k: float(v) for k, v in result['captions'][img_key][user].items()
                }

    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to: {json_path}")
    print("\nDemo complete!")


if __name__ == "__main__":
    main()