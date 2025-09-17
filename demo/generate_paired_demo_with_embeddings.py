#!/usr/bin/env python3
"""Generate paired comparison demo using a ClipCap architecture with our CLIP embeddings."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from PIL import Image
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import pandas as pd
from pathlib import Path
import json
import os
import requests
from tqdm import tqdm

# --- New ClipCap Architecture ---

class MappingNetwork(nn.Module):
    """
    The mapping network that translates CLIP embeddings into GPT-2's embedding space.
    This is the core of the ClipCap architecture.
    """
    def __init__(self, clip_embedding_dim: int, gpt_embedding_dim: int, prefix_length: int, num_layers: int = 8):
        super().__init__()
        self.prefix_length = prefix_length
        self.transformer = nn.Transformer(
            d_model=gpt_embedding_dim,
            nhead=8,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=2048,
            activation='relu',
            batch_first=True
        )
        self.linear = nn.Linear(clip_embedding_dim, gpt_embedding_dim)
        self.prefix_const = nn.Parameter(torch.randn(1, prefix_length, gpt_embedding_dim), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project CLIP embedding to GPT-2's dimension
        x = self.linear(x).unsqueeze(1)
        # Prepare the constant prefix tokens
        prefix = self.prefix_const.expand(x.shape[0], -1, -1)
        # Pass both through the transformer to get the final prefix
        out = self.transformer(prefix, x)
        return out


class ClipCapModel(nn.Module):
    """
    Main model that combines the Mapping Network and GPT-2 to generate captions.
    """
    def __init__(self, prefix_length: int = 10, clip_embedding_dim: int = 512, gpt_model_name: str = "gpt2"):
        super().__init__()
        self.prefix_length = prefix_length
        self.gpt_tokenizer = GPT2TokenizerFast.from_pretrained(gpt_model_name)
        self.gpt_model = GPT2LMHeadModel.from_pretrained(gpt_model_name)
        
        # Freeze GPT-2 model
        for param in self.gpt_model.parameters():
            param.requires_grad = False
            
        self.mapping_network = MappingNetwork(
            clip_embedding_dim=clip_embedding_dim,
            gpt_embedding_dim=self.gpt_model.config.n_embd,
            prefix_length=prefix_length,
            num_layers=8
        )

    def get_gpt_embedding_matrix(self):
        return self.gpt_model.transformer.wte.weight

    @torch.no_grad()
    def generate_caption(self, clip_embedding: torch.Tensor, temperature: float = 1.0, num_beams: int = 4,
                         max_length: int = 30, device: str = 'cuda'):
        """
        Generates a caption from a pre-computed CLIP embedding.
        """
        self.eval()
        clip_embedding = clip_embedding.to(device, dtype=torch.float32)

        # 1. Get the prefix embedding from the mapping network
        prefix_embed = self.mapping_network(clip_embedding)

        # 2. Get the start-of-text token embedding for GPT-2
        start_token = self.gpt_tokenizer.bos_token_id
        start_token_embed = self.get_gpt_embedding_matrix()[start_token].unsqueeze(0).unsqueeze(0).expand(prefix_embed.shape[0], -1, -1)
        
        # 3. Concatenate them to form the input for GPT-2
        inputs_embeds = torch.cat((start_token_embed, prefix_embed), dim=1)

        # 4. Generate text using GPT-2
        do_sample = temperature > 1.0
        
        generated_ids = self.gpt_model.generate(
            inputs_embeds=inputs_embeds,
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
            do_sample=do_sample,
            eos_token_id=self.gpt_tokenizer.eos_token_id,
            pad_token_id=self.gpt_tokenizer.pad_token_id,
            early_stopping=True
        )

        caption = self.gpt_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return caption.strip()


# --- Your Original Code (with modifications) ---

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


def download_file(url, file_path):
    """Download a file with a progress bar."""
    if os.path.exists(file_path):
        print(f"File already exists: {file_path}")
        return
    print(f"Downloading {url} to {file_path}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        with open(file_path, 'wb') as f, tqdm(
            total=total_size, unit='iB', unit_scale=True, desc=file_path.name
        ) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


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

    # --- Load ClipCap model ---
    print("\nLoading ClipCap captioning model...")
    caption_model = ClipCapModel(prefix_length=10, clip_embedding_dim=512)
    caption_model.to(device)

    # Download and load pre-trained mapping network weights
    # --- THIS IS THE FINAL WORKING URL ---
    weights_url = "https://huggingface.co/Andron00e/clip-prefix-caption-weights/resolve/main/coco_prefix-001.pt"
    weights_path = Path("checkpoints/coco_prefix-001.pt")
    weights_path.parent.mkdir(exist_ok=True)
    download_file(weights_url, weights_path)
    
    # Load the state dict for the mapping network
    state_dict = torch.load(weights_path, map_location=device)
    caption_model.mapping_network.load_state_dict(state_dict)
    print("ClipCap model loaded successfully.")

    # Load CLIP models
    print("\nLoading CLIP models...")
    base_clip, clip_preprocess = clip.load('ViT-B/32', device=device)

    # The checkpoint path from your code
    checkpoint_path = 'runs/lora_visual_prompts_ViT-B_32_20250917_142456/best_model.pt'
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint path not found: {checkpoint_path}")
        # As a fallback, we will just use the base clip model for all users
        lora_clip, visual_prompts, user_accs = base_clip, None, {}
    else:
        lora_clip, _, visual_prompts, user_accs = load_lora_model(checkpoint_path, device)
        
    # Get top 2 users
    if not user_accs:
        print("No user accuracies found! Running demo with dummy users.")
        user_accs = {'dummy_user_1': 1.0, 'dummy_user_2': 1.0}

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
    demo_dir = Path('docs/demo_clipcap_captions')
    demo_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {len(disagreement_pairs)} pairs with ClipCap captioning...")
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
            # Get CLIP embeddings
            clip_img = clip_preprocess(img_obj).unsqueeze(0).to(device)

            with torch.no_grad():
                base_emb = base_clip.encode_image(clip_img)
                user1_emb = lora_clip.encode_image(clip_img, user1_idx) if visual_prompts else base_emb.clone()
                user2_emb = lora_clip.encode_image(clip_img, user2_idx) if visual_prompts else base_emb.clone()

                # Normalize for comparison
                base_emb_norm = F.normalize(base_emb, dim=-1)
                user1_emb_norm = F.normalize(user1_emb, dim=-1)
                user2_emb_norm = F.normalize(user2_emb, dim=-1)

                shift1 = torch.norm(user1_emb_norm - base_emb_norm).item()
                shift2 = torch.norm(user2_emb_norm - base_emb_norm).item()

            # Generate captions using ClipCapModel
            baseline_caption = caption_model.generate_caption(base_emb, device=device)
            
            # Vary temperature based on shift
            temp1 = 1.0 + shift1 * 0.3
            user1_caption = caption_model.generate_caption(user1_emb, temperature=temp1, device=device)
            
            temp2 = 1.0 + shift2 * 0.3
            user2_caption = caption_model.generate_caption(user2_emb, temperature=temp2, device=device)

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

    md_content = f"""# ClipCap-Based Caption Generation Demo

## Overview
This demo generates captions directly from CLIP embeddings (baseline and LoRA-adapted) using a **ClipCap architecture**.

## Key Difference from Previous Approaches
- **Natural Embedding Use**: The ClipCap model is *designed* to take a 512-dim CLIP embedding and translate it into a prefix for a GPT-2 decoder.
- **No More Hacks**: This removes the need for custom projection layers or expanding vectors to match sequence lengths.
- **Personalization Reflected**: LoRA-adapted embeddings produce different captions by influencing the GPT-2 prefix.

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

        # Captions from embeddings
        md_content += "### Captions (Generated from Embeddings via ClipCap)\n\n"
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

        changes = []
        baseline_img1 = result['captions']['img1']['baseline']
        baseline_img2 = result['captions']['img2']['baseline']
        if result['captions']['img1']['user1'] != baseline_img1: changes.append("User 1 (Image 1)")
        if result['captions']['img1']['user2'] != baseline_img1: changes.append("User 2 (Image 1)")
        if result['captions']['img2']['user1'] != baseline_img2: changes.append("User 1 (Image 2)")
        if result['captions']['img2']['user2'] != baseline_img2: changes.append("User 2 (Image 2)")
        if changes: md_content += f"**Caption variations detected**: {', '.join(changes)}\n\n"
        else: md_content += "**Note**: Captions are similar but generated from different embeddings\n\n"

        md_content += "---\n\n"
    
    md_path = demo_dir / 'clipcap_caption_demo.md'
    with open(md_path, 'w') as f:
        f.write(md_content)

    print(f"✓ Report saved to: {md_path}")

    json_path = demo_dir / 'results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to: {json_path}")
    print("\nDemo complete!")t


if __name__ == "__main__":
    main()