#!/usr/bin/env python3
"""
Analyze agreement between User 1 and User 4 on shared image pairs.
"""

import pandas as pd
from collections import defaultdict

# Load data files
df_labels = pd.read_excel('data/big_compare_label.xlsx')
df_data = pd.read_excel('data/big_compare_data.xlsx')

# Merge dataframes
df = df_data.merge(df_labels, left_on='_id', right_on='item_id', how='inner')

# Users to compare
user1_id = '5fc68c7d781dffc92b8a11e5'
user4_id = '601577ed152e6be8454a39cb'

print(f"Total comparisons in merged dataset: {len(df)}")
print(f"Unique labelers: {df['user_id'].nunique()}")
print()

# Filter data for our two users
user1_data = df[df['user_id'] == user1_id].copy()
user4_data = df[df['user_id'] == user4_id].copy()

print(f"User 1 ({user1_id[:12]}...) has {len(user1_data)} comparisons")
print(f"User 4 ({user4_id[:12]}...) has {len(user4_data)} comparisons")
print()

# Create comparison dictionaries
user1_comparisons = {}
user4_comparisons = {}

# Process User 1 comparisons
for _, row in user1_data.iterrows():
    im1 = str(row['im1_path'])
    im2 = str(row['im2_path'])

    # Create a consistent pair key (smaller id first)
    pair_key = tuple(sorted([im1, im2]))

    if pair_key not in user1_comparisons:
        user1_comparisons[pair_key] = {}

    # Process each attribute
    for attr in ['attractive', 'smart', 'trustworthy']:
        value = row[attr]
        if value in [1, 2]:
            # 2 means first image (im1) wins, 1 means second image (im2) wins
            winner = im1 if value == 2 else im2
            user1_comparisons[pair_key][attr] = winner

# Process User 4 comparisons
for _, row in user4_data.iterrows():
    im1 = str(row['im1_path'])
    im2 = str(row['im2_path'])

    # Create a consistent pair key (smaller id first)
    pair_key = tuple(sorted([im1, im2]))

    if pair_key not in user4_comparisons:
        user4_comparisons[pair_key] = {}

    # Process each attribute
    for attr in ['attractive', 'smart', 'trustworthy']:
        value = row[attr]
        if value in [1, 2]:
            # 2 means first image (im1) wins, 1 means second image (im2) wins
            winner = im1 if value == 2 else im2
            user4_comparisons[pair_key][attr] = winner

# Find shared pairs
shared_pairs = set(user1_comparisons.keys()) & set(user4_comparisons.keys())
print(f"Total shared image pairs between User 1 and User 4: {len(shared_pairs)}")
print()

if len(shared_pairs) == 0:
    print("No shared image pairs found between these users!")
else:
    # Analyze agreement
    overall_agree = 0
    overall_disagree = 0
    target_stats = defaultdict(lambda: {'agree': 0, 'disagree': 0})

    for pair in shared_pairs:
        user1_labels = user1_comparisons[pair]
        user4_labels = user4_comparisons[pair]

        # Check each target
        for target in user1_labels:
            if target in user4_labels:
                if user1_labels[target] == user4_labels[target]:
                    overall_agree += 1
                    target_stats[target]['agree'] += 1
                else:
                    overall_disagree += 1
                    target_stats[target]['disagree'] += 1

    # Print results
    print('=' * 60)
    print('OVERALL AGREEMENT STATISTICS')
    print('=' * 60)
    total_comparisons = overall_agree + overall_disagree
    if total_comparisons > 0:
        agree_pct = (overall_agree / total_comparisons) * 100
        disagree_pct = (overall_disagree / total_comparisons) * 100
        print(f'Total shared comparisons with labels: {total_comparisons}')
        print(f'Agreement: {overall_agree} ({agree_pct:.2f}%)')
        print(f'Disagreement: {overall_disagree} ({disagree_pct:.2f}%)')
        print()

        print('=' * 60)
        print('AGREEMENT BY TARGET VARIABLE')
        print('=' * 60)
        for target in ['attractive', 'smart', 'trustworthy']:
            if target in target_stats:
                total = target_stats[target]['agree'] + target_stats[target]['disagree']
                if total > 0:
                    agree_pct = (target_stats[target]['agree'] / total) * 100
                    disagree_pct = (target_stats[target]['disagree'] / total) * 100
                    print(f'\n{target.capitalize()}:')
                    print(f'  Total comparisons: {total}')
                    print(f'  Agree: {target_stats[target]["agree"]:4d} ({agree_pct:6.2f}%)')
                    print(f'  Disagree: {target_stats[target]["disagree"]:4d} ({disagree_pct:6.2f}%)')

        # Summary statistics
        print()
        print('=' * 60)
        print('SUMMARY')
        print('=' * 60)
        print(f"Out of {len(shared_pairs)} shared image pairs:")
        print(f"- Users evaluated the same targets {total_comparisons} times")
        print(f"- They agreed {overall_agree} times ({overall_agree/total_comparisons*100:.1f}%)")
        print(f"- They disagreed {overall_disagree} times ({overall_disagree/total_comparisons*100:.1f}%)")

        # Cohen's kappa for inter-rater reliability
        po = overall_agree / total_comparisons  # Observed agreement
        # For binary choice (winner from 2 images), expected agreement by chance is 0.5
        pe = 0.5
        kappa = (po - pe) / (1 - pe)
        print(f"\nCohen's Kappa (inter-rater reliability): {kappa:.3f}")
        if kappa < 0:
            print("  (Less agreement than expected by chance)")
        elif kappa < 0.20:
            print("  (Slight agreement)")
        elif kappa < 0.40:
            print("  (Fair agreement)")
        elif kappa < 0.60:
            print("  (Moderate agreement)")
        elif kappa < 0.80:
            print("  (Substantial agreement)")
        else:
            print("  (Almost perfect agreement)")
    else:
        print("No overlapping target evaluations found!")

# Additional analysis - look at some examples
print()
print('=' * 60)
print('SAMPLE DISAGREEMENTS (first 5)')
print('=' * 60)
count = 0
for pair in list(shared_pairs)[:50]:  # Check first 50 pairs
    if count >= 5:
        break
    user1_labels = user1_comparisons[pair]
    user4_labels = user4_comparisons[pair]

    for target in user1_labels:
        if target in user4_labels:
            if user1_labels[target] != user4_labels[target]:
                # Extract just the filename from the path
                img1_name = pair[0].split('/')[-1] if '/' in pair[0] else pair[0]
                img2_name = pair[1].split('/')[-1] if '/' in pair[1] else pair[1]
                winner1_name = user1_labels[target].split('/')[-1] if '/' in user1_labels[target] else user1_labels[target]
                winner4_name = user4_labels[target].split('/')[-1] if '/' in user4_labels[target] else user4_labels[target]

                print(f"Images: {img1_name[:15]}... vs {img2_name[:15]}...")
                print(f"  Target: {target}")
                print(f"  User 1 chose: {winner1_name[:15]}...")
                print(f"  User 4 chose: {winner4_name[:15]}...")
                print()
                count += 1
                if count >= 5:
                    break