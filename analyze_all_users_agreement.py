#!/usr/bin/env python3
"""
Compute pairwise agreement matrix for all users.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import itertools

# Load data files
print("Loading data...")
df_labels = pd.read_excel('data/big_compare_label.xlsx')
df_data = pd.read_excel('data/big_compare_data.xlsx')

# Merge dataframes
df = df_data.merge(df_labels, left_on='_id', right_on='item_id', how='inner')

# Get all unique users with sufficient data (same threshold as in training)
user_counts = df['user_id'].value_counts()
min_samples = 100
valid_users = user_counts[user_counts >= min_samples].index.tolist()

print(f"Found {len(valid_users)} users with >= {min_samples} samples")
print()

# Sort users for consistent ordering
valid_users = sorted(valid_users)

# Create a dictionary to store each user's comparisons
user_comparisons = {}

print("Processing user comparisons...")
for user_id in valid_users:
    user_data = df[df['user_id'] == user_id]
    user_comparisons[user_id] = {}

    for _, row in user_data.iterrows():
        im1 = str(row['im1_path'])
        im2 = str(row['im2_path'])

        # Create a consistent pair key (smaller id first)
        pair_key = tuple(sorted([im1, im2]))

        if pair_key not in user_comparisons[user_id]:
            user_comparisons[user_id][pair_key] = {}

        # Process each attribute
        for attr in ['attractive', 'smart', 'trustworthy']:
            value = row[attr]
            if value in [1, 2]:
                # 2 means first image (im1) wins, 1 means second image (im2) wins
                winner = im1 if value == 2 else im2
                user_comparisons[user_id][pair_key][attr] = winner

# Compute pairwise agreement matrix
n_users = len(valid_users)
agreement_matrix = np.zeros((n_users, n_users))
comparison_counts = np.zeros((n_users, n_users))

print("Computing pairwise agreements...")
for i, user1_id in enumerate(valid_users):
    for j, user2_id in enumerate(valid_users):
        if i <= j:  # Only compute upper triangle (including diagonal)
            user1_comps = user_comparisons[user1_id]
            user2_comps = user_comparisons[user2_id]

            # Find shared pairs
            shared_pairs = set(user1_comps.keys()) & set(user2_comps.keys())

            if len(shared_pairs) > 0:
                agree_count = 0
                total_count = 0

                for pair in shared_pairs:
                    user1_labels = user1_comps[pair]
                    user2_labels = user2_comps[pair]

                    # Check each target
                    for target in user1_labels:
                        if target in user2_labels:
                            total_count += 1
                            if user1_labels[target] == user2_labels[target]:
                                agree_count += 1

                if total_count > 0:
                    agreement_pct = (agree_count / total_count) * 100
                    agreement_matrix[i, j] = agreement_pct
                    agreement_matrix[j, i] = agreement_pct  # Symmetric
                    comparison_counts[i, j] = total_count
                    comparison_counts[j, i] = total_count

# Create DataFrame for better display
agreement_df = pd.DataFrame(
    agreement_matrix,
    index=[f"User{i+1}" for i in range(n_users)],
    columns=[f"User{i+1}" for i in range(n_users)]
)

counts_df = pd.DataFrame(
    comparison_counts.astype(int),
    index=[f"User{i+1}" for i in range(n_users)],
    columns=[f"User{i+1}" for i in range(n_users)]
)

# Print results
print()
print("=" * 80)
print("PAIRWISE AGREEMENT MATRIX (% Agreement)")
print("=" * 80)
print()
print(agreement_df.round(1).to_string())

print()
print("=" * 80)
print("NUMBER OF SHARED COMPARISONS MATRIX")
print("=" * 80)
print()
print(counts_df.to_string())

# Print summary statistics
print()
print("=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

# Get off-diagonal elements (excluding self-agreement)
off_diagonal_mask = ~np.eye(n_users, dtype=bool)
off_diagonal_agreements = agreement_matrix[off_diagonal_mask]
off_diagonal_agreements = off_diagonal_agreements[off_diagonal_agreements > 0]  # Exclude zeros

if len(off_diagonal_agreements) > 0:
    print(f"Average pairwise agreement: {off_diagonal_agreements.mean():.2f}%")
    print(f"Std deviation: {off_diagonal_agreements.std():.2f}%")
    print(f"Min agreement: {off_diagonal_agreements.min():.2f}%")
    print(f"Max agreement: {off_diagonal_agreements.max():.2f}%")
    print()

    # Find most and least agreeable pairs
    for i in range(n_users):
        for j in range(i+1, n_users):
            if agreement_matrix[i, j] == off_diagonal_agreements.max():
                print(f"Most agreeable pair: User{i+1} & User{j+1} ({agreement_matrix[i, j]:.1f}%)")
            if agreement_matrix[i, j] == off_diagonal_agreements.min():
                print(f"Least agreeable pair: User{i+1} & User{j+1} ({agreement_matrix[i, j]:.1f}%)")

# Save to CSV for further analysis
print()
print("Saving results to CSV files...")
agreement_df.round(2).to_csv('user_agreement_matrix.csv')
counts_df.to_csv('user_comparison_counts.csv')
print("Saved: user_agreement_matrix.csv and user_comparison_counts.csv")

# Also create a heatmap-ready format with user IDs
user_id_map = {f"User{i+1}": valid_users[i][:12] + "..." for i in range(n_users)}
print()
print("User ID mapping:")
for user_label, user_id in user_id_map.items():
    full_id = valid_users[int(user_label.replace("User", "")) - 1]
    sample_count = user_counts[full_id]
    print(f"  {user_label}: {user_id} ({sample_count} samples)")

# Compute Cohen's Kappa for each pair
print()
print("=" * 80)
print("COHEN'S KAPPA VALUES (Inter-rater Reliability)")
print("=" * 80)

kappa_matrix = np.zeros((n_users, n_users))
for i in range(n_users):
    for j in range(n_users):
        if i != j and agreement_matrix[i, j] > 0:
            po = agreement_matrix[i, j] / 100  # Observed agreement
            pe = 0.5  # Expected agreement by chance for binary choice
            kappa = (po - pe) / (1 - pe)
            kappa_matrix[i, j] = kappa

kappa_df = pd.DataFrame(
    kappa_matrix,
    index=[f"User{i+1}" for i in range(n_users)],
    columns=[f"User{i+1}" for i in range(n_users)]
)

print()
print(kappa_df.round(3).to_string())

# Kappa interpretation
print()
print("Kappa interpretation: <0: worse than chance, 0-0.2: slight, 0.2-0.4: fair,")
print("                      0.4-0.6: moderate, 0.6-0.8: substantial, >0.8: almost perfect")