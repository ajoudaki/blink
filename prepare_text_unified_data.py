#!/usr/bin/env python3
"""
Prepare data for unified text model.
Since we don't have actual face images, we'll use the norming features as proxy embeddings.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import json


def prepare_data():
    """Prepare data and create proxy image embeddings from norming features."""

    # Load labels data
    with open('analysis/data/labels.pkl', 'rb') as f:
        df = pickle.load(f)

    print(f"Loaded {len(df)} total labels")
    print(f"Columns: {df.columns.tolist()[:20]}...")

    # Load norming data for feature-based embeddings
    norming_df = pd.read_csv('analysis/data/norming.csv')
    print(f"Loaded norming data for {len(norming_df)} faces")

    # Create mapping from Target (face ID) to features
    # Use facial measurements as proxy for image features (since we don't have actual images)
    feature_cols = [
        'Age', 'Attractive', 'Smart', 'Trustworthy', 'Dominant',
        'Happy', 'Angry', 'Sad', 'Surprised', 'Afraid',
        'Nose_Width', 'Nose_Length', 'Lip_Thickness', 'Face_Length',
        'Avg_Eye_Height', 'Avg_Eye_Width', 'Face_Width_Cheeks',
        'Forehead', 'Faceshape', 'Heartshapeness', 'Noseshape',
        'LipFullness', 'EyeShape', 'EyeSize', 'fWHR'
    ]

    # Filter to available columns
    available_features = [col for col in feature_cols if col in norming_df.columns]
    print(f"Using {len(available_features)} features as proxy embeddings")

    # Create feature vectors for each face
    face_embeddings = {}
    for _, row in norming_df.iterrows():
        face_id = row['Target']
        features = []
        for col in available_features:
            val = row[col]
            if pd.isna(val):
                features.append(0.0)
            else:
                features.append(float(val))

        # Normalize features
        features = np.array(features, dtype=np.float32)
        if np.std(features) > 0:
            features = (features - np.mean(features)) / np.std(features)

        # Pad to 512 dimensions to match CLIP (repeat features cyclically)
        target_dim = 512
        if len(features) < target_dim:
            repeat_times = target_dim // len(features) + 1
            features = np.tile(features, repeat_times)[:target_dim]

        face_embeddings[face_id] = features

    print(f"Created embeddings for {len(face_embeddings)} faces")

    # Process rating data
    rating_data = []
    comparison_data = []

    # Filter to rows with valid labels
    valid_df = df[df['label'].notna()].copy()
    print(f"Found {len(valid_df)} labels with non-null values")

    # Determine task type based on label structure
    for _, row in valid_df.iterrows():
        label = row['label']
        user_id = row['user_id']

        # Parse label (it might be JSON string)
        if isinstance(label, str):
            try:
                label_data = json.loads(label)
            except:
                continue
        else:
            label_data = label

        # Check if it's rating or comparison
        if isinstance(label_data, dict):
            # Extract target and check for rating vs comparison
            if 'attractiveness' in label_data:
                # Individual rating
                for target_key in ['attractiveness', 'intelligence', 'trustworthiness']:
                    if target_key in label_data:
                        target = target_key.replace('attractiveness', 'attractive').replace('intelligence', 'smart').replace('trustworthiness', 'trustworthy')

                        rating_val = label_data[target_key]
                        if isinstance(rating_val, dict) and 'rating' in rating_val:
                            rating = rating_val['rating']
                            face_id = rating_val.get('image', row.get('Target', row.get('item_id')))

                            if face_id in face_embeddings and rating is not None:
                                rating_data.append({
                                    'user_id': user_id,
                                    'image': face_id,
                                    'target': target,
                                    'rating': int(rating)
                                })

            elif 'winner' in str(label_data) or 'loser' in str(label_data):
                # Pairwise comparison
                for target_key in ['attractiveness', 'intelligence', 'trustworthiness']:
                    if target_key in label_data:
                        target = target_key.replace('attractiveness', 'attractive').replace('intelligence', 'smart').replace('trustworthiness', 'trustworthy')

                        comp_val = label_data[target_key]
                        if isinstance(comp_val, dict):
                            winner = comp_val.get('winner')
                            loser = comp_val.get('loser')

                            if winner in face_embeddings and loser in face_embeddings:
                                comparison_data.append({
                                    'user_id': user_id,
                                    'winner_image': winner,
                                    'loser_image': loser,
                                    'target': target
                                })

    print(f"Extracted {len(rating_data)} rating labels")
    print(f"Extracted {len(comparison_data)} comparison labels")

    # Save processed data
    cache_dir = Path("cached_data")
    cache_dir.mkdir(exist_ok=True)

    # Save embeddings
    with open(cache_dir / "clip_embeddings.pkl", 'wb') as f:
        pickle.dump(face_embeddings, f)
    print(f"Saved embeddings to cached_data/clip_embeddings.pkl")

    # Save rating data
    if rating_data:
        rating_df = pd.DataFrame(rating_data)
        rating_df.to_csv(cache_dir / "rating_labels.csv", index=False)
        print(f"Saved {len(rating_df)} rating labels")
        print(f"Rating targets: {rating_df['target'].value_counts().to_dict()}")
        print(f"Unique users: {rating_df['user_id'].nunique()}")

    # Save comparison data
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(cache_dir / "comparison_labels.csv", index=False)
        print(f"Saved {len(comparison_df)} comparison labels")
        print(f"Comparison targets: {comparison_df['target'].value_counts().to_dict()}")
        print(f"Unique users: {comparison_df['user_id'].nunique()}")

    return face_embeddings, rating_data, comparison_data


if __name__ == "__main__":
    prepare_data()