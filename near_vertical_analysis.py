"""
Timepix3 Near-Vertical Incidence Angle Physical Distinguishability Analysis
============================================================================
Analyzes whether angles 80-90 degrees are physically distinguishable
using handcrafted features from ToT (Time-over-Threshold) data.
"""

import os
import sys
import glob
import warnings

# Set thread limits before importing numpy/sklearn to avoid threadpoolctl issues
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ks_2samp, kurtosis, skew
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm
import time

warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
DATA_ROOT = r"E:\C1Analysis\C_Processed_1"
TARGET_ANGLES = [80, 82, 84, 86, 88, 90]
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPS = 1e-10  # epsilon for numerical stability


# ============================================================
# Step 0: Data Exploration
# ============================================================
def explore_data():
    """Explore dataset directory structure and data format."""
    print("=" * 70)
    print("STEP 0: Data Exploration")
    print("=" * 70)

    # List all subdirectories
    all_dirs = sorted([d for d in os.listdir(DATA_ROOT)
                       if os.path.isdir(os.path.join(DATA_ROOT, d))])
    print(f"\nAll angle folders in dataset: {all_dirs}")

    # Count samples per target angle
    angle_counts = {}
    angle_files = {}
    for angle in TARGET_ANGLES:
        angle_dir = os.path.join(DATA_ROOT, str(angle))
        if os.path.isdir(angle_dir):
            files = sorted(glob.glob(os.path.join(angle_dir, "*.txt")))
            angle_counts[angle] = len(files)
            angle_files[angle] = files
        else:
            print(f"  WARNING: Directory for angle {angle} not found!")
            angle_counts[angle] = 0
            angle_files[angle] = []

    print(f"\nSample counts per target angle:")
    for angle, count in angle_counts.items():
        print(f"  {angle} deg: {count} samples")
    print(f"  Total: {sum(angle_counts.values())} samples")

    # Check data format with a sample file
    sample_file = angle_files[TARGET_ANGLES[0]][0]
    sample_data = np.loadtxt(sample_file)
    print(f"\nSample file: {os.path.basename(sample_file)}")
    print(f"  Shape: {sample_data.shape}")
    print(f"  dtype: {sample_data.dtype}")
    active = sample_data[sample_data > 0]
    print(f"  Active pixels: {len(active)}")
    if len(active) > 0:
        print(f"  ToT range (active): [{active.min():.2f}, {active.max():.2f}]")
        print(f"  ToT mean (active): {active.mean():.2f}")

    return angle_files


# ============================================================
# Step 1: Feature Extraction
# ============================================================
def compute_gini(values):
    """Compute Gini coefficient for an array of values."""
    if len(values) <= 1:
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    index = np.arange(1, n + 1)
    return (2.0 * np.sum(index * sorted_vals) / (n * np.sum(sorted_vals) + EPS)) - (n + 1.0) / n


def extract_features_single(data):
    """Extract all features from a single 2D ToT array."""
    active_mask = data > 0
    active_values = data[active_mask]
    n_pixels = len(active_values)

    features = {}

    # === Handle edge case: no active pixels ===
    if n_pixels == 0:
        features['n_pixels'] = 0
        features['bbox_width'] = 0
        features['bbox_height'] = 0
        features['aspect_ratio'] = 1.0
        features['eccentricity'] = 0.0
        features['compactness'] = 0.0
        features['diagonal_length'] = 0.0
        features['total_tot'] = 0.0
        features['max_tot'] = 0.0
        features['mean_tot'] = 0.0
        features['std_tot'] = 0.0
        features['max_tot_fraction'] = 1.0
        features['tot_entropy'] = 0.0
        features['tot_gini'] = 0.0
        features['tot_kurtosis'] = 0.0
        features['tot_skewness'] = 0.0
        features['energy_gradient'] = 0.0
        features['weighted_centroid_offset'] = 0.0
        features['max_tot_position_ratio'] = 0.0
        features['second_moment'] = 0.0
        features['tot_weighted_radius'] = 0.0
        return features

    # Active pixel coordinates
    ys, xs = np.where(active_mask)

    # === Geometric features ===
    features['n_pixels'] = n_pixels
    x_range = xs.max() - xs.min()
    y_range = ys.max() - ys.min()
    bbox_w = x_range + 1
    bbox_h = y_range + 1
    features['bbox_width'] = bbox_w
    features['bbox_height'] = bbox_h
    features['aspect_ratio'] = max(bbox_w, bbox_h) / (min(bbox_w, bbox_h) + EPS)

    if n_pixels >= 3:
        coords = np.column_stack([xs, ys]).astype(float)
        cov_matrix = np.cov(coords.T)
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        eigenvalues = np.maximum(eigenvalues, 0)
        lam_max = max(eigenvalues)
        lam_min = min(eigenvalues)
        features['eccentricity'] = np.sqrt(1.0 - lam_min / (lam_max + EPS))
    else:
        features['eccentricity'] = 0.0

    features['compactness'] = n_pixels / (bbox_w * bbox_h + EPS)
    features['diagonal_length'] = np.sqrt(x_range**2 + y_range**2)

    # === Energy deposition features ===
    total_tot = active_values.sum()
    max_tot = active_values.max()
    features['total_tot'] = total_tot
    features['max_tot'] = max_tot
    features['mean_tot'] = active_values.mean()
    features['std_tot'] = active_values.std() if n_pixels >= 2 else 0.0
    features['max_tot_fraction'] = max_tot / (total_tot + EPS)

    # Entropy
    p = active_values / (total_tot + EPS)
    p = p[p > 0]
    features['tot_entropy'] = -np.sum(p * np.log(p + EPS))

    # Gini
    features['tot_gini'] = compute_gini(active_values)

    # Kurtosis & Skewness
    if n_pixels >= 4:
        features['tot_kurtosis'] = float(kurtosis(active_values, fisher=True))
        features['tot_skewness'] = float(skew(active_values))
    else:
        features['tot_kurtosis'] = 0.0
        features['tot_skewness'] = 0.0

    # Energy gradient
    if n_pixels >= 3:
        sorted_desc = np.sort(active_values)[::-1]
        diffs = np.abs(np.diff(sorted_desc))
        features['energy_gradient'] = diffs.mean()
    else:
        features['energy_gradient'] = 0.0

    # === Spatial-energy cross features ===
    # Geometric centroid
    geo_cx = xs.mean()
    geo_cy = ys.mean()

    # Weighted centroid
    w_sum = total_tot + EPS
    w_cx = np.sum(active_values * xs) / w_sum
    w_cy = np.sum(active_values * ys) / w_sum

    if n_pixels >= 2:
        features['weighted_centroid_offset'] = np.sqrt(
            (w_cx - geo_cx)**2 + (w_cy - geo_cy)**2)
    else:
        features['weighted_centroid_offset'] = 0.0

    # Distances from weighted centroid
    dists = np.sqrt((xs - w_cx)**2 + (ys - w_cy)**2)

    # max_tot_position_ratio
    max_idx = np.argmax(active_values)
    dist_max_tot = dists[max_idx]
    max_dist = dists.max()
    features['max_tot_position_ratio'] = dist_max_tot / (max_dist + EPS) if max_dist > 0 else 0.0

    # Second moment
    features['second_moment'] = np.sum(active_values * dists**2) / w_sum

    # ToT weighted radius
    features['tot_weighted_radius'] = np.sum(active_values * dists) / w_sum

    return features


def extract_all_features(angle_files):
    """Extract features for all samples across target angles."""
    print("\n" + "=" * 70)
    print("STEP 1: Feature Extraction")
    print("=" * 70)

    csv_path = os.path.join(OUTPUT_DIR, "near_vertical_features.csv")

    # If CSV already exists, load it directly
    if os.path.exists(csv_path):
        print(f"  Found existing features CSV, loading: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"DataFrame shape: {df.shape}")
        print(f"\nSamples per angle:")
        print(df['angle'].value_counts().sort_index().to_string())
        return df

    all_records = []

    for angle in TARGET_ANGLES:
        files = angle_files[angle]
        print(f"  Processing angle {angle} deg: {len(files)} files...", flush=True)
        for i, fpath in enumerate(files):
            data = np.loadtxt(fpath)
            feats = extract_features_single(data)
            feats['angle'] = angle
            feats['sample_id'] = f"{angle}_{i}"
            all_records.append(feats)
            if (i + 1) % 5000 == 0:
                print(f"    {i+1}/{len(files)} done", flush=True)
        print(f"    {angle} deg done.", flush=True)
    df = pd.DataFrame(all_records)

    # NaN/Inf check
    nan_count = df.isna().sum().sum()
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    if nan_count > 0 or inf_count > 0:
        print(f"  WARNING: Found {nan_count} NaN, {inf_count} Inf values. Replacing...")
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    df.to_csv(csv_path, index=False)
    print(f"\nFeatures saved to: {csv_path}")
    print(f"DataFrame shape: {df.shape}")
    print(f"\nSamples per angle:")
    print(df['angle'].value_counts().sort_index().to_string())
    print(f"\nFeature statistics:")
    feature_cols = [c for c in df.columns if c not in ('angle', 'sample_id')]
    print(df[feature_cols].describe().to_string())

    return df


# ============================================================
# Step 2: Feature Distribution Visualization
# ============================================================
def plot_feature_distributions(df):
    """Plot histograms of key features across angles."""
    print("\n" + "=" * 70)
    print("STEP 2: Feature Distribution Visualization")
    print("=" * 70)

    plot_features = [
        'n_pixels', 'eccentricity', 'total_tot', 'max_tot_fraction',
        'tot_entropy', 'tot_gini', 'max_tot', 'std_tot',
        'tot_kurtosis', 'second_moment', 'tot_weighted_radius',
        'weighted_centroid_offset'
    ]

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']
    angle_labels = [str(a) + '°' for a in TARGET_ANGLES]

    fig, axes = plt.subplots(4, 3, figsize=(20, 24))
    axes = axes.flatten()

    for idx, feat in enumerate(plot_features):
        ax = axes[idx]
        for j, angle in enumerate(TARGET_ANGLES):
            subset = df[df['angle'] == angle][feat].dropna()
            # Clip extreme outliers for visualization
            q01, q99 = subset.quantile(0.01), subset.quantile(0.99)
            subset_clipped = subset[(subset >= q01) & (subset <= q99)]
            ax.hist(subset_clipped, bins=60, alpha=0.35, density=True,
                    color=colors[j], label=angle_labels[j])
        ax.set_title(feat, fontweight='bold', fontsize=13)
        ax.legend(fontsize=8)
        ax.set_ylabel('Density')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "feature_distributions_by_angle.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Distribution plot saved to: {save_path}")


# ============================================================
# Step 3: KS Statistical Test
# ============================================================
def ks_test_analysis(df):
    """Perform KS tests between adjacent angle pairs."""
    print("\n" + "=" * 70)
    print("STEP 3: KS Statistical Test")
    print("=" * 70)

    feature_cols = [c for c in df.columns if c not in ('angle', 'sample_id')]
    angle_pairs = [(80, 82), (82, 84), (84, 86), (86, 88), (88, 90)]
    pair_labels = [f"{a}-{b}" for a, b in angle_pairs]

    ks_stats = pd.DataFrame(index=feature_cols, columns=pair_labels, dtype=float)
    ks_pvals = pd.DataFrame(index=feature_cols, columns=pair_labels, dtype=float)

    for feat in feature_cols:
        for (a1, a2), label in zip(angle_pairs, pair_labels):
            d1 = df[df['angle'] == a1][feat].values
            d2 = df[df['angle'] == a2][feat].values
            stat, pval = ks_2samp(d1, d2)
            ks_stats.loc[feat, label] = stat
            ks_pvals.loc[feat, label] = pval

    # Print formatted table
    print(f"\n{'Feature':<30}", end="")
    for label in pair_labels:
        print(f"{label:>12}", end="")
    print()
    print("-" * (30 + 12 * len(pair_labels)))

    for feat in feature_cols:
        print(f"{feat:<30}", end="")
        for label in pair_labels:
            stat = ks_stats.loc[feat, label]
            pval = ks_pvals.loc[feat, label]
            if pval < 0.001:
                sig = "***"
            elif pval < 0.01:
                sig = "** "
            elif pval < 0.05:
                sig = "*  "
            else:
                sig = "ns "
            print(f"  {stat:.3f}{sig}", end="")
        print()

    # KS Heatmap
    fig, ax = plt.subplots(figsize=(10, 12))
    ks_float = ks_stats.astype(float)
    sns.heatmap(ks_float, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0, vmax=0.3, ax=ax, linewidths=0.5)
    ax.set_title('KS Statistic Heatmap (Adjacent Angle Pairs)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Angle Pair')
    ax.set_ylabel('Feature')
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "ks_heatmap.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nKS heatmap saved to: {save_path}")

    return ks_stats, ks_pvals


# ============================================================
# Step 4: Random Forest Classification
# ============================================================
def random_forest_analysis(df):
    """Train Random Forest and evaluate joint distinguishability."""
    print("\n" + "=" * 70)
    print("STEP 4: Random Forest Multi-Feature Classification")
    print("=" * 70)

    feature_cols = [c for c in df.columns if c not in ('angle', 'sample_id')]
    X = df[feature_cols].values
    y = df['angle'].values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=15,
        class_weight='balanced', random_state=42, n_jobs=-1
    )

    fold_accs = []
    y_pred_all = np.zeros_like(y)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
        rf.fit(X_scaled[train_idx], y[train_idx])
        y_pred_fold = rf.predict(X_scaled[val_idx])
        y_pred_all[val_idx] = y_pred_fold
        acc = accuracy_score(y[val_idx], y_pred_fold)
        fold_accs.append(acc)
        print(f"  Fold {fold_idx+1}: Accuracy = {acc:.4f}")

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    print(f"\n5-Fold Mean Accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")
    print(f"Random Baseline (6 classes): {1/6:.4f}")

    # Classification report
    print(f"\nClassification Report:")
    report = classification_report(y, y_pred_all, target_names=[f"{a} deg" for a in TARGET_ANGLES])
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y, y_pred_all, labels=TARGET_ANGLES)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f"{a}°" for a in TARGET_ANGLES],
                yticklabels=[f"{a}°" for a in TARGET_ANGLES], ax=ax)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix - Random Forest', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "confusion_matrix_rf.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")

    # Feature importance (train on full data)
    rf_full = RandomForestClassifier(
        n_estimators=300, max_depth=15,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf_full.fit(X_scaled, y)
    importances = rf_full.feature_importances_
    sorted_idx = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(sorted_idx)), importances[sorted_idx], color='steelblue')
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_cols[i] for i in sorted_idx])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Random Forest Feature Importance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Feature importance saved to: {save_path}")

    return mean_acc, std_acc, cm, importances, feature_cols, report


# ============================================================
# Step 5: Dimensionality Reduction Visualization
# ============================================================
def dimensionality_reduction(df):
    """PCA and t-SNE visualization."""
    print("\n" + "=" * 70)
    print("STEP 5: Dimensionality Reduction Visualization")
    print("=" * 70)

    feature_cols = [c for c in df.columns if c not in ('angle', 'sample_id')]
    X = df[feature_cols].values
    y = df['angle'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    print("  Computing PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    var_ratio = pca.explained_variance_ratio_
    cumulative_var = sum(var_ratio)

    # t-SNE with subsampling
    print("  Computing t-SNE (subsampled)...")
    max_per_class = 2000
    subsample_idx = []
    rng = np.random.RandomState(42)
    for angle in TARGET_ANGLES:
        angle_idx = np.where(y == angle)[0]
        if len(angle_idx) > max_per_class:
            chosen = rng.choice(angle_idx, max_per_class, replace=False)
        else:
            chosen = angle_idx
        subsample_idx.extend(chosen)
    subsample_idx = np.array(subsample_idx)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    try:
        X_tsne = tsne.fit_transform(X_scaled[subsample_idx])
        tsne_success = True
    except Exception as e:
        print(f"  WARNING: t-SNE failed ({e}), skipping t-SNE plot.")
        tsne_success = False
    y_tsne = y[subsample_idx]

    # Plot
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']
    if tsne_success:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(9, 7))

    for j, angle in enumerate(TARGET_ANGLES):
        mask = y == angle
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[j],
                    label=f"{angle}°", alpha=0.15, s=8)
    ax1.set_xlabel(f'PC1 ({var_ratio[0]:.1%})')
    ax1.set_ylabel(f'PC2 ({var_ratio[1]:.1%})')
    ax1.set_title(f'PCA (Cumulative Variance: {cumulative_var:.1%})',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, markerscale=3)

    if tsne_success:
        for j, angle in enumerate(TARGET_ANGLES):
            mask = y_tsne == angle
            ax2.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=colors[j],
                        label=f"{angle}°", alpha=0.15, s=8)
        ax2.set_xlabel('t-SNE 1')
        ax2.set_ylabel('t-SNE 2')
        ax2.set_title('t-SNE (perplexity=30)', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=9, markerscale=3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "dimensionality_reduction.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Dimensionality reduction plot saved to: {save_path}")


# ============================================================
# Step 6: Generate Analysis Report
# ============================================================
def generate_report(df, ks_stats, ks_pvals, mean_acc, std_acc, cm,
                    importances, feature_cols, classification_rpt):
    """Generate comprehensive analysis report."""
    print("\n" + "=" * 70)
    print("STEP 6: Analysis Report")
    print("=" * 70)

    lines = []
    lines.append("=" * 70)
    lines.append("ANALYSIS REPORT: Timepix3 Near-Vertical Angle Distinguishability")
    lines.append("=" * 70)
    lines.append("")

    # 1. Data overview
    lines.append("1. DATA OVERVIEW")
    lines.append("-" * 40)
    for angle in TARGET_ANGLES:
        sub = df[df['angle'] == angle]
        n = len(sub)
        mean_npix = sub['n_pixels'].mean()
        mean_tot = sub['total_tot'].mean()
        max_tot = sub['max_tot'].max()
        lines.append(f"  {angle} deg: {n} samples, "
                     f"avg pixels={mean_npix:.1f}, "
                     f"avg total_ToT={mean_tot:.1f}, "
                     f"max single ToT={max_tot:.1f}")
    feat_only = [c for c in df.columns if c not in ('angle', 'sample_id')]
    tot_range = df['total_tot']
    lines.append(f"\n  Total ToT range: [{tot_range.min():.1f}, {tot_range.max():.1f}]")
    max_tot_range = df['max_tot']
    lines.append(f"  Max single-pixel ToT range: [{max_tot_range.min():.1f}, {max_tot_range.max():.1f}]")
    lines.append("")

    # 2. Feature importance ranking
    lines.append("2. FEATURE IMPORTANCE RANKING (Random Forest)")
    lines.append("-" * 40)
    sorted_idx = np.argsort(importances)[::-1]
    for rank, idx in enumerate(sorted_idx[:10], 1):
        lines.append(f"  #{rank}: {feature_cols[idx]} = {importances[idx]:.4f}")
    lines.append("")

    # 3. KS test conclusions
    lines.append("3. KS TEST CONCLUSIONS")
    lines.append("-" * 40)
    angle_pairs = [(80, 82), (82, 84), (84, 86), (86, 88), (88, 90)]
    pair_labels = [f"{a}-{b}" for a, b in angle_pairs]

    for label in pair_labels:
        ks_col = ks_stats[label].astype(float)
        strong = ks_col[ks_col > 0.1].sort_values(ascending=False)
        moderate = ks_col[(ks_col > 0.05) & (ks_col <= 0.1)].sort_values(ascending=False)
        lines.append(f"\n  Angle pair {label}:")
        if len(strong) > 0:
            feats_str = ", ".join([f"{f}({v:.3f})" for f, v in strong.items()])
            lines.append(f"    Strong (KS>0.1): {feats_str}")
        else:
            lines.append(f"    Strong (KS>0.1): NONE")
        if len(moderate) > 0:
            feats_str = ", ".join([f"{f}({v:.3f})" for f, v in moderate.items()])
            lines.append(f"    Moderate (0.05<KS<=0.1): {feats_str}")
        weakest = ks_col.max()
        lines.append(f"    Max KS statistic: {weakest:.4f}")
    lines.append("")

    # 4. Random Forest accuracy
    lines.append("4. RANDOM FOREST CLASSIFICATION RESULTS")
    lines.append("-" * 40)
    lines.append(f"  5-Fold Mean Accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")
    lines.append(f"  Random baseline (6 classes): {1/6:.4f}")
    lines.append(f"  Improvement over random: {(mean_acc - 1/6) / (1/6) * 100:.1f}%")
    lines.append(f"\n  Classification Report:")
    lines.append(classification_rpt)

    # Most confused pairs from confusion matrix
    lines.append("  Most confused angle pairs (from confusion matrix):")
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    for i in range(len(TARGET_ANGLES)):
        for j in range(len(TARGET_ANGLES)):
            if i != j and cm_norm[i, j] > 0.15:
                lines.append(f"    {TARGET_ANGLES[i]} deg -> predicted as "
                             f"{TARGET_ANGLES[j]} deg: {cm_norm[i,j]:.1%}")
    lines.append("")

    # 5. Comprehensive verdict
    lines.append("5. COMPREHENSIVE VERDICT")
    lines.append("-" * 40)

    for (a1, a2), label in zip(angle_pairs, pair_labels):
        ks_col = ks_stats[label].astype(float)
        max_ks = ks_col.max()
        n_strong = (ks_col > 0.1).sum()
        # Check confusion matrix
        i1 = TARGET_ANGLES.index(a1)
        i2 = TARGET_ANGLES.index(a2)
        confusion_rate = cm_norm[i1, i2] + cm_norm[i2, i1]

        if max_ks > 0.15 and n_strong >= 3:
            verdict = "DISTINGUISHABLE (moderate)"
        elif max_ks > 0.1 and n_strong >= 1:
            verdict = "WEAKLY DISTINGUISHABLE"
        elif max_ks > 0.05:
            verdict = "BARELY DISTINGUISHABLE"
        else:
            verdict = "INDISTINGUISHABLE"

        lines.append(f"  {a1} vs {a2} deg: {verdict}")
        lines.append(f"    Max KS={max_ks:.3f}, "
                     f"features with KS>0.1: {n_strong}, "
                     f"mutual confusion rate: {confusion_rate:.1%}")

    # Binning recommendation
    lines.append(f"\n  ANGLE BINNING RECOMMENDATION:")
    if mean_acc < 0.25:
        lines.append("  Given accuracy near random baseline, most adjacent pairs are")
        lines.append("  physically indistinguishable. Recommended binning strategies:")
        lines.append("    Option A: Merge all into single class [80-90] (2 deg resolution impossible)")
        lines.append("    Option B: Try 3 bins [80-82], [84-86], [88-90]")
        lines.append("    Option C: Try 2 bins [80-84], [86-90]")
        lines.append("  The optimal binning depends on which pairs show the highest KS statistics.")
    elif mean_acc < 0.40:
        lines.append("  Some separation exists but is weak. Consider merging highly confused pairs.")
        lines.append("  Focus deep learning on the features with highest RF importance.")
    else:
        lines.append("  Meaningful separation exists. Deep learning optimization is viable.")
        lines.append("  Focus on energy deposition features (max_tot_fraction, tot_entropy, etc.).")

    lines.append("")

    # 6. Recommendations for modeling
    lines.append("6. RECOMMENDATIONS FOR SUBSEQUENT MODELING")
    lines.append("-" * 40)
    top5_feat = [feature_cols[i] for i in np.argsort(importances)[::-1][:5]]
    lines.append(f"  Top 5 features to prioritize: {', '.join(top5_feat)}")
    lines.append("")
    lines.append("  If pursuing deep learning:")
    lines.append("    - Inject top handcrafted features as auxiliary inputs")
    lines.append("    - Use attention mechanisms to focus on energy distribution patterns")
    lines.append("    - Consider regression instead of classification for continuous angle estimation")
    lines.append("    - Augment with synthetic data at boundary angles")
    lines.append("")
    lines.append("  If adjusting classification granularity:")
    lines.append("    - Merge angles where mutual confusion > 30%")
    lines.append("    - Report angular resolution limit of Timepix3 detector in thesis")
    lines.append("    - This is a legitimate physics result demonstrating detector limitations")

    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)

    report_text = "\n".join(lines)
    print(report_text)

    report_path = os.path.join(OUTPUT_DIR, "analysis_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\nReport saved to: {report_path}")


# ============================================================
# Main
# ============================================================
def main():
    start_time = time.time()

    # Step 0: Explore data
    angle_files = explore_data()

    # Step 1: Extract features
    df = extract_all_features(angle_files)

    # Step 2: Distribution visualization
    plot_feature_distributions(df)

    # Step 3: KS tests
    ks_stats, ks_pvals = ks_test_analysis(df)

    # Step 4: Random Forest
    mean_acc, std_acc, cm, importances, feature_cols, report = random_forest_analysis(df)

    # Step 5: Dimensionality reduction
    dimensionality_reduction(df)

    # Step 6: Report
    generate_report(df, ks_stats, ks_pvals, mean_acc, std_acc, cm,
                    importances, feature_cols, report)

    elapsed = time.time() - start_time
    print(f"\nTotal elapsed time: {elapsed:.1f} seconds")
    print("All outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
