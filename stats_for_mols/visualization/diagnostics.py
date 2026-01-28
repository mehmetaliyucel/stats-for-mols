# drug_stats/visualization/diagnostics.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import math

def plot_target_distribution(data, target_col, splitter, n_folds_to_plot=5):
    """
    Plots the distribution of the target variable for Train vs Test sets across folds.
    Handles both Continuous (Regression) and Binary (Classification) targets.
    """
    splits = list(splitter.split())
    print(f"Total folds available: {len(splits)}")
    n_plots = min(len(splits), n_folds_to_plot)
    n_cols = 5
    nrows= math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(nrows, n_cols, figsize=(5 * n_cols, 4 * nrows), sharey=True)
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    for i in range(n_plots):
        split_data = splits[i]
        
        val_idx = None
        if len(split_data) == 3:
            train_idx, val_idx, test_idx = split_data
        else:
            train_idx, test_idx = split_data
        
        y_train = data.iloc[train_idx][target_col]
        y_test = data.iloc[test_idx][target_col]
        y_val = data.iloc[val_idx][target_col] if val_idx is not None else None

        # Determine plot type based on target cardinality
        is_categorical = data[target_col].dtype == 'object' or len(data[target_col].unique()) < 15

        if is_categorical:
            labels = [y_train, y_test]
            sets = ['Train'] * len(y_train) + ['Test'] * len(y_test)
            
            if y_val is not None:
                labels.insert(1, y_val)
                sets[len(y_train):len(y_train)] = ['Validation'] * len(y_val)
            
            df_plot = pd.DataFrame({
                'Label': np.concatenate(labels),
                'Set': sets
            })
            
            sns.countplot(data=df_plot, x='Label', hue='Set', ax=axes[i], 
                          palette={'Train': 'blue', 'Validation': 'green', 'Test': 'orange'}, alpha=0.6)
            
        else:
            sns.kdeplot(y_train, ax=axes[i], fill=True, label='Train', color='blue', alpha=0.3)

            sns.kdeplot(y_test, ax=axes[i], fill=True, label='Test', color='orange', alpha=0.3)
            if y_val is not None:
                sns.kdeplot(y_val, ax=axes[i], fill=True, label='Validation', color='green', alpha=0.3)
            
        axes[i].set_title(f"Fold {i+1} Target Dist.")
        axes[i].set_xlabel(target_col)
        if i == 0: axes[i].legend()
    for j in range(i+1, nrows * n_cols):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    return fig

def plot_multiclass_target_distribution(data, target_col, splitter, n_folds_to_plot=5):
    """
    Plots the distribution of a multiclass target variable for Train vs Test sets across folds.
    Specifically designed for >2 classes to check for class imbalance shifts.
    """
    splits = list(splitter.split())
    print(f"Total folds available: {len(splits)}")

    n_plots = min(len(splits), n_folds_to_plot)
    n_cols = 5
    nrows= math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(nrows, n_cols, figsize=(5 * n_cols, 4 * nrows), sharey=True)
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    for i in range(n_plots):
        split_data = splits[i]
        
        val_idx = None
        if len(split_data) == 3:
            train_idx, val_idx, test_idx = split_data
        else:
            train_idx, test_idx = split_data
        
        # Prepare DataFrames for each split
        df_train = data.iloc[train_idx][[target_col]].copy()
        df_train['Set'] = 'Train'
        
        df_test = data.iloc[test_idx][[target_col]].copy()
        df_test['Set'] = 'Test'
        
        dfs_to_concat = [df_train, df_test]

        if val_idx is not None:
            df_val = data.iloc[val_idx][[target_col]].copy()
            df_val['Set'] = 'Validation'
            # Insert validation between train and test for better legend order
            dfs_to_concat.insert(1, df_val)

        df_combined = pd.concat(dfs_to_concat)

        # Plot
        sns.countplot(data=df_combined, y=target_col, hue='Set', ax=axes[i], 
                      palette={'Train': 'blue', 'Validation': 'green', 'Test': 'orange'}, alpha=0.5)
        
        axes[i].set_title(f"Fold {i+1} Multiclass Dist.")
        axes[i].set_xlabel("Count")
        if i == 0: axes[i].legend()
        
    for j in range(i+1, nrows * n_cols):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    return fig

def plot_descriptor_overlap(data, descriptor_cols, splitter, n_folds_to_plot=5):
    """
    Plots the distribution of selected descriptors for Train vs Test sets across folds.
    """
    splits = list(splitter.split())
    print(f"Total folds available: {len(splits)}")
    n_plots = min(len(splits), n_folds_to_plot)
    n_cols = 5
    nrows= math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(len(descriptor_cols), n_plots, figsize=(5 * n_plots, 4 * len(descriptor_cols)), sharey='row', squeeze=False)

    idx_map = {col: i for i, col in enumerate(descriptor_cols)}
    for i in range(n_plots):
        split_data = splits[i]
        
        val_idx = None
        if len(split_data) == 3:
            train_idx, val_idx, test_idx = split_data
        else:
            train_idx, test_idx = split_data        
        
        for col in descriptor_cols:
            row_idx = idx_map[col]
            ax = axes[row_idx, i]
            
            train_data = data.iloc[train_idx][col].dropna()
            test_data = data.iloc[test_idx][col].dropna()
            val_data = data.iloc[val_idx][col].dropna() if val_idx is not None else None

            sns.kdeplot(train_data, ax=ax, fill=True, label='Train', color='blue', alpha=0.3)
            if val_data is not None:
                sns.kdeplot(val_data, ax=ax, fill=True, label='Validation', color='green', alpha=0.3)
            sns.kdeplot(test_data, ax=ax, fill=True, label='Test', color='orange', alpha=0.3)
            
            ax.set_title(f"Fold {i+1} - {col}")
            ax.set_xlabel(col)
            if i == 0 and row_idx == 0: ax.legend()

    for j in range(i+1, nrows * n_cols):
        fig.delaxes(axes[row_idx, j])
        
    plt.tight_layout()
    return fig

def plot_pca_chemical_space(data, descriptor_cols, splitter, n_folds_to_plot=5):
    """
    Plots PCA of chemical space for Train vs Test sets across folds.
    """
    splits = list(splitter.split())
    print(f"Total folds available: {len(splits)}")
    n_plots = min(len(splits), n_folds_to_plot)
    if n_plots < 1:
        return None
    n_cols = 5
    nrows= math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(nrows, n_cols, figsize=(6 * n_cols, 5 * nrows), sharex=True, sharey=True)
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    # Fit PCA on the entire valid dataset
    pca = PCA(n_components=2)
    descriptor_data = data[descriptor_cols].dropna()
    pca.fit(descriptor_data)
    
    for i in range(n_plots):
        split_data = splits[i]
        
        val_idx = None
        if len(split_data) == 3:
            train_idx, val_idx, test_idx = split_data
        else:
            train_idx, test_idx = split_data
        
        train_data = data.iloc[train_idx][descriptor_cols].dropna()
        test_data = data.iloc[test_idx][descriptor_cols].dropna()
        val_data = data.iloc[val_idx][descriptor_cols].dropna() if val_idx is not None else None
        
        train_pca = pca.transform(train_data)
        test_pca = pca.transform(test_data)
        
        axes[i].scatter(train_pca[:, 0], train_pca[:, 1], label='Train', alpha=0.3, color='blue', s=10)
        if val_data is not None:
            val_pca = pca.transform(val_data)
            axes[i].scatter(val_pca[:, 0], val_pca[:, 1], label='Validation', alpha=0.4, color='green', marker='^', s=15)
        axes[i].scatter(test_pca[:, 0], test_pca[:, 1], label='Test', alpha=0.5, color='orange', marker='x', s=20)
        
        axes[i].set_title(f"Fold {i+1} PCA Space")
        axes[i].set_xlabel("PC 1")
        if i == 0: 
            axes[i].set_ylabel("PC 2")
            axes[i].legend()
    for j in range(i+1, nrows * n_cols):
        fig.delaxes(axes[j])
    plt.tight_layout()
    return fig

def plot_chemical_space_overlap(data, smiles_col, splitter, n_folds_to_plot=1):
    """
    Calculates the maximum Tanimoto similarity of each test compound to the training set.
    """
    splits = list(splitter.split())
    print(f"Total folds available: {len(splits)}")
    n_plots = min(len(splits), n_folds_to_plot)
    n_cols = 5
    nrows= math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(nrows, n_cols, figsize=(6 * n_cols, 4 * nrows), sharey=True)
    axes = axes.flatten() if n_plots > 1 else [axes]

    # Pre-calculate fingerprints for whole dataset to save time
    mols = [Chem.MolFromSmiles(s) for s in data[smiles_col]]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in mols if m]
    valid_idxs = [i for i, m in enumerate(mols) if m]
    
    # Map original index to fp index
    idx_map = {orig: i for i, orig in enumerate(valid_idxs)}

    for i in range(n_plots):
        split_data = splits[i]
        
        val_idx = None
        if len(split_data) == 3:
            train_idx, val_idx, test_idx = split_data
        else:   
            train_idx, test_idx = split_data
        
        train_fps = [fps[idx_map[idx]] for idx in train_idx if idx in idx_map]
        test_fps = [fps[idx_map[idx]] for idx in test_idx if idx in idx_map]
        val_fps = [fps[idx_map[idx]] for idx in val_idx if idx in idx_map] if val_idx is not None else []
        
        if not train_fps:
            continue

        # Validation vs Train
        max_sims_val = []
        if val_fps:
            for val_fp in val_fps:
                sims = DataStructs.BulkTanimotoSimilarity(val_fp, train_fps)
                max_sims_val.append(max(sims))

        # Test vs Train
        max_sims_test = []
        if test_fps:
            for test_fp in test_fps:
                sims = DataStructs.BulkTanimotoSimilarity(test_fp, train_fps)
                max_sims_test.append(max(sims))
        
        if max_sims_test:
            sns.histplot(max_sims_test, ax=axes[i], bins=20, kde=True, color='orange', label='Test vs Train', alpha=0.5, stat="density")
        
        if max_sims_val:
            sns.histplot(max_sims_val, ax=axes[i], bins=20, kde=True, color='green', label='Val vs Train', alpha=0.5, stat="density")
        
        axes[i].set_title(f"Fold {i+1}: Max Similarity to Train")
        axes[i].set_xlabel("Tanimoto Similarity")
        axes[i].set_ylabel("Density")
        axes[i].axvline(x=0.4, color='red', linestyle='--', label='Novelty Cutoff (0.4)')
        if i == 0: axes[i].legend()
    for j in range(i+1, nrows * n_cols):
        fig.delaxes(axes[j])
    plt.tight_layout()
    return fig

def plot_qq_residuals(metrics_df, model_col='model', metric_col='metric'):
    models = metrics_df[model_col].unique()
    n_models = len(models)
    
    n_cols = 5
    nrows= math.ceil(n_models / n_cols)
    fig, axes = plt.subplots(nrows, n_cols, figsize=(5 * n_cols, 5 * nrows), sharey=True)
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for i, model in enumerate(models):
        subset = metrics_df[metrics_df[model_col] == model][metric_col]
        
        if len(subset) < 2:
            continue
            
        residuals = (subset - subset.mean()) / subset.std()
        stats.probplot(residuals, dist="norm", plot=axes[i])
        
        axes[i].set_title(f"Q-Q Plot: {model}")
        axes[i].get_lines()[0].set_markerfacecolor('blue')
        axes[i].get_lines()[0].set_markeredgewidth(0)
        axes[i].get_lines()[1].set_color('red')
        
    for j in range(i+1, nrows * n_cols):
        fig.delaxes(axes[j])
    plt.tight_layout()

    return fig