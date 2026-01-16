# drug_stats/visualization/diagnostics.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from ..sampling.split_methods import DataSplitter
from scipy.stats import stats
from sklearn.decomposition import PCA
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def plot_target_distribution(data, target_col, splitter, n_folds_to_plot=5):
    """
    Plots the distribution of the target variable for Train vs Test sets across folds.
    Helps ensure that the split didn't create a distribution shift.
    """
    splits = list(splitter.split())
    
    n_plots = min(len(splits), n_folds_to_plot)
    
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4), sharey=True)
    if n_plots == 1: axes = [axes]
    
    for i in range(n_plots):
        train_idx, test_idx = splits[i]
        
        y_train = data.iloc[train_idx][target_col]
        y_test = data.iloc[test_idx][target_col]
        
        sns.kdeplot(y_train, ax=axes[i], fill=True, label='Train', color='blue', alpha=0.3)
        sns.kdeplot(y_test, ax=axes[i], fill=True, label='Test', color='orange', alpha=0.3)
        
        axes[i].set_title(f"Fold {i+1} Target Dist.")
        axes[i].set_xlabel(target_col)
        if i == 0: axes[i].legend()
        
    plt.tight_layout()
    return fig
def plot_descriptor_overlap(data, descriptor_cols, splitter, n_folds_to_plot=5):
    """
    Plots the distribution of selected descriptors for Train vs Test sets across folds.
    """
    splits = list(splitter.split())
    n_plots = min(len(splits), n_folds_to_plot)
    fig, axes = plt.subplots(len(descriptor_cols), n_plots, figsize=(5 * n_plots, 4 * len(descriptor_cols)), sharey='row', squeeze=False)

    idx_map = {col: i for i, col in enumerate(descriptor_cols)}
    for i in range(n_plots):
        train_idx, test_idx = splits[i]
        for col in descriptor_cols:
            row_idx = idx_map[col]
            ax = axes[row_idx, i]
            train_data = data.iloc[train_idx][col].dropna()
            test_data = data.iloc[test_idx][col].dropna()
            sns.kdeplot(train_data, ax=ax, fill=True, label='Train', color='blue', alpha=0.3)
            sns.kdeplot(test_data, ax=ax, fill=True, label='Test', color='orange', alpha=0.3)
            ax.set_title(f"Fold {i+1} - {col}")
            ax.set_xlabel(col)
            if i == 0: ax.legend() # Only add legend to first column
    plt.tight_layout()
    return fig
def plot_pca_chemical_space(data, descriptor_cols, splitter, n_folds_to_plot=1):
    """
    Plots PCA of chemical space for Train vs Test sets across folds.
    """
    splits = list(splitter.split())
    n_plots = min(len(splits), n_folds_to_plot)
    
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1: axes = [axes]
    
    pca = PCA(n_components=2)
    descriptor_data = data[descriptor_cols].dropna()
    pca.fit(descriptor_data)
    
    for i in range(n_plots):
        train_idx, test_idx = splits[i]
        
        train_data = data.iloc[train_idx][descriptor_cols].dropna()
        test_data = data.iloc[test_idx][descriptor_cols].dropna()
        
        train_pca = pca.transform(train_data)
        test_pca = pca.transform(test_data)
        
        axes[i].scatter(train_pca[:, 0], train_pca[:, 1], label='Train', alpha=0.3, color='blue')
        axes[i].scatter(test_pca[:, 0], test_pca[:, 1], label='Test', alpha=0.3, color='orange')
        
        axes[i].set_title(f"Fold {i+1} PCA Chemical Space")
        axes[i].set_xlabel("PCA Component 1")
        axes[i].set_ylabel("PCA Component 2")
        if i == 0: axes[i].legend()
        
    plt.tight_layout()
    return fig
def plot_multiclass_target_distribution(data, target_col, splitter, n_folds_to_plot=5):
    """
    Plots the distribution of a multiclass target variable for Train vs Test sets across folds.
    Helps ensure that the split didn't create a distribution shift.
    """
    splits = list(splitter.split())
    
    n_plots = min(len(splits), n_folds_to_plot)
    
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4), sharey=True)
    if n_plots == 1: axes = [axes]
    
    for i in range(n_plots):
        train_idx, test_idx = splits[i]
        
        df_train = data.iloc[train_idx][[target_col]].copy()
        df_test = data.iloc[test_idx][[target_col]].copy()
        df-train['Set'] = 'Train'
        df_test['Set'] = 'Test'
        df_combined = pd.concat([df_train, df_test])

        sns.countplot(data=df_combined,y=target_col, hue='Set', ax=axes[i], palette={'Train': 'blue', 'Test': 'orange'},alpha=0.3)
        
        
        axes[i].set_title(f"Fold {i+1} Target Dist.")
        axes[i].set_xlabel("Count")
        if i == 0: axes[i].legend()
        
    plt.tight_layout()
    return fig

def plot_chemical_space_overlap(data, smiles_col, splitter, n_folds_to_plot=1):
    """
    Calculates the maximum Tanimoto similarity of each test compound to the training set.
    """

    splits = list(splitter.split())
    n_plots = min(len(splits), n_folds_to_plot)
    
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    if n_plots == 1: axes = [axes]

    # Pre-calculate fingerprints for whole dataset to save time
    mols = [Chem.MolFromSmiles(s) for s in data[smiles_col]]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in mols if m]
    valid_idxs = [i for i, m in enumerate(mols) if m]
    
    # Map original index to fp index
    idx_map = {orig: i for i, orig in enumerate(valid_idxs)}

    for i in range(n_plots):
        train_idx, test_idx = splits[i]
        
        max_sims = []
        
        # Sadece geçerli parmak izi olanlar
        train_fps = [fps[idx_map[idx]] for idx in train_idx if idx in idx_map]
        test_fps = [fps[idx_map[idx]] for idx in test_idx if idx in idx_map]
        
        if not train_fps or not test_fps:
            continue

        # Her test molekülü için train setindeki EN BENZER molekülü bul
        for test_fp in test_fps:
            sims = DataStructs.BulkTanimotoSimilarity(test_fp, train_fps)
            max_sims.append(max(sims))
            
        sns.histplot(max_sims, ax=axes[i], bins=20, kde=True, color='green')
        axes[i].set_title(f"Fold {i+1}: Test vs Train Similarity")
        axes[i].set_xlabel("Max Tanimoto Similarity to Train Set")
        axes[i].set_ylabel("Count")
        
        # Referans çizgiler
        axes[i].axvline(x=0.4, color='red', linestyle='--', label='Low Sim (<0.4)')
        axes[i].legend()

    plt.tight_layout()
    return fig
def plot_qq_residuals(metrics_df, model_col = 'model', metric_col = 'metric'):
    models = metrics_df[model_col].unique()
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=True)
    if n_models == 1: axes = [axes]
    for i, model in enumerate(models):
        subset = metrics_df[metrics_df[model_col] == model][metric_col]
        residuals = (subset - subset.mean()) / subset.std()
        stats.probplot(residuals, dist="norm", plot=axes[i])
        axes[i].set_title(f"Q-Q Plot: {model}")
        axes[i].get_lines()[0].set_markerfacecolor('blue')
        axes[i].get_lines()[0].set_markeredgewidth(0)
        axes[i].get_lines()[1].set_color('red')
    plt.tight_layout()
    return fig