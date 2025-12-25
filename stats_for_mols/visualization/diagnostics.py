# drug_stats/visualization/diagnostics.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from ..sampling.split_methods import DataSplitter
from scipy.stats import stats

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