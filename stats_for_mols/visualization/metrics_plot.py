import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix
from math import pi

class MetricVisualizer:
    """
    Comprehensive plotting tools for regression, classification, and ranking tasks.
    """


    @staticmethod
    def plot_regression_scatter(y_true, y_pred, model_name="Model", ax=None):
        """
        Plots Predicted vs Actual values with an identity line (y=x).
        Includes error margins (2-fold error lines) common in drug discovery.
        Reference: Figure 4 in paper
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        # Scatter points
        sns.scatterplot(x=y_true, y=y_pred, alpha=0.5, ax=ax, label=model_name)
        
        # Identity line (Perfect prediction)
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Fit')

        # +/- 0.3 log units lines (approx 2-fold error) - Optional but useful for log data
        # Assuming data might be log-scaled (common in pIC50, logS)
        ax.plot([min_val, max_val], [min_val - 0.3, max_val - 0.3], 'r:', alpha=0.5, label='2-fold error')
        ax.plot([min_val, max_val], [min_val + 0.3, max_val + 0.3], 'r:', alpha=0.5)

        ax.set_xlabel("Measured / Experimental Value")
        ax.set_ylabel("Predicted Value")
        ax.set_title(f"{model_name}: Predicted vs Actual")
        ax.legend()
        return ax

    @staticmethod
    def plot_residuals(y_true, y_pred, ax=None):
        """
        Plots Residuals (True - Pred) vs Predicted values.
        Helps diagnose systematic errors (heteroscedasticity).
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
            
        residuals = y_true - y_pred
        sns.scatterplot(x=y_pred, y=residuals, alpha=0.5, ax=ax)
        ax.axhline(0, color='black', linestyle='--')
        ax.set_xlabel("Predicted Value")
        ax.set_ylabel("Residuals (True - Pred)")
        ax.set_title("Residual Plot")
        return ax

    # ---------------------------------------------------------
    # 2. CLASSIFICATION PLOTS
    # ---------------------------------------------------------
    @staticmethod
    def plot_roc_pr_curves(results_dict):
        """
        Plots both ROC and Precision-Recall curves side-by-side.
        Useful for imbalanced datasets.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        palette = sns.color_palette('husl', len(results_dict))

        for i, (name, res) in enumerate(results_dict.items()):
            y_true = res['y_true']
            y_proba = res['y_proba']
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            ax1.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})', color=palette[i])
        ax1.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.7)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.5)
        for i, (name, res) in enumerate(results_dict.items()):
            y_true = res['y_true']
            y_proba = res['y_proba']
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            pr_auc = auc(recall, precision)
            ax2.plot(recall, precision, label=f'{name} (AUC = {pr_auc:.2f})', color=palette[i])
        baseline = np.mean(y_true)
        ax2.axhline(baseline, color='r', linestyle='--', label=f'Baseline (Prev={baseline:.2f})')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.5)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred_binary, ax=None):
        """
        Visualizes the Confusion Matrix.
        Reference: 
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 4))
            
        cm = confusion_matrix(y_true, y_pred_binary)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        return ax


    @staticmethod
    def plot_enrichment_curve(results_dict, ax=None):
        """
        Plots Enrichment Factor curves for multiple models on the same graph.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        palette = sns.color_palette("husl", len(results_dict))
        fractions = np.linspace(0.01, 1.0, 100)

        for i, (name, res) in enumerate(results_dict.items()):
            y_true = np.array(res['y_true'])
            y_proba = np.array(res['y_proba'])
            
            # Sorting
            df = pd.DataFrame({'y_true': y_true, 'y_proba': y_proba})
            df = df.sort_values('y_proba', ascending=False)
            
            total_actives = df['y_true'].sum()
            total_compounds = len(df)
            base_rate = total_actives / total_compounds
            
            enrichments = []
            
            for f in fractions:
                top_n = int(f * total_compounds)
                if top_n == 0: 
                    enrichments.append(0)
                    continue
                    
                subset = df.head(top_n)
                found_actives = subset['y_true'].sum()
                precision = found_actives / top_n
                
                ef = precision / base_rate if base_rate > 0 else 0
                enrichments.append(ef)
                
            ax.plot(fractions * 100, enrichments, color=palette[i], lw=2, label=name)

        ax.axhline(1, color='k', linestyle='--', label='Random (EF=1)')
        ax.set_xlabel('Top % of Library Screened')
        ax.set_ylabel('Enrichment Factor (EF)')
        ax.set_title('Enrichment Factor Comparison')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        return ax

    # ---------------------------------------------------------
    # 4. HOLISTIC COMPARISON (RADAR CHART)
    # ---------------------------------------------------------
    @staticmethod
    def plot_radar_summary(metrics_df, metrics_to_plot=None):
        """
        Compares multiple models across multiple normalized metrics using a Radar Chart.
        Reference: Holistic Evaluation.
        
        metrics_df: DataFrame where index is Model Name, columns are metrics (R2, RMSE, etc.)
        metrics_to_plot: List of metrics to include in the radar chart.
        """
        if metrics_to_plot is None:
            metrics_to_plot = metrics_df.columns.tolist()
            
        # Normalize metrics to [0, 1] range for fair comparison
        # Note: For error metrics (RMSE, MAE), lower is better. We invert them: 1 - normalized_val
        df_norm = metrics_df[metrics_to_plot].copy()
        
        for col in df_norm.columns:
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            norm = (df_norm[col] - min_val) / (max_val - min_val) if max_val > min_val else 0
            
            # Invert "Lower is Better" metrics
            if col.lower() in ['rmse', 'mae', 'mse', 'log_loss']:
                df_norm[col] = 1 - norm
            else:
                df_norm[col] = norm

        # Radar Chart Logic
        categories = list(df_norm.columns)
        N = len(categories)
        
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1] # Close the loop
        
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
        
        for model_name, row in df_norm.iterrows():
            values = row.values.flatten().tolist()
            values += values[:1] # Close the loop
            ax.plot(angles, values, linewidth=1, linestyle='solid', label=model_name)
            ax.fill(angles, values, alpha=0.1)
            
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_title("Holistic Model Comparison (Normalized)")
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        
        return fig