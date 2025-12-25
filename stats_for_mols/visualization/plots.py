# drug_stats/visualization/plots.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import norm
def _get_p_col(columns):
    """
    Given a list of columns from pairwise test results,
    returns the appropriate p-value column name.
    """
    p_col_candidates = ['p-tukey', 'p-unc', 'p-val', 'p-conover', 'p-corr', 'p-adjust','p']
    for col in p_col_candidates:
        if col in columns:
            return col
    raise ValueError("No recognized p-value column found in pairwise results.")
def plot_simultaneous_ci(pairwise_df, metric_name="Score", higher_is_better=False):
    """
    Plots confidence intervals of performance differences relative to the best model.
    Handles missing 'se', empty data, and directionality.
    """
    if pairwise_df is None or pairwise_df.empty:
        print("Warning: Pairwise dataframe is empty. Cannot plot CI.")
        return
    

    # 1. Modelleri ve Ortalamalarını Çıkar
    # Parametrik veya Non-parametrik testten gelen tabloya göre kolon isimlerini garantiye alalım
    if 'A' not in pairwise_df.columns or 'B' not in pairwise_df.columns:
         print("Error: Pairwise dataframe missing 'A' or 'B' columns.")
         return
    p_col = _get_p_col(pairwise_df.columns)
    if p_col is None:
        print("Error: No p-value column found in pairwise dataframe.")
        return
    print(f"DEBUG: Using p-value column '{p_col}' for significance testing.")
    models = list(set(pairwise_df['A']).union(set(pairwise_df['B'])))
    model_means = {}
    
    # Her modelin ortalamasını bul
    for m in models:
        # Modelin 'A' olduğu satırlara bak
        row_a = pairwise_df[pairwise_df['A'] == m]
        if not row_a.empty:
            model_means[m] = row_a.iloc[0]['mean(A)']
        else:
            # Modelin 'B' olduğu satırlara bak
            row_b = pairwise_df[pairwise_df['B'] == m]
            if not row_b.empty:
                model_means[m] = row_b.iloc[0]['mean(B)']
            else:
                model_means[m] = -np.inf if higher_is_better else np.inf # Veri yoksa saf dışı bırak

    # En iyi modeli seç
    if not model_means:
        print("Error: Could not determine model means.")
        return

    if higher_is_better:
        best_model = max(model_means, key=model_means.get)
    else:
        best_model = min(model_means, key=model_means.get)
    
    print(f"DEBUG: Best Model identified as '{best_model}' based on {metric_name}")

    plot_data = []
    for m in models:
        if m == best_model:
            continue
            
        # Best vs M satırını bul (Sırası A-B veya B-A olabilir)
        row = pairwise_df[((pairwise_df['A'] == best_model) & (pairwise_df['B'] == m)) | 
                          ((pairwise_df['A'] == m) & (pairwise_df['B'] == best_model))]
        
        if row.empty:
            continue
            
        row = row.iloc[0]
        
        # Fark ve P-değeri
        diff = row['diff']
        p_val = row[p_col] 
        
        # SE (Standard Error) Tahmini
        if 'se' in row:
            se = row['se']
        else:
            # P-değerinden SE türetme (Non-parametrik durumlar için)
            if p_val >= 0.999: se = 0
            else:
                z_score = norm.ppf(1 - p_val / 2)
                se = abs(diff / z_score) if z_score != 0 else 0

        # Güven Aralığı
        ci_lower = diff - 1.96 * se
        ci_upper = diff + 1.96 * se
        
        is_significant = p_val < 0.05
        color = 'red' if is_significant else 'grey'
        
        # Görselleştirme için metin
        comp_label = f"{m} vs {best_model}"
        
        plot_data.append({
            'Comparison': comp_label,
            'Mean Difference': diff,
            'Lower': ci_lower,
            'Upper': ci_upper,
            'Significant': is_significant,
            'Color': color
        })
        
    if not plot_data:
        print("Warning: No comparison data generated for plotting. (Are models named correctly?)")
        return

    # DataFrame oluştururken kolonları elle belirtiyoruz (KeyError önlemi)
    df_plot = pd.DataFrame(plot_data, columns=['Comparison', 'Mean Difference', 'Lower', 'Upper', 'Significant', 'Color'])
    
    plt.figure(figsize=(10, len(plot_data) * 0.8 + 2))
    
    # Error Bar Plot
    for i, row in df_plot.iterrows():
        err_low = abs(row['Mean Difference'] - row['Lower'])
        err_high = abs(row['Upper'] - row['Mean Difference'])
        
        plt.errorbar(x=row['Mean Difference'], y=i, 
                     xerr=[[err_low], [err_high]],
                     fmt='o', color=row['Color'], capsize=5)
        
    plt.axvline(x=0, color='black', linestyle='--')
    plt.yticks(range(len(df_plot)), df_plot['Comparison'])
    plt.xlabel(f"Mean Difference in {metric_name} (Estimated CI)")
    plt.title(f"Simultaneous Confidence Intervals (vs {best_model})")
    plt.grid(axis='x', linestyle=':', alpha=0.7)
    
    return plt.gcf()

def plot_mcsim(pairwise_df, metric_name="Score"):
    """
    Generates the Multiple Comparisons Similarity (MCSim) Plot (Figure 8).
    Heatmap of Mean Differences with stars for statistical significance.
    """
    # 1. Pivot tablosu oluştur: Index=Model A, Columns=Model B, Values=Diff
    # Önce tam bir matris oluşturmamız lazım
    models = sorted(list(set(pairwise_df['A']).union(set(pairwise_df['B']))))
    n = len(models)
    p_col = _get_p_col(pairwise_df.columns)
    if p_col is None:
        print("Error: No p-value column found in pairwise dataframe.")
        return
    print(f"DEBUG: Using p-value column '{p_col}' for significance testing.")

    diff_matrix = pd.DataFrame(np.zeros((n, n)), index=models, columns=models)
    p_matrix = pd.DataFrame(np.ones((n, n)), index=models, columns=models)
    
    for _, row in pairwise_df.iterrows():
        m1, m2 = row['A'], row['B']
        diff = row['diff']
        p = row[p_col]
        
        # Simetrik doldur
        diff_matrix.loc[m1, m2] = diff
        diff_matrix.loc[m2, m1] = -diff # Fark tersine döner
        
        p_matrix.loc[m1, m2] = p
        p_matrix.loc[m2, m1] = p

    # 2. Yıldızlar (*, **, ***)
    def get_star(p):
        if p < 0.001: return '***'
        elif p < 0.01: return '**'
        elif p < 0.05: return '*'
        else: return ''
        
    annot_matrix = p_matrix.applymap(get_star)
    

    
    plt.figure(figsize=(8, 6))
    sns.heatmap(diff_matrix, annot=annot_matrix, fmt='', linewidths=0.5, linecolor='gray',
                cmap="RdBu_r", center=0, cbar_kws={'label': f'Difference in {metric_name}'})
    
    plt.title(f"MCSim Plot {metric_name}\n Effect Size & Significance")
    plt.tight_layout()
    
    return plt.gcf()