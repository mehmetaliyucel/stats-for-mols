# drug_stats/statistics/nonparametric.py
import pingouin as pg
import scikit_posthocs as sp
import pandas as pd

def run_nonparametric_tests(df, model_col, score_col, subject_col='fold'):
    """
    Runs non-parametric statistical tests on the provided DataFrame.
    if n_models < 10: Uses Conover with Holm-Bonferroni post-hoc test (Family-wise error control)
    if n_models >10; Uses Conover with Benjamini-Hochberg correction (Less conservative for many comparisons)
    References:
    supplementary material of:Practically Significant Method Comparison Protocols for Machine
    Learning in Small Molecule Drug Discovery
    """
    results = {}
    

    friedman = pg.friedman(dv=score_col, within=model_col, subject=subject_col, data=df)
    p_value_global = friedman.loc['Friedman', 'p-unc']
    
    results['test_name'] = 'Friedman Test'
    results['global_pvalue'] = p_value_global
    results['friedman_table'] = friedman
    n_models = df[model_col].nunique()
    if n_models > 10:
        print(f'INFO: More than 10 models ({n_models}) detected. Using Conover with Benjamini-Hochberg correction.')
        # Conover with Benjamini-Hochberg correction
        sp_method = 'fdr_bh'
    else:
        print(f'INFO: Using Conover post-hoc test without correction.')
        sp_method = None  # No correction

    
    # scikit-posthocs expects a wide format or melted data. 
    # posthoc_conover returns a P-value matrix.
    p_matrix = sp.posthoc_conover(df, val_col=score_col, group_col=model_col, p_adjust=sp_method)
    

    pairwise_list = []
    models = p_matrix.columns.tolist()
    import itertools
    

    means = df.groupby(model_col)[score_col].mean()
    
    for m1, m2 in itertools.combinations(models, 2):
        p_val = p_matrix.loc[m1, m2]
        
        pairwise_list.append({
            'A': m1,
            'B': m2,
            'mean(A)': means[m1],
            'mean(B)': means[m2],
            'diff': means[m1] - means[m2],
            'p-conover': p_val, 
            'test': 'Conover'
        })
        
    results['pairwise'] = pd.DataFrame(pairwise_list)
    
    return results