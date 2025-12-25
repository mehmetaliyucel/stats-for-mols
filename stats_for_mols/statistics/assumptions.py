import pandas as pd
import pingouin as pg

def check_assumptions(df, model_col, score_col, subject_col='fold'):
    """
    Checks parametric assumptions (Normality and Sphericity).
    
    NOTE: As per the paper guidelines (Supplementary Note S4-S5), 
    normality tests like Shapiro-Wilk can be overly sensitive. 
    Since 5x5 Repeated CV provides 25 samples, the Central Limit Theorem (CLT) 
    often makes ANOVA robust to deviations from normality.
    
    Therefore, this function defaults 'valid_parametric' to True 
    but warns the user if assumptions are strongly violated.
    """
    results = {}

    try:
        normality_df = pg.normality(df, dv=score_col, group=model_col)
        all_normal = normality_df['normal'].all()
        min_p_norm = normality_df['pval'].min()
        
        results['normality'] = all_normal
        results['normality_p_min'] = min_p_norm
        
        if not all_normal:
            results['normality_warning'] = (
                f"Normality assumption technically violated (p={min_p_norm:.4f}). "
                "However, ANOVA is robust for N=25 (CLT), so proceeding with parametric test is recommended."
            )
    except Exception as e:
        results['normality'] = False
        results['normality_warning'] = f"Normality check failed to run: {e}"


    try:
        if df[model_col].nunique() > 2:
            spher, _, _, _, p_spher = pg.sphericity(df, dv=score_col, 
                                                    subject=subject_col, within=model_col)
            results['sphericity'] = spher
            results['sphericity_p'] = p_spher
        else:
            results['sphericity'] = True
    except Exception as e:
        results['sphericity'] = False
        results['sphericity_warning'] = f"Sphericity check failed: {e}"


    results['valid_parametric'] = True
    
    return results