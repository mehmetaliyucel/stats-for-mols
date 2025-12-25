import pingouin as pg
import numpy as np
import pandas as pd

def run_parametric_tests(df, model_col, score_col, subject_col='fold'):
    """
    Runs Repeated Measures ANOVA.
    Handles Zero-Variance (identical scores) cases gracefully.
    """
    results = {}
    
    # --- GÜVENLİK KONTROLÜ: SIFIR VARYANS ---
    # Eğer tüm skorlar aynıysa (örn: herkes 1.0 aldıysa), test yapmaya gerek yok.
    if df[score_col].std() == 0:
        print(f"  -> Warning: Constant data detected for {score_col}. Skipping ANOVA.")
        results['test_name'] = 'Identical Data (No Variance)'
        results['global_pvalue'] = 1.0 # Fark yok -> Null hipotez kabul
        results['anova_table'] = None
        results['pairwise'] = None
        return results
    # ----------------------------------------

    try:
        # 1. Repeated Measures ANOVA (Global Test)
        aov = pg.rm_anova(dv=score_col, within=model_col, subject=subject_col, data=df, detailed=True)
        p_value_global = aov.loc[0, 'p-unc']
        
        results['test_name'] = 'Repeated Measures ANOVA'
        results['global_pvalue'] = p_value_global
        results['anova_table'] = aov
        
        # 2. Post-hoc Tests
        n_models = df[model_col].nunique()
        
        if n_models > 10:
            # FDR Correction for many models
            pairwise = pg.pairwise_tests(data=df, dv=score_col, within=model_col, 
                                         subject=subject_col, padjust='fdr_bh')
            # Görselleştirici uyumu için isim değişikliği
            if 'p-corr' in pairwise.columns:
                pairwise = pairwise.rename(columns={'p-corr': 'p-tukey'})
            elif 'p-adjust' in pairwise.columns:
                pairwise = pairwise.rename(columns={'p-adjust': 'p-tukey'})
        else:
            # Tukey HSD (Pingouin pairwise_tukey 'between' ister)
            pairwise = pg.pairwise_tukey(data=df, dv=score_col, between=model_col)
        
        results['pairwise'] = pairwise

    except Exception as e:
        print(f"  -> Error in Parametric Test for {score_col}: {e}")
        results['test_name'] = 'Failed'
        results['global_pvalue'] = 1.0 # Hata durumunda varsayılan
        results['pairwise'] = None

    return results