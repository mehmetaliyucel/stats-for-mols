import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    matthews_corrcoef, cohen_kappa_score, roc_auc_score,
    average_precision_score, precision_score, recall_score, f1_score, accuracy_score
)

class MetricCalculator:
    """
    A unified interface for calculating metrics tailored for drug discovery.
    Supports standard metrics and domain-specific ones (Enrichment, Recall@Precision, Top-k Ranking).
    """
    
    @staticmethod
    def get_regression_metrics(y_true, y_pred, top_k_fractions=[0.1, 0.2]):
        """
        Calculates regression metrics: MAE, RMSE, R2.
        Crucially, calculates Spearman and Kendall correlation globally AND for the top-k predictions.
        
        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.
            top_k_fractions: List of fractions (e.g., [0.1] for top 10%) to evaluate ranking.
        """
        # 1. Temel Hata Metrikleri
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # 2. Global Korelasyonlar
        pearson_val = pearsonr(y_true, y_pred)[0]
        spearman_val = spearmanr(y_true, y_pred)[0]
        kendall_val = kendalltau(y_true, y_pred)[0]
        
        metrics = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Pearson_Global": pearson_val,
            "Spearman_Global": spearman_val,
            "Kendall_Global": kendall_val
        }

        # 3. Top-k Ranking Metrics (Listenin tepesindeki sıralama başarısı)
        # Tahminlere göre sırala ve en yüksek skorlu %k'lık kısmı al
        df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
        df_sorted = df.sort_values(by='y_pred', ascending=False) # Büyükten küçüğe
        n = len(df)
        
        for k in top_k_fractions:
            top_n = int(n * k)
            if top_n < 5: 
                # Çok az veri varsa ranking metriği anlamsız olur
                continue
                
            subset = df_sorted.head(top_n)
            
            # Bu alt küme içinde sıralama ne kadar doğru?
            # Not: subset['y_pred'] zaten sıralı, ancak y_true'nun buna ne kadar uyduğunu ölçüyoruz.
            sp_k = spearmanr(subset['y_true'], subset['y_pred'])[0]
            kt_k = kendalltau(subset['y_true'], subset['y_pred'])[0]
            
            metrics[f"Spearman_Top{int(k*100)}%"] = sp_k
            metrics[f"Kendall_Top{int(k*100)}%"] = kt_k
            
        return metrics

    @staticmethod
    def get_classification_metrics(y_true, y_proba, threshold=0.5):
        """
        Calculates classification metrics including Enrichment Factors and Recall@k.
        """
        if y_proba.ndim  == 2 and y_proba.shape[1] > 1:
            return  MetricCalculator._get_multiclass_metrics(y_true, y_proba)
        else:
            return MetricCalculator._get_binary_metrics(y_true, y_proba, threshold)
    @staticmethod
    def _get_binary_metrics(y_true, y_proba, threshold):
        '''
        Binary classification metrics.
        
        '''
        metrics = {}
        if y_proba.ndim ==2:
            y_proba = y_proba[:, 1] if y_proba.shape[1] ==2 else y_proba.ravel()
        y_pred = (y_proba >= threshold).astype(int)
        try:
            metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
            metrics['Accuracy'] = accuracy_score(y_true, y_pred)
            metrics['Kappa'] = cohen_kappa_score(y_true, y_pred)
            metrics['F1_Score'] = f1_score(y_true, y_pred)
            metrics['ROC_AUC'] = roc_auc_score(y_true, y_proba)
            metrics['PR_AUC'] = average_precision_score(y_true, y_proba)
            metrics['Precision'] = precision_score(y_true, y_pred, zero_division=0)
            metrics['Recall'] = recall_score(y_true, y_pred)
            prevalence = np.mean(y_true)
            precision = precision_score(y_true, y_pred, zero_division=0)
            enrichment = precision / prevalence if prevalence > 0 else 0.0
            enrichment_at_1= MetricCalculator._enrichment_at_k(y_true, y_proba, k_percent=1)
            enrichment_at_5= MetricCalculator._enrichment_at_k(y_true, y_proba, k_percent=5)
            metrics['Enrichment_Factor'] = enrichment
            metrics['Enrichment@1%'] = enrichment_at_1
            metrics['Enrichment@5%'] = enrichment_at_5
            metrics['Recall@90%Prec'] = MetricCalculator._recall_at_precision(y_true, y_proba, target_precision=0.90)[f"Recall@90Prec"]
            metrics['TNR@90%Recall'] = MetricCalculator._tnr_at_recall(y_true, y_proba, target_recall=0.90)[f"TNR@90Recall"]

        except ValueError as e:
            print(f"Warning: Metric calculation failed with error: {e}")
                  
        return metrics
    @staticmethod
    def _get_multiclass_metrics(y_true, y_proba):
        '''
        Multiclass classification metrics,
        '''
        metrics = {}
        n_classes = y_proba.shape[1]
        y_pred = np.argmax(y_proba, axis=1)
        try:
            metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
            metrics['Accuracy'] = accuracy_score(y_true, y_pred)
            metrics['Kappa'] = cohen_kappa_score(y_true, y_pred)
            metrics['F1_Score_Macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
            metrics['F1_Score_Weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['Precision_Macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
            metrics['Recall_Macro'] = recall_score(y_true, y_pred, average='macro')
            metrics['Presicion_Weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['Recall_Weighted'] = recall_score(y_true, y_pred, average='weighted')
            metrics['ROC_AUC_Macro'] = roc_auc_score(y_true, y_proba, multi_class='ovo', average='macro')
            metrics['ROC_AUC_Weighted'] = roc_auc_score(y_true, y_proba, multi_class='ovo', average='weighted')
            metrics['PR_AUC_Macro'] = average_precision_score(y_true, y_proba, average='macro')
            metrics['PR_AUC_Weighted'] = average_precision_score(y_true, y_proba, average='weighted')
        except:
            print("Warning: Multiclass metric calculation failed.")
        return metrics

    @staticmethod
    def _enrichment_at_k(y_true, y_prob, k_percent=1):
        if k_percent <= 0:
            return 0.0
        if y_prob.ndim > 1 and y_prob.shape[1] > 1:
            n_classes = y_prob.shape[1]
            results = {}
            for class_idx in range(n_classes):
                y_true_binary = (y_true == class_idx).astype(int)
                y_prob_class = y_prob[:, class_idx]
                ef= MetricCalculator._binary_enrichment_at_k(y_true_binary, y_prob_class, k_percent)
                results[f'Enrichment_at_{k_percent}%_Class_{class_idx}'] = ef
            return results
        else:
            return MetricCalculator._binary_enrichment_at_k(y_true, y_prob, k_percent)
    @staticmethod
    def _binary_enrichment_at_k(y_true, y_prob, k_percent):
        n_total = len(y_true)
        n_positives = np.sum(y_true)
        if n_positives == 0:
            return 0.0
        df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob}) # Easy to follow indexing
        df_sorted = df.sort_values(by='y_prob', ascending=False)
        percent_k_to_take = k_percent / 100.0
        k = max(1, int(n_total * percent_k_to_take))
        top_k = df_sorted.head(k)['y_true'].sum()
        precision_sat_k = top_k / k
        prevalence = n_positives / n_total
        enrichment_factor = precision_sat_k / prevalence if prevalence > 0 else 0.0
        return enrichment_factor
    @staticmethod
    def _recall_at_precision(y_true, y_proba, target_precision):
        from sklearn.metrics import precision_recall_curve
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
        valid_indices = np.where(precisions[:-1] >= target_precision)[0]
        if len(valid_indices) == 0:
            return {f"Recall@{int(target_precision*100)}Prec": 0.0}
        return {f"Recall@{int(target_precision*100)}Prec": np.max(recalls[valid_indices])}

    @staticmethod
    def _tnr_at_recall(y_true, y_proba, target_recall):
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        valid_indices = np.where(tpr >= target_recall)[0]
        if len(valid_indices) == 0:
            return {f"TNR@{int(target_recall*100)}Recall": 0.0}
        return {f"TNR@{int(target_recall*100)}Recall": 1 - fpr[valid_indices[0]]}