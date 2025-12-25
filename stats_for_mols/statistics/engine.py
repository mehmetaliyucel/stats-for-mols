# drug_stats/statistics/engine.py
from .assumptions import check_assumptions
from .parametric import run_parametric_tests
from .nonparametric import run_nonparametric_tests
import pandas as pd
class StatisticalValidator:
    """
    Orchestrates the statistical validation process defined in the paper.
    Automatically selects between Parametric and Non-Parametric tests.
    """
    def __init__(self, metrics_df, model_col='model', metric_col=None, subject_col='fold', alpha=0.05):
        """
        metrics_df: DataFrame with columns [model_col, metric_col, subject_col]
        """
        self.df = metrics_df.copy()
        self.model_col = model_col
        self.subject_col = subject_col
        self.alpha = alpha

        if isinstance(metric_col, str):
            self.metric_col = [metric_col]  # Return a list for uniform processing
        elif isinstance(metric_col, list):
            self.metric_col = metric_col
        elif metric_col is None:
            exclude_cols = [model_col, subject_col]
            self.metric_col = [col for col in self.df.columns if col not in exclude_cols]
        else:
            raise ValueError("metric_col must be a string, list of strings, or None.")
    def validate(self, force_parametric=False):
        """
        Executes the validation pipeline.
        
        Args:
            force_parametric (bool): If True, skips assumption checks and forces ANOVA.
                                     (Useful if user is confident or following Guidelines 2 strictly)
        Returns:
            dict: Comprehensive results dictionary.
        """
        final_report = {}
        for metric in self.metric_col:
            print(f"\nValidating Metric: {metric}")
            self.metric_col = metric  # Set current metric

            assumptions = check_assumptions(self.df, self.model_col, self.metric_col, self.subject_col)
            

            if assumptions['valid_parametric'] or force_parametric:
                print(f"Assumptions met (or forced). Running Parametric Tests (Repeated Measures ANOVA).")
                test_results = run_parametric_tests(self.df, self.model_col, self.metric_col, self.subject_col)
                method_used = "parametric"
            else:
                print(f"Assumptions violated (Normality={assumptions['normality']}, Sphericity={assumptions['sphericity']}). Running Non-Parametric Tests (Friedman).")
                test_results = run_nonparametric_tests(self.df, self.model_col, self.metric_col, self.subject_col)
                method_used = "nonparametric"

            final_report[metric] = {
                "assumptions": assumptions,
                "method_used": method_used,
                "global_test": test_results.get('test_name'),
                "global_pvalue": test_results.get('global_pvalue'),
                "pairwise_results": test_results.get('pairwise')
            }
        
        return final_report
    @staticmethod
    def save_report_excel(report,filename='statistical_report.xlsx'):
        """
        Saves the statistical report to an Excel file with multiple sheets.
        """
        summary_data = []
        for metric, data  in report.items():
            summary_data.append({
                'Metric': metric,
                'Method Used': data['method_used'],
                'Global Test': data['global_test'],
                'Global p-value': data['global_pvalue'],
                'Normality': data['assumptions']['normality'],
                'Sphericity': data['assumptions']['sphericity'],
            })
        summary_df = pd.DataFrame(summary_data)
        with pd.ExcelWriter(filename) as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            for metric, data in report.items():
                pairwise_df = pd.DataFrame(data['pairwise'])
            if pairwise_df is not None and not pairwise_df.empty: 
                pairwise_df.to_excel(writer, sheet_name=f'Pairwise_{metric}', index=False)
        print(f"Statistical report saved to {filename}")