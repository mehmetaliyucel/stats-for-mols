# Sampling Strategy 

class SamplingStrategySelector:
    """
    According to dataset size, select appropriate sampling strategy.
    """
    def __init__(self, n_samples):
        self.n_samples = n_samples
        self.strategy_config = self._determine_strategy()


    def _determine_strategy(self):
        """
        Determine sampling strategy based on number of samples.
        """
        if self.n_samples >= 100000:
            return {
                "method_name": "single_split",
                "n_splits": 1,
                "n_repeats": 1,
                "description": "For large dataset (>=100k): Using single hold-out split."

            }
        
        elif 500<= self.n_samples < 100000:
            return {
                "method_name": "repeated_cv",
                "n_splits": 5,
                "n_repeats": 5,
                "description": "For standard dataset (500-100k): Using 5x5 repeated cross-validation."
            }
        else:
            return {
                "method_name": "repeated_cv",
                "n_splits": 2,
                "n_repeats": 5,
                "description": "For small dataset (<500): Using 2x5 repeated cross-validation."

            }

    def get_strategy(self):
        """
        Get the selected sampling strategy configuration.
        """
        return self.strategy_config