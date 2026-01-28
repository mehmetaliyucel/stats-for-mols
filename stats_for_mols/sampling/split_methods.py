import numpy as np
from sklearn.model_selection import (RepeatedKFold, RepeatedStratifiedKFold,
                                     ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold, KFold)
from sklearn.utils import check_random_state
from .strategy_selector import SamplingStrategySelector
from .splitting_strategies import ScaffoldRepeatedKFold, SplitStrategy, ButinaClusterKFold, UMAPRepeatedKFold


class DataSplitter:
    """
    A class to handle various data splitting strategies for machine learning tasks.
    """
    def __init__(self, data, target_col,split_type ='random', task_type='classification', random_state=42):
        self.data = data
        self.target = data[target_col].values
        self.n_samples = len(data)
        self.split_type = split_type
        self.task_type = task_type
        self.random_state = check_random_state(random_state)
        

        selector = SamplingStrategySelector(self.n_samples)
        self.config = selector.get_strategy()

    def split(self):
        method_name = self.config['method_name']
        n_splits = self.config['n_splits']
        n_repeats = self.config['n_repeats']


        print(f"Using {method_name} with {n_splits} splits and {n_repeats} repeats.")

        if self.split_type == 'random':
            if method_name == 'repeated_cv':
                if self.task_type == "classification":
                    cv = RepeatedStratifiedKFold(n_splits=n_splits,
                                                n_repeats=n_repeats,
                                                random_state=self.random_state)
                else:
                    cv = RepeatedKFold(n_splits=n_splits,
                                    n_repeats=n_repeats,
                                    random_state=self.random_state,)
                yield from cv.split(self.data, self.target)
            elif method_name == 'single_split':
                if self.task_type == "classification":
                    cv = StratifiedShuffleSplit(n_splits=n_splits,
                                                test_size=0.2,
                                                random_state=self.random_state)
                else:
                    cv = ShuffleSplit(n_splits=n_splits,
                                    test_size=0.2,
                                    random_state=self.random_state)
                yield from cv.split(self.data, self.target)
            elif method_name == 'nested_cv':
                if self.task_type == "classification":
                    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
                    inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

                    for train_val_idx, test_idx in outer_cv.split(self.data, self.target):
                        train_val_data = self.data.iloc[train_val_idx]
                        train_val_target = self.target[train_val_idx]


                        for train_idx, val_idx in inner_cv.split(train_val_data, train_val_target):
                            yield (train_val_idx[train_idx], train_val_idx[val_idx], test_idx)
                else:
                    outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
                    inner_cv = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
                    
                    for train_val_idx, test_idx in outer_cv.split(self.data):
                        train_val_data = self.data.iloc[train_val_idx]


                        for train_idx, val_idx in inner_cv.split(train_val_data):
                            yield (train_val_idx[train_idx], train_val_idx[val_idx], test_idx)
            elif method_name == 'light_nested_cv':
                if self.task_type ==  "classification":
                    train_test = RepeatedStratifiedKFold(n_splits=n_splits,
                                                  n_repeats=n_repeats,
                                                  random_state=self.random_state)
                    

                    single = StratifiedShuffleSplit(n_splits=1,  random_state=self.random_state, test_size=0.2)

                    for train_idx, test_idx in train_test.split(self.data, self.target):
                        for train_idx_inner, val_idx_inner in single.split(self.data.iloc[train_idx], self.target[train_idx]):
                            yield (train_idx[train_idx_inner], train_idx[val_idx_inner], test_idx)
                else:
                    train_test = RepeatedKFold(n_splits=n_splits,
                                            n_repeats=n_repeats,
                                            random_state=self.random_state)
                    single = ShuffleSplit(n_splits=1, random_state=self.random_state, test_size=0.2)
                    for train_idx, test_idx in train_test.split(self.data):
                        for train_idx_inner, val_idx_inner in single.split(self.data.iloc[train_idx]):
                            yield (train_idx[train_idx_inner], train_idx[val_idx_inner], test_idx)
        elif self.split_type == 'scaffold':
            if method_name == 'repeated_cv':
                splitter = ScaffoldRepeatedKFold(n_splits=n_splits,
                                                n_repeats=n_repeats,
                                                random_state=self.random_state)
                return splitter.split(self.data)
            elif method_name == 'single_split':
                splitter = ScaffoldRepeatedKFold(n_splits=n_splits,
                                                n_repeats=1,
                                                random_state=self.random_state)
                return splitter.split(self.data)
            else:
                raise ValueError(f"Unknown method_name: {method_name}")
        elif self.split_type == 'butina':
            if method_name == 'repeated_cv':
                splitter = ButinaClusterKFold(n_splits=n_splits,
                                              n_repeats=n_repeats,
                                              random_state=self.random_state)
                return splitter.split(self.data)
            elif method_name == 'single_split':
                splitter = ButinaClusterKFold(n_splits=n_splits,
                                              n_repeats=1,
                                              random_state=self.random_state)
                return splitter.split(self.data)
            else:
                raise ValueError(f"Unknown method_name: {method_name}")
        elif self.split_type == 'umap':
            if method_name == 'repeated_cv':
                splitter = UMAPRepeatedKFold(n_splits=n_splits,
                                             n_repeats=n_repeats,
                                             random_state=self.random_state)
                return splitter.split(self.data)
            elif method_name == 'single_split':
                splitter = UMAPRepeatedKFold(n_splits=n_splits,
                                             n_repeats=1,
                                             random_state=self.random_state)
                return splitter.split(self.data)
            else:
                raise ValueError(f"Unknown method_name: {method_name}")
        else:
            raise ValueError(f"Unknown split_type: {self.split_type}")

        