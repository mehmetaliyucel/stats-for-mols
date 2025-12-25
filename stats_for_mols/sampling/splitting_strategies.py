# sampling/splitting_strategies.py
import numpy as np
from collections import defaultdict
from sklearn.model_selection import KFold, StratifiedKFold
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.cluster import KMeans
import umap

class SplitStrategy:
    def split(self, X, y=None, groups=None):
        raise NotImplementedError("Subclasses should implement this!")
    def _get_smiles_list(self, X):
        if hasattr(X, 'columns'):
            smiles_col = next((col for col in X.columns if col.lower() in ['smiles','canonical_smiles']), None)
            if not smiles_col:
                raise ValueError("No SMILES column found in DataFrame.")
        
            return X[smiles_col].values
        elif isinstance(X, list):
            return X
        return X
    def _calculate_fingerprints(self, smiles_list):
        mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        valid_idx = [i for i,mol in enumerate(mols) if mol is not None]
        fps = [AllChem.GetMorganFingerprintAsBitVect(mols[i], 2, nBits=2048) for i in valid_idx]
        return fps, valid_idx
class ScaffoldRepeatedKFold(SplitStrategy):
    """
    In every fold, it shuffles the scaffolds and assigns them to folds
    trying to keep the distribution of scaffolds across folds as even as possible.
    """
    def __init__(self, n_splits=5, n_repeats=5, random_state=None):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
    def _generate_scaffolds(self, smiles_list):
        scaffolds= defaultdict(list)
        for idx, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
                scaffolds[scaffold].append(idx)
            else:
                scaffolds['INVALID'].append(idx)
        return scaffolds
    def split(self, X, y= None, groups=None):
        """
        X: Pandas DataFrame or list-like of SMILES strings. if dataframe, it should contain smiles column or mol column.
        """

        smiles_list = self._get_smiles_list(X)
        scaffolds_dict = self._generate_scaffolds(smiles_list)
        scaffold_keys = list(scaffolds_dict.keys())
        if "INVALID" in scaffold_keys: scaffold_keys.remove("INVALID")  #might need better handling of invalid smiles
        for repeat in range(self.n_repeats):
            current_seed = self.random_state + repeat if self.random_state is not None else None
            rng_repeat = np.random.RandomState(current_seed)
            rng_repeat.shuffle(scaffold_keys)
            fold_scaffold_sets = np.array_split(scaffold_keys, self.n_splits)
            for fold_idx in range(self.n_splits):
                test_scaffolds = set(fold_scaffold_sets[fold_idx])
                train_indices, test_indices = [], []
                for scaffold, indices in scaffolds_dict.items():
                    if scaffold in test_scaffolds:
                        test_indices.extend(indices)
                    else:
                        train_indices.extend(indices)
                yield np.array(train_indices), np.array(test_indices)
class ButinaClusterKFold(SplitStrategy):
    """
    Clusters mols using Butina algorithm and assigns clusters to folds.
    """
    def __init__(self, n_splits, n_repeats=5, cutoff = 0.2, random_state=None):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.cutoff = cutoff
        self.random_state = random_state
    def split(self, X, y=None, groups=None):
        smiles_list = self._get_smiles_list(X)
        fps, valid_idx = self._calculate_fingerprints(smiles_list)
        dists = []
        n_fps = len(fps)
        for i in range(1, n_fps):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
            dists.extend([1 - x for x in sims])
        clusters = Butina.ClusterData(dists, n_fps, self.cutoff, isDistData=True)
        mapped_clusters = [ [valid_idx[idx] for idx in cluster] for cluster in clusters]
        for repeat in range(self.n_repeats):
            current_seed = self.random_state + repeat if self.random_state is not None else None
            rng_repeat = np.random.RandomState(current_seed)
            clusters_copy = mapped_clusters.copy()
            rng_repeat.shuffle(clusters_copy)

            # Distribute clusters to folds
            #this is very primitive way to distribute clusters to folds
            fold_clusters = np.array_split(clusters_copy, self.n_splits)
            for fold_idx in range(self.n_splits):
                test_indices = []
                train_indices = []
                for i, cluster_in_fold in enumerate(fold_clusters):
                    flat_indices = [idx for cluster in cluster_in_fold for idx in cluster]
                    if i == fold_idx:
                        test_indices.extend(flat_indices)
                    else:
                        train_indices.extend(flat_indices)
                yield np.array(train_indices), np.array(test_indices)
class UMAPRepeatedKFold(SplitStrategy):
    """
    It reduces the dimensionality of the fingerprints using UMAP and then clusters them using KMeans.
    """
    def __init__(self, n_splits=5, n_repeats = 5 , random_state=42):
        self.n_splits = n_splits
        self.n_repeats = n_repeats

        self.random_state = random_state
        self.umap_module= umap
    def split(self, X, y=None, groups=None):
        smiles_list = self._get_smiles_list(X)
        fps, valid_idx = self._calculate_fingerprints(smiles_list)
        X_fp = []
        for fp in fps:
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            X_fp.append(arr)
        X_fp = np.vstack(X_fp)
        for repeat in range(self.n_repeats):
            current_seed = self.random_state + repeat if self.random_state is not None else None
            reducer = self.umap_module.UMAP(n_components=10, random_state=current_seed)
            X_umap = reducer.fit_transform(X_fp)
            kmeans = KMeans(n_clusters=self.n_splits, random_state=current_seed)
            labels = kmeans.fit_predict(X_umap)
            for fold_idx in range(self.n_splits):
                test_mask = (labels == fold_idx)
                train_mask = ~test_mask
                test_indices = np.array(valid_idx)[test_mask]
                train_indices = np.array(valid_idx)[train_mask]
                yield np.array(train_indices), np.array(test_indices)

    