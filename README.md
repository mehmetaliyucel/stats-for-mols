# stats-for-mols: Statistical Method Comparison for Drug Discovery

stats-for-mols is a Python library designed to implement the rigorous benchmarking protocols proposed in the paper ["Practically Significant Method Comparison Protocols for Machine Learning in Small Molecule Drug Discovery"](https://pubs.acs.org/doi/10.1021/acs.jcim.5c01609?ref=pdf).

This toolkit focuses on replicability, statistical rigor, and practical significance, decoupling the evaluation logic from the training loop to provide a flexible set of tools for any ML pipeline.
## 📦 Modules

The library consists of 4 main modules that follow the "Decision Tree" for method comparison:

* **Sampling** (`sampling`)
    * Implements domain-specific splitting strategies like Scaffold Split, Butina Clustering, and UMAP-based splits.
    * Automatically selects the appropriate Cross-Validation strategy (e.g., 5×5 Repeated CV) based on dataset size.

* **Performance** (`performance`)
    * Calculates standard metrics (RMSE, MCC, ROC-AUC) and domain-critical metrics.
    * Includes Enrichment Factors, Recall@Precision, TNR@Recall, and Top-k Ranking metrics (Spearman/Kendall).

* **Statistics** (`statistics`)
    * An automated statistical engine that checks parametric assumptions (Normality, Sphericity).
    * Selects the correct test: Repeated Measures ANOVA + Tukey HSD or Friedman + Conover/Benjamini-Hochberg.

* **Visualization** (`visualization`)
    * Generates publication-ready plots recommended by the guidelines:
        * MCSim Plots (Multiple Comparisons Similarity).
        * Simultaneous Confidence Intervals.
        * Enrichment Curves and Radar Charts for holistic evaluation.

## 🛠 Installation

### Option 1: Development (Recommended)
If you want to modify the code or run the tutorials:

```bash
# 1. Clone the repository
git clone https://github.com/mehmetaliyucel/stats-for-mols.git
cd stats-for-mols
```
# 2. Install in editable mode with uv
uv pip install -e .

### Option 2: Install as a Library

If you just want to use the package in another project:
```bash
uv pip install https://github.com/mehmetaliyucel/stats-for-mols.git
```
📄 Reference

If you use this package in your research, please cite the original paper:

    [Ash, J. R., Wognum, C., et al. "Practically Significant Method Comparison Protocols for Machine Learning in Small Molecule Drug Discovery." Journal of Chemical Information and Modeling (2025)](https://pubs.acs.org/doi/10.1021/acs.jcim.5c01609)