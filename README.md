# Scale-Adaptive Piecewise-Quadratic Rectifiers for Rare-Event Genomic and Multi-Domain Screening (v4.0.0)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![PyTorch 2.1+ CUDA](https://img.shields.io/badge/PyTorch-2.1%2B%20(CUDA)-ee4c2c.svg)](https://pytorch.org/)

Official PyTorch implementation, empirical benchmark datasets, per-seed confusion logs, and automated mathematical verification suite for the manuscript:

> **Scale-Adaptive Piecewise-Quadratic Rectifiers for Rare-Event Genomic and Multi-Domain Screening: Offset-Scale Equivalence, Mechanistic Diagnostics, and Reproducible Computational Validation**  
> *Preecha Noiumkar and Thawatchai Chomsiri*  
> Research Center of Information Technology for the Future, Department of Information Technology, Faculty of Informatics, Mahasarakham University, Thailand.

---

## 1. Overview & Key Contributions

Piecewise-quadratic $C^1$-continuous rectifiers replace ReLU's non-differentiable kink with a parabolic transition. While symmetric **SmeLU** centers this transition on $[-\beta, +\beta]$, **Smooth RectLU** and **AS-RectLU** anchor the transition asymmetrically on $[L, L+W]$ with default $L=0, W=2$.

This repository provides theoretical and empirical proof that:
1. **Exact Offset Equivalence:** Smooth RectLU satisfies the algebraic identity $f_S(x; L, W) = \text{SmeLU}_{\beta=W/2}(x - L - W/2)$. Initializing at $L=0, W=2$ imposes an implicit negative bias offset ($\Delta b = -1$), zero origin slope ($f'(0^+)=0$), and severe forward second-moment contraction ($\mathbb{E}[f^2]/\mathbb{E}[x^2] = 0.0874$).
2. **Synergistic Stagnation Mechanism ($\text{H1} \times \text{H2}$):** Majority-class collapse under severe class imbalance ($10:1$ to $200:1$) arises from the interaction between negative initialization offset, small relative pre-activation scale ($\sigma_l / W \ll 1$), and imbalance-driven negative drift.
3. **Symmetric Per-Sample $\sigma$-AS-RectLU:** Dynamically locking transition width to each sample's root-mean-square scale via $W_l^{(i)} = \kappa\,\text{stopgrad}(\hat{\sigma}_{\text{RMS}}(x_l^{(i)}))$ and $L_l^{(i)} = -W_l^{(i)}/2$ restores degree-1 positive homogeneity ($f(\alpha x) = \alpha f(x)$), eliminates stagnation across 4-layer CNNs, Clair3-style PileupNets, and DeepVariant-style Inception backbones, and reduces GPU training time by **33%** compared to differentiable RMS scaling or explicit LayerNorm.

---

## 2. Repository Structure

- `pyproject.toml` : PEP 621 installable package specification (E12)
- `README.md` : Package documentation and reproduction guide
- `LICENSE` : MIT open-source license
- `configs/` : Hyperparameter & grid experiment configurations
- `src/` : Core PyTorch activation & GIAB extraction modules
- `experiments/` : Empirical benchmark scripts (`step01` to `step09`)
- `data/processed_giab/` : Shortcut-controlled GIAB 1D Unit-L2 & metadata
- `results/` : Per-seed logs and aggregated summary tables (`.csv`)
- `notebooks/` : Phase-plane & diagnostic visualization notebooks
- `manuscript/` : Unified LaTeX manuscript (`main_unified.tex`)

---

## 3. Installation & Automated Verification (E12)

    git clone https://github.com/preecha-msu/rectlu-genomic-screening.git
    cd rectlu-genomic-screening
    pip install -e .

Run the automated mathematical identity and GPU reproducibility unit tests:

    python -c "from experiments.step09_pipeline_benchmark_and_repro import run_software_reproducibility_audit; run_software_reproducibility_audit()"

**Expected Output:**

    [Unit Test 1] Algebraic Identity f_S(x;0,2) == SmeLU_1(x-1)     | Max Abs Error: 0.00e+00 -> PASS
    [Unit Test 2] Positive Homogeneity f(a*x) == a*f(x) (a=3.75)    | Max Abs Error: 1.91e-06 -> PASS
    [Unit Test 3] Compensated Bias (b0=+1) Backward Grad Equivalence | Max Grad Diff: 0.00e+00 -> PASS
    [Unit Test 4] Deterministic GPU Forward Reproducibility          | Max Logit Diff: 0.00e+00 -> PASS

---

## 4. Reproducing All Manuscript Experiments (E1-E12)

    python experiments/step07_real_giab_eval.py
    python experiments/step08_reviewer_rebuttal_suite.py
    python experiments/step09_pipeline_benchmark_and_repro.py

---

## 5. Acknowledgments
This research project was financially supported by the Faculty of Informatics, Mahasarakham University.
