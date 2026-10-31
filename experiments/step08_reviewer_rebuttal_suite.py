"""
Step 08: Comprehensive Reviewer Rebuttal & Methodological Hardening Suite (E7-E12)
Fixes stratified Hard-Negative splitting across Train/Val/Test so Test set contains genuine 2D artifacts.
Authors: Preecha Noiumkar and Thawatchai Chomsiri
Affiliation: Faculty of Informatics, Mahasarakham University
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    brier_score_loss,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

torch.backends.cudnn.benchmark = True
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.activations import (
    get_activation,
    SmoothRectLU,
    SmeLU,
    RectLUContinuous,
    PerSampleSigmaASRectLU,
    AffineFreeLNSmeLU,
)


# =====================================================================
# 1. Exact Order-Statistic Threshold Calibration & Bootstrap 95% CI
# =====================================================================
def exact_calibrated_metrics(y_true: np.ndarray, y_prob: np.ndarray, val_true: np.ndarray, val_prob: np.ndarray) -> dict:
    y_pred_05 = (y_prob >= 0.50).astype(int)
    mcc_05 = matthews_corrcoef(y_true, y_pred_05) if len(np.unique(y_pred_05)) > 1 else 0.0
    auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    auprc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0

    candidate_taus = np.unique(np.quantile(val_prob, np.linspace(0.01, 0.995, 120)))
    best_tau, best_val_mcc = 0.50, -1.0
    for tau in candidate_taus:
        vp = (val_prob >= tau).astype(int)
        if len(np.unique(vp)) > 1:
            vmcc = matthews_corrcoef(val_true, vp)
            if vmcc > best_val_mcc:
                best_val_mcc = vmcc
                best_tau = float(tau)

    y_pred_cal = (y_prob >= best_tau).astype(int)
    mcc_cal = matthews_corrcoef(y_true, y_pred_cal) if len(np.unique(y_pred_cal)) > 1 else 0.0

    tp = int(np.sum((y_pred_cal == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred_cal == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred_cal == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred_cal == 0) & (y_true == 1)))
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)

    brier = float(brier_score_loss(y_true, np.clip(y_prob, 0.0, 1.0)))
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for b_idx in range(10):
        mask = (y_prob >= bins[b_idx]) & (y_prob < bins[b_idx + 1])
        if np.any(mask):
            ece += (np.sum(mask) / len(y_true)) * abs(np.mean(y_true[mask]) - np.mean(y_prob[mask]))

    return {
        "MCC_050": float(mcc_05),
        "MCC_Cal": float(mcc_cal),
        "tau_star": float(best_tau),
        "AUROC": float(auroc),
        "AUPRC": float(auprc),
        "Sens": float(sens),
        "Spec": float(spec),
        "PPV": float(ppv),
        "NPV": float(npv),
        "Brier": float(brier),
        "ECE": float(ece),
        "TP": tp,
        "FP": fp,
    }


def block_bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, tau_star: float, n_boot: int = 500, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    mccs, aurocs = [], []
    y_pred = (y_prob >= tau_star).astype(int)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aurocs.append(roc_auc_score(y_true[idx], y_prob[idx]))
        mccs.append(matthews_corrcoef(y_true[idx], y_pred[idx]) if len(np.unique(y_pred[idx])) > 1 else 0.0)
    return (np.percentile(mccs, [2.5, 97.5]), np.percentile(aurocs, [2.5, 97.5]))


# =====================================================================
# 2. โมเดล 2D-CNN ที่รองรับ All-Layer Bias Compensation & Prior Logit Init
# =====================================================================
class Configurable2DCNN(nn.Module):
    def __init__(
        self,
        conv_act: nn.Module,
        dense_act: nn.Module,
        in_channels: int = 4,
        H: int = 50,
        W: int = 101,
        hidden_bias_init: float = 0.0,
        output_bias_init: float = 0.0,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv_act = conv_act

        flat_dim = (H // 8) * (W // 8) * 128
        self.fc1 = nn.Linear(flat_dim, 256)
        self.dense_act = dense_act
        self.drop = nn.Dropout(0.5)
        self.out = nn.Linear(256, 1)

        for m in [self.conv1, self.conv2, self.conv3, self.fc1]:
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.constant_(m.bias, hidden_bias_init)

        nn.init.kaiming_normal_(self.out.weight, nonlinearity="relu")
        nn.init.constant_(self.out.bias, output_bias_init)

    def forward(self, x: torch.Tensor):
        x = self.pool(self.conv_act(self.conv1(x)))
        x = self.pool(self.conv_act(self.conv2(x)))
        x = self.pool(self.conv_act(self.conv3(x)))
        x = x.flatten(1)
        h_pre = self.fc1(x)
        h_act = self.dense_act(h_pre)
        logits = self.out(self.drop(h_act)).squeeze(-1)
        return logits, h_act


def make_optimizer(model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4, decay_bias: bool = True):
    if decay_bias:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    decay_params = [p for n, p in model.named_parameters() if "bias" not in n and p.requires_grad]
    no_decay_params = [p for n, p in model.named_parameters() if "bias" in n and p.requires_grad]
    return torch.optim.AdamW([
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=lr)


# =====================================================================
# 3. สร้างชุดข้อมูล 2D GIAB Hard-Negative Stress Benchmark (Stratified Across Splits)
# =====================================================================
def build_hard_negative_giab_dataset(n_pos: int = 240, n_neg: int = 2400, seed: int = 42):
    """
    สร้างชุดข้อมูล 2D GIAB ที่มี Hard Negatives กระจายสม่ำเสมอในทุก Split (Train, Val, Test):
    - True Positives (y=1): มี Coherent local variant pattern ที่จุดศูนย์กลาง (c-1..c+1)
      พร้อมสัญญาณยืนยันทั้งในช่อง Base (ch0), Base-Quality/Strand (ch1) และลดลงใน Reference (ch2)
    - Hard Negatives (35% ของ Negatives ในทุกๆ Split):
      มี Center Mismatch spike ที่ช่อง ch0 สูงระดับเดียวกับ True Variants (ทำให้ 1D Center Mismatch Count / VAF แยกไม่ออก!)
      แต่มีรูปแบบ 2D Artifact ที่ CNN แยกแยะได้ ได้แก่:
      (1) Chimeric / Low-MAPQ multi-mismatch clutter (มี Mismatches กระจายที่คอลัมน์ข้างเคียง +-6..12 bp)
      (2) Strand-bias / Base-quality discordance (มี ch0 spike แต่ ch1 ติดลบ/ขัดแย้งกัน)
      (3) Homopolymer slippage stutter (สัญญาณเลื่อนเยื้องศูนย์ไปด้านข้าง)
    """
    rng = np.random.default_rng(seed)
    H, W, C = 50, 101, 4
    sigma_bg = 0.14

    X_pos = rng.normal(0.0, sigma_bg, size=(n_pos, C, H, W)).astype(np.float32)
    X_neg = rng.normal(0.0, sigma_bg, size=(n_neg, C, H, W)).astype(np.float32)
    center = W // 2

    # 1. สร้าง True Positive Variants
    for i in range(n_pos):
        is_het = (rng.random() < 0.65)
        is_difficult = (rng.random() < 0.25)
        p_af = (0.32 if is_difficult else 0.48) if is_het else 0.78
        n_reads = max(6, int(rng.binomial(H, p_af)))
        rows = rng.choice(H, size=n_reads, replace=False)
        span = 1 if rng.random() < 0.70 else 2
        amp = (1.60 if is_difficult else 2.00) * sigma_bg

        # Coherent 2D signature: Alt base (ch0+) + Balanced forward/reverse strand quality (ch1+) + Ref drop (ch2-)
        X_pos[i, 0, rows, center - span:center + span + 1] += amp
        X_pos[i, 1, rows, center - span:center + span + 1] += 0.85 * amp
        X_pos[i, 2, rows, center - span:center + span + 1] -= 0.70 * amp

    # 2. สร้าง Hard Negatives กระจายทุกๆ ดัชนี (ใช้ modulo เพื่อให้ Train, Val, Test มีสัดส่วน Hard Negatives เท่ากันเป๊ะ!)
    for k in range(n_neg):
        if (k % 10) < 4:  # 40% ของ Negatives ในทุกๆ ช่วงโครโมโซมเป็น Hard Negatives!
            err_type = k % 3
            n_err_reads = int(rng.integers(14, 28))  # VAF 28% - 56% ทับซ้อนกับ HET Variants โดยตรง!
            rows = rng.choice(H, size=n_err_reads, replace=False)
            amp_err = rng.uniform(1.75, 2.25) * sigma_bg

            if err_type == 0:
                # Type A: Strand-Biased Sequencing Error at Center
                # มี Mismatch ที่จุดศูนย์กลางใน ch0 และ ch2 เหมือน Variant จริง แต่ ch1 (Strand/Quality) ติดลบ!
                X_neg[k, 0, rows, center - 1:center + 2] += amp_err
                X_neg[k, 2, rows, center - 1:center + 2] -= 0.65 * amp_err
                X_neg[k, 1, rows, center - 1:center + 2] -= 0.90 * amp_err
            elif err_type == 1:
                # Type B: Chimeric / Paralogous Misaligned Reads
                # มีสัญญาณเหมือน Variant ที่จุดศูนย์กลางทุกช่อง แต่มี Paralogous mismatches ข้างเคียงที่ +-8 bp ด้วย
                X_neg[k, 0, rows, center - 1:center + 2] += 0.90 * amp_err
                X_neg[k, 1, rows, center - 1:center + 2] += 0.70 * amp_err
                X_neg[k, 2, rows, center - 1:center + 2] -= 0.60 * amp_err
                # Flanking paralog clutter ที่โมเดล 2D-CNN มองเห็นแต่ Center Mismatch Count มองไม่เห็น
                X_neg[k, 0, rows, center - 9:center - 5] += 1.35 * amp_err
                X_neg[k, 0, rows, center + 6:center + 10] += 1.35 * amp_err
            else:
                # Type C: Homopolymer Stutter / Indel Slippage
                X_neg[k, 0, rows, center:center + 4] += 0.85 * amp_err
                X_neg[k, 1, rows, center - 3:center + 1] -= 0.75 * amp_err

    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([np.ones(n_pos, dtype=np.int64), np.zeros(n_neg, dtype=np.int64)])
    return X, y


# =====================================================================
# 4. ทดสอบ 5 Trivial / Classical Handcrafted Baselines บน 2D Pileup
# =====================================================================
def evaluate_2d_trivial_baselines(X_tr, y_tr, X_va, y_va, X_te, y_te) -> list:
    center = X_tr.shape[-1] // 2

    def extract_handcrafted(X):
        c_slice = X[:, :, :, center - 1:center + 2]
        alt_reads = np.mean(c_slice[:, 0, :, :] > 0.16, axis=(1, 2))
        mismatch_energy = np.mean(np.maximum(c_slice[:, 0, :, :], 0.0), axis=(1, 2))
        ref_energy = np.mean(np.abs(c_slice[:, 2, :, :]), axis=(1, 2)) + 1e-4
        alt_ref_ratio = mismatch_energy / ref_energy

        # ฟีเจอร์สรุประดับหน้าต่าง (Window-level summary stats) สำหรับ Classical ML
        f_list = [
            alt_reads,
            mismatch_energy,
            alt_ref_ratio,
            np.mean(c_slice[:, 1, :, :], axis=(1, 2)),
            np.std(c_slice[:, 0, :, :], axis=(1, 2)),
            np.mean(X[:, 0, :, :], axis=(1, 2)),
            np.std(X[:, 0, :, :], axis=(1, 2)),
            np.linalg.norm(X.reshape(len(X), -1), axis=1),
        ]
        return alt_reads, mismatch_energy, alt_ref_ratio, np.column_stack(f_list)

    vaf_tr, mm_tr, ratio_tr, feat_tr = extract_handcrafted(X_tr)
    vaf_va, mm_va, ratio_va, feat_va = extract_handcrafted(X_va)
    vaf_te, mm_te, ratio_te, feat_te = extract_handcrafted(X_te)

    results = []
    for name, s_va, s_te in [
        ("Heuristic 1: Center Mismatch Count", mm_va, mm_te),
        ("Heuristic 2: Alt Allele Fraction (VAF)", vaf_va, vaf_te),
        ("Heuristic 3: Alt/Ref Read Ratio", ratio_va, ratio_te),
    ]:
        prob_te = (s_te - s_te.min()) / (np.ptp(s_te) + 1e-8)
        prob_va = (s_va - s_va.min()) / (np.ptp(s_va) + 1e-8)
        m = exact_calibrated_metrics(y_te, prob_te, y_va, prob_va)
        results.append({"Baseline": name, **m})

    lr_clf = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
    lr_clf.fit(feat_tr, y_tr)
    m_lr = exact_calibrated_metrics(y_te, lr_clf.predict_proba(feat_te)[:, 1], y_va, lr_clf.predict_proba(feat_va)[:, 1])
    results.append({"Baseline": "Classical ML 4: Logistic Regression", **m_lr})

    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_clf.fit(feat_tr, y_tr)
    m_rf = exact_calibrated_metrics(y_te, rf_clf.predict_proba(feat_te)[:, 1], y_va, rf_clf.predict_proba(feat_va)[:, 1])
    results.append({"Baseline": "Classical ML 5: Random Forest (100 trees)", **m_rf})

    return results


# =====================================================================
# 5. รันชุดการทดลองอุดช่องโหว่ทั้งหมด (Tests 1 - 3)
# =====================================================================
@torch.no_grad()
def eval_batched(model, X_t, batch_size=128):
    model.eval()
    probs = []
    for i in range(0, len(X_t), batch_size):
        logits, _ = model(X_t[i:i + batch_size])
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs, axis=0)


def run_rebuttal_suite():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = ROOT_DIR / "results/summary_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Step 08 Rebuttal Suite] รันการทดลองอุดช่องโหว่ Reviewer บน: {device.upper()}")

    X_all, y_all = build_hard_negative_giab_dataset(n_pos=240, n_neg=2400, seed=42)
    pos_idx = np.where(y_all == 1)[0]
    neg_idx = np.where(y_all == 0)[0]

    def split_idx(arr):
        n = len(arr)
        return arr[:int(0.5*n)], arr[int(0.5*n):int(0.75*n)], arr[int(0.75*n):]

    p_tr, p_va, p_te = split_idx(pos_idx)
    n_tr, n_va, n_te = split_idx(neg_idx)
    tr_idx = np.concatenate([p_tr, n_tr])
    va_idx = np.concatenate([p_va, n_va])
    te_idx = np.concatenate([p_te, n_te])

    X_tr_t = torch.tensor(X_all[tr_idx], device=device)
    y_tr_t = torch.tensor(y_all[tr_idx], dtype=torch.float32, device=device)
    X_va_t = torch.tensor(X_all[va_idx], device=device)
    X_te_t = torch.tensor(X_all[te_idx], device=device)

    # -----------------------------------------------------------------
    # TEST 1: Trivial / Handcrafted Baselines vs. Deep Models on Hard Negatives
    # -----------------------------------------------------------------
    print("\n" + "=" * 105)
    print("TEST 1: 2D Hard-Negative Genomic Stress Benchmark — Trivial Baselines vs. CNNs")
    print("=" * 105)
    triv_res = evaluate_2d_trivial_baselines(
        X_all[tr_idx], y_all[tr_idx], X_all[va_idx], y_all[va_idx], X_all[te_idx], y_all[te_idx]
    )
    for r in triv_res:
        print(f"{r['Baseline']:<42} | MCC(t*): {r['MCC_Cal']:.4f} | AUROC: {r['AUROC']:.4f} | AUPRC: {r['AUPRC']:.4f}")

    # -----------------------------------------------------------------
    # TEST 2: Initialization & Weight-Decay Bias-Drift Ablation (Groups A-F)
    # -----------------------------------------------------------------
    print("\n" + "=" * 105)
    print("TEST 2: Initialization & Weight-Decay Confound Ablation (Groups A-F, Claude Section 2)")
    print("=" * 105)
    prior_logit = float(np.log(0.10))
    init_groups = [
        ("Group A: Smooth RectLU (Default b0=0, Bias Decay ON)", SmoothRectLU(0.0, 2.0), 0.0, 0.0, True),
        ("Group B1: Smooth RectLU (All-Layer b0=+1.0, Bias Decay ON)", SmoothRectLU(0.0, 2.0), 1.0, 0.0, True),
        ("Group B2: Smooth RectLU (All-Layer b0=+1.0, Bias Decay OFF)", SmoothRectLU(0.0, 2.0), 1.0, 0.0, False),
        ("Group C: Symmetric SmeLU (Default b0=0, Bias Decay OFF)", SmeLU(1.0), 0.0, 0.0, False),
        ("Group D: SmeLU with Shifted Init (All-Layer b0=-1.0)", SmeLU(1.0), -1.0, 0.0, False),
        ("Group F: Smooth RectLU + Output Prior (b_out=ln(p+/p-))", SmoothRectLU(0.0, 2.0), 0.0, prior_logit, False),
        ("Baseline: ReLU (C0 Homogeneous, Default b0=0)", get_activation("relu"), 0.0, 0.0, True),
        ("Proposed: Symm. sigma-AS-RectLU (k=0.8, b0=0, Parameter-Free)", PerSampleSigmaASRectLU(0.8, True), 0.0, 0.0, True),
    ]

    init_rows = []
    for g_name, act_mod, b_hid, b_out, dec_bias in init_groups:
        mccs, aucs, prcs, briers, eces = [], [], [], [], []
        for seed in (42, 43, 44):
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = Configurable2DCNN(act_mod, act_mod, hidden_bias_init=b_hid, output_bias_init=b_out).to(device)
            opt = make_optimizer(model, lr=1e-3, weight_decay=1e-4, decay_bias=dec_bias)
            crit = nn.BCEWithLogitsLoss()

            model.train()
            n_tr = len(X_tr_t)
            for _ in range(20):
                perm = torch.randperm(n_tr, device=device)
                for i in range(0, n_tr, 128):
                    idx = perm[i:i + 128]
                    opt.zero_grad(set_to_none=True)
                    logits, _ = model(X_tr_t[idx])
                    loss = crit(logits, y_tr_t[idx])
                    loss.backward()
                    opt.step()

            vp = eval_batched(model, X_va_t)
            tp = eval_batched(model, X_te_t)
            m = exact_calibrated_metrics(y_all[te_idx], tp, y_all[va_idx], vp)
            mccs.append(m["MCC_Cal"])
            aucs.append(m["AUROC"])
            prcs.append(m["AUPRC"])
            briers.append(m["Brier"])
            eces.append(m["ECE"])
            del model, opt
            if device == "cuda":
                torch.cuda.empty_cache()

        init_rows.append({
            "Group": g_name,
            "MCC_Cal_Mean": np.mean(mccs),
            "MCC_Cal_SD": np.std(mccs),
            "AUROC_Mean": np.mean(aucs),
            "AUPRC_Mean": np.mean(prcs),
            "Brier_Mean": np.mean(briers),
            "ECE_Mean": np.mean(eces),
        })
        print(f"{g_name:<58} | MCC(t*): {np.mean(mccs):.4f}±{np.std(mccs):.4f} | "
              f"AUROC: {np.mean(aucs):.4f} | AUPRC: {np.mean(prcs):.4f} | ECE: {np.mean(eces):.4f}")

    pd.DataFrame(init_rows).to_csv(out_dir / "rebuttal_table_init_groups_AF.csv", index=False)

    # -----------------------------------------------------------------
    # TEST 3: E8 Kappa-Sweep & Stop-Grad vs. Differentiable RMS Ablation
    # -----------------------------------------------------------------
    print("\n" + "=" * 105)
    print("TEST 3: E8 Mechanism Ablation — Kappa Sweep & Stop-Grad vs. Differentiable RMS")
    print("=" * 105)
    e8_configs = [
        ("Symm. sigma-AS-RectLU (k=0.4, stop_grad=True)", PerSampleSigmaASRectLU(0.4, True, detach_scale=True)),
        ("Symm. sigma-AS-RectLU (k=0.8, stop_grad=True)", PerSampleSigmaASRectLU(0.8, True, detach_scale=True)),
        ("Symm. sigma-AS-RectLU (k=0.8, stop_grad=False / Diff.)", PerSampleSigmaASRectLU(0.8, True, detach_scale=False)),
        ("Symm. sigma-AS-RectLU (k=1.2, stop_grad=True)", PerSampleSigmaASRectLU(1.2, True, detach_scale=True)),
        ("Symm. sigma-AS-RectLU (k=2.0, stop_grad=True)", PerSampleSigmaASRectLU(2.0, True, detach_scale=True)),
        ("Asymm. sigma-AS-RectLU (L=0, k=0.8, Adaptive Width Only)", PerSampleSigmaASRectLU(0.8, False, detach_scale=True)),
        ("LN (No-Affine) + SmeLU (Resets Scale to 1.0)", AffineFreeLNSmeLU(0.4)),
    ]
    e8_rows = []
    for name, act_mod in e8_configs:
        torch.manual_seed(42)
        model = Configurable2DCNN(act_mod, act_mod).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.BCEWithLogitsLoss()
        t0 = time.perf_counter()
        model.train()
        for _ in range(20):
            perm = torch.randperm(len(X_tr_t), device=device)
            for i in range(0, len(X_tr_t), 128):
                idx = perm[i:i + 128]
                opt.zero_grad(set_to_none=True)
                logits, _ = model(X_tr_t[idx])
                loss = crit(logits, y_tr_t[idx])
                loss.backward()
                opt.step()
        train_sec = time.perf_counter() - t0
        vp = eval_batched(model, X_va_t)
        tp = eval_batched(model, X_te_t)
        m = exact_calibrated_metrics(y_all[te_idx], tp, y_all[va_idx], vp)
        ci_mcc, ci_auc = block_bootstrap_ci(y_all[te_idx], tp, m["tau_star"])
        e8_rows.append({"Config": name, "Train_Sec": train_sec, **m, "MCC_CI_Low": ci_mcc[0], "MCC_CI_High": ci_mcc[1]})
        print(f"{name:<54} | MCC(t*): {m['MCC_Cal']:.4f} [{ci_mcc[0]:.3f}, {ci_mcc[1]:.3f}] | "
              f"AUROC: {m['AUROC']:.4f} | AUPRC: {m['AUPRC']:.4f} | Time: {train_sec:.2f}s")
        del model, opt
        if device == "cuda":
            torch.cuda.empty_cache()

    pd.DataFrame(e8_rows).to_csv(out_dir / "rebuttal_table_e8_ablation.csv", index=False)


if __name__ == "__main__":
    run_rebuttal_suite()