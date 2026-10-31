"""
Step 07: Fast & Memory-Safe Real NIST GIAB Evaluation (4GB VRAM Optimized)
Authors: Preecha Noiumkar and Thawatchai Chomsiri
Affiliation: Faculty of Informatics, Mahasarakham University
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.activations import (
    get_activation,
    PerSampleSigmaASRectLU,
    AffineFreeLNSmeLU,
    RectLUContinuous,
    SmoothRectLU,
    SmeLU,
)
from src.giab_stratified_extractor import generate_stratified_giab_benchmarks


class PathwayA1DCNN(nn.Module):
    def __init__(self, conv_act: nn.Module, dense_act: nn.Module, input_len: int = 256):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.conv_act = conv_act

        flat_dim = (input_len // 8) * 128
        self.fc1 = nn.Linear(flat_dim, 256)
        self.dense_act = dense_act
        self.drop = nn.Dropout(0.5)
        self.out = nn.Linear(256, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = self.pool(self.conv_act(self.conv1(x)))
        x = self.pool(self.conv_act(self.conv2(x)))
        x = self.pool(self.conv_act(self.conv3(x)))
        x = x.flatten(1)
        h_pre = self.fc1(x)
        h_act = self.dense_act(h_pre)
        logits = self.out(self.drop(h_act)).squeeze(-1)
        return logits, h_act


class PathwayB2DCNN(nn.Module):
    def __init__(self, conv_act: nn.Module, dense_act: nn.Module, in_channels: int = 4, H: int = 50, W: int = 101):
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
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        x = self.pool(self.conv_act(self.conv1(x)))
        x = self.pool(self.conv_act(self.conv2(x)))
        x = self.pool(self.conv_act(self.conv3(x)))
        x = x.flatten(1)
        h_pre = self.fc1(x)
        h_act = self.dense_act(h_pre)
        logits = self.out(self.drop(h_act)).squeeze(-1)
        return logits, h_act


def compute_comprehensive_metrics(y_true: np.ndarray, y_prob: np.ndarray, val_true: np.ndarray = None, val_prob: np.ndarray = None) -> dict:
    y_pred_05 = (y_prob >= 0.50).astype(int)
    tp = int(np.sum((y_pred_05 == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred_05 == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred_05 == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred_05 == 0) & (y_true == 1)))

    mcc_05 = matthews_corrcoef(y_true, y_pred_05) if len(np.unique(y_pred_05)) > 1 else 0.0
    auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    auprc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0

    best_tau = 0.50
    if val_true is not None and val_prob is not None:
        best_val_mcc = -1.0
        for tau in np.linspace(0.02, 0.95, 50):
            vp = (val_prob >= tau).astype(int)
            vmcc = matthews_corrcoef(val_true, vp) if len(np.unique(vp)) > 1 else 0.0
            if vmcc > best_val_mcc:
                best_val_mcc = vmcc
                best_tau = tau

    y_pred_cal = (y_prob >= best_tau).astype(int)
    mcc_cal = matthews_corrcoef(y_true, y_pred_cal) if len(np.unique(y_pred_cal)) > 1 else 0.0

    return {
        "MCC_050": float(mcc_05),
        "MCC_Cal": float(mcc_cal),
        "AUROC": float(auroc),
        "AUPRC": float(auprc),
        "Pred_Pos": tp + fp,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
    }


def evaluate_stratified_subsets(y_true: np.ndarray, y_prob: np.ndarray, var_type: np.ndarray, zygosity: np.ndarray, cohort: np.ndarray, val_true: np.ndarray, val_prob: np.ndarray) -> dict:
    neg_mask = (y_true == 0)
    results = {}

    for vt in ["SNV", "INDEL"]:
        mask = neg_mask | ((y_true == 1) & (var_type == vt))
        if np.sum((y_true == 1) & (var_type == vt)) > 2:
            m = compute_comprehensive_metrics(y_true[mask], y_prob[mask], val_true, val_prob)
            results[f"{vt}_MCC"] = m["MCC_Cal"]
            results[f"{vt}_AUROC"] = m["AUROC"]
        else:
            results[f"{vt}_MCC"], results[f"{vt}_AUROC"] = np.nan, np.nan

    for zyg in ["HET", "HOM_ALT"]:
        mask = neg_mask | ((y_true == 1) & (zygosity == zyg))
        if np.sum((y_true == 1) & (zygosity == zyg)) > 2:
            m = compute_comprehensive_metrics(y_true[mask], y_prob[mask], val_true, val_prob)
            results[f"{zyg}_MCC"] = m["MCC_Cal"]
            results[f"{zyg}_AUROC"] = m["AUROC"]
        else:
            results[f"{zyg}_MCC"], results[f"{zyg}_AUROC"] = np.nan, np.nan

    for c_name in np.unique(cohort[y_true == 1]):
        mask = neg_mask | ((y_true == 1) & (cohort == c_name))
        if np.sum((y_true == 1) & (cohort == c_name)) > 2:
            m = compute_comprehensive_metrics(y_true[mask], y_prob[mask], val_true, val_prob)
            results[f"{c_name}_MCC"] = m["MCC_Cal"]
            results[f"{c_name}_AUROC"] = m["AUROC"]

    return results


@torch.no_grad()
def batched_eval_gpu(model: nn.Module, X_t: torch.Tensor, batch_size: int = 128):
    """รัน Inference ทีละ Batch ย่อยบน GPU เพื่อไม่ให้เกิน 4GB VRAM"""
    model.eval()
    n = X_t.shape[0]
    probs = []
    dead_weighted_sum = 0.0

    for i in range(0, n, batch_size):
        xb = X_t[i:i + batch_size]
        logits, h_act = model(xb)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        dead_weighted_sum += float(torch.mean((h_act.abs() < 1e-7).float()).item()) * len(xb)

    return np.concatenate(probs, axis=0), (dead_weighted_sum / n) * 100.0


def train_and_eval_gpu_resident(
    model_builder,
    X_train_t: torch.Tensor, y_train_t: torch.Tensor,
    X_val_t: torch.Tensor,
    X_test_t: torch.Tensor,
    seed: int,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "cuda"
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if "cuda" in device:
        torch.cuda.manual_seed_all(seed)

    model = model_builder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    n_train = X_train_t.shape[0]
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n_train, device=device)
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb = X_train_t[idx]
            yb = y_train_t[idx]
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    val_prob, _ = batched_eval_gpu(model, X_val_t, batch_size=batch_size)
    test_prob, dead_unit_pct = batched_eval_gpu(model, X_test_t, batch_size=batch_size)

    # เคลียร์โมเดลและแคชออกจาก VRAM ทันทีหลังจบแต่ละ Seed
    del model, optimizer, criterion
    if "cuda" in device:
        torch.cuda.empty_cache()

    return val_prob, test_prob, dead_unit_pct


def run_step07_pipeline(seeds=(42, 43, 44, 45, 46), epochs: int = 20):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print(f"[Step 07] เริ่มการประเมินผล NIST GIAB (4GB VRAM Safe Mode) บน: {device.upper()} ({gpu_name})")

    data_dir = ROOT_DIR / "data/processed_giab"
    results_dir = ROOT_DIR / "results/logs_per_seed"
    results_dir.mkdir(parents=True, exist_ok=True)

    generate_stratified_giab_benchmarks(output_dir=data_dir)

    pileup_npz = data_dir / "2d_bam_pileups/giab_2d_stratified_depth_sweep.npz"
    kmer_npz = data_dir / "1d_kmers_l2norm/giab_1d_k4_stratified_unit_l2.npz"

    d2 = np.load(pileup_npz, allow_pickle=True)
    y_all = d2["y"]
    var_type_all = d2["variant_type"]
    zygosity_all = d2["zygosity"]
    cohort_all = d2["cohort"]

    pos_idx = np.where(y_all == 1)[0]
    neg_idx = np.where(y_all == 0)[0]

    def split_blocks(idx_arr):
        n = len(idx_arr)
        return idx_arr[:int(0.50 * n)], idx_arr[int(0.50 * n):int(0.75 * n)], idx_arr[int(0.75 * n):]

    p_tr, p_va, p_te = split_blocks(pos_idx)
    n_tr, n_va, n_te = split_blocks(neg_idx)

    tr_idx = np.concatenate([p_tr, n_tr])
    va_idx = np.concatenate([p_va, n_va])
    te_idx = np.concatenate([p_te, n_te])

    configs_2d = [
        ("ReLU (C0)", lambda: PathwayB2DCNN(get_activation("relu"), get_activation("relu"))),
        ("Symm. SmeLU (L=-1.0, W=2.0)", lambda: PathwayB2DCNN(SmeLU(beta=1.0), SmeLU(beta=1.0))),
        ("Smooth RectLU (L=0.0, W=2.0)", lambda: PathwayB2DCNN(SmoothRectLU(0.0, 2.0), SmoothRectLU(0.0, 2.0))),
        ("Cont. RectLU (L=0.0, W=2.0)", lambda: PathwayB2DCNN(RectLUContinuous(0.0, 2.0), RectLUContinuous(0.0, 2.0))),
        ("Cont. RectLU (L=-1.0, W=2.0)", lambda: PathwayB2DCNN(RectLUContinuous(-1.0, 2.0), RectLUContinuous(-1.0, 2.0))),
        ("Cont. RectLU Wide (L=0.0, W=8.5)", lambda: PathwayB2DCNN(RectLUContinuous(0.0, 8.5), RectLUContinuous(0.0, 8.5))),
        ("Cont. RectLU Wide (L=-1.5, W=8.5)", lambda: PathwayB2DCNN(RectLUContinuous(-1.5, 8.5), RectLUContinuous(-1.5, 8.5))),
        ("Hybrid (Smooth-Conv + ReLU-Dense)", lambda: PathwayB2DCNN(SmoothRectLU(0.0, 2.0), get_activation("relu"))),
        ("LN (No-Affine) + SmeLU", lambda: PathwayB2DCNN(AffineFreeLNSmeLU(0.4), AffineFreeLNSmeLU(0.4))),
        ("Symm. sigma-AS-RectLU (k=0.8)", lambda: PathwayB2DCNN(PerSampleSigmaASRectLU(0.8, True), PerSampleSigmaASRectLU(0.8, True))),
    ]

    X_50x = np.transpose(d2["X_depth_50x"], (0, 3, 1, 2)).astype(np.float32)
    X_tr_t = torch.tensor(X_50x[tr_idx], device=device)
    y_tr_t = torch.tensor(y_all[tr_idx], dtype=torch.float32, device=device)
    X_va_t = torch.tensor(X_50x[va_idx], device=device)
    X_te_t = torch.tensor(X_50x[te_idx], device=device)

    stratified_rows = []

    print("\n" + "=" * 115)
    print("STAGE 1: 2D Real-BAM Imbalanced (10:1) & Stratified Variant/Zygosity Evaluation (N=5 Seeds, 20 Epochs)")
    print("=" * 115)

    for idx_c, (cfg_name, builder) in enumerate(configs_2d, 1):
        t_start = time.perf_counter()
        print(f"[{idx_c:02d}/{len(configs_2d):02d}] {cfg_name:<33} ", end="", flush=True)
        for seed in seeds:
            val_prob, test_prob, dead_pct = train_and_eval_gpu_resident(
                builder, X_tr_t, y_tr_t, X_va_t, X_te_t,
                seed=seed, epochs=epochs, device=device
            )
            print(".", end="", flush=True)
            base_m = compute_comprehensive_metrics(y_all[te_idx], test_prob, y_all[va_idx], val_prob)
            strat_m = evaluate_stratified_subsets(
                y_all[te_idx], test_prob,
                var_type_all[te_idx], zygosity_all[te_idx], cohort_all[te_idx],
                y_all[va_idx], val_prob
            )
            stratified_rows.append({"Configuration": cfg_name, "Seed": seed, "Dead_Pct": dead_pct, **base_m, **strat_m})

        elapsed = time.perf_counter() - t_start
        df_cfg = pd.DataFrame([r for r in stratified_rows if r["Configuration"] == cfg_name])
        print(f" ({elapsed:4.1f}s) | MCC(0.5): {df_cfg['MCC_050'].mean():.4f} | MCC(t*): {df_cfg['MCC_Cal'].mean():.4f} | "
              f"AUC: {df_cfg['AUROC'].mean():.4f} | SNV: {df_cfg['SNV_MCC'].mean():.4f} | INDEL: {df_cfg['INDEL_MCC'].mean():.4f} | "
              f"HET: {df_cfg['HET_MCC'].mean():.4f} | HOM: {df_cfg['HOM_ALT_MCC'].mean():.4f}")

    out_2d_csv = results_dir / "v5_real_giab2d_imbalanced_multiseed.csv"
    pd.DataFrame(stratified_rows).to_csv(out_2d_csv, index=False)
    print(f"[Saved] บันทึกผลลัพธ์ 2D Stratified ลงใน: {out_2d_csv}")

    # คืนพื้นที่ VRAM ของ X_50x ก่อนเริ่ม Stage 2
    del X_tr_t, X_va_t, X_te_t
    if "cuda" in device:
        torch.cuda.empty_cache()
    gc.collect()

    # -----------------------------------------------------------------
    # Part 2: Read-Depth Downsampling Sweep (50x, 30x, 15x, 10x)
    # -----------------------------------------------------------------
    print("\n" + "=" * 115)
    print("STAGE 2: Biological Scale-Width Mismatch via Read-Depth Downsampling (50x -> 30x -> 15x -> 10x)")
    print("=" * 115)
    depth_configs = [
        ("ReLU (C0)", configs_2d[0][1]),
        ("Smooth RectLU (L=0.0, W=2.0)", configs_2d[2][1]),
        ("Hybrid (Smooth-Conv + ReLU-Dense)", configs_2d[7][1]),
        ("Symm. sigma-AS-RectLU (k=0.8)", configs_2d[9][1]),
    ]
    depth_rows = []
    for depth in [50, 30, 15, 10]:
        t_d = time.perf_counter()
        X_d = np.transpose(d2[f"X_depth_{depth}x"], (0, 3, 1, 2)).astype(np.float32)
        Xd_tr = torch.tensor(X_d[tr_idx], device=device)
        Xd_va = torch.tensor(X_d[va_idx], device=device)
        Xd_te = torch.tensor(X_d[te_idx], device=device)

        for cfg_name, builder in depth_configs:
            for seed in seeds[:3]:
                val_p, test_p, dead_pct = train_and_eval_gpu_resident(
                    builder, Xd_tr, y_tr_t, Xd_va, Xd_te,
                    seed=seed, epochs=epochs, device=device
                )
                m = compute_comprehensive_metrics(y_all[te_idx], test_p, y_all[va_idx], val_p)
                depth_rows.append({"Depth": f"{depth}x", "Configuration": cfg_name, "Seed": seed, "Dead_Pct": dead_pct, **m})

        del Xd_tr, Xd_va, Xd_te
        if "cuda" in device:
            torch.cuda.empty_cache()

        df_d = pd.DataFrame([r for r in depth_rows if r["Depth"] == f"{depth}x"])
        summary_str = " | ".join(
            f"{c.split('(')[0].strip()}: MCC={df_d[df_d['Configuration']==c]['MCC_Cal'].mean():.4f} (AUC={df_d[df_d['Configuration']==c]['AUROC'].mean():.4f})"
            for c, _ in depth_configs
        )
        print(f"Depth {depth:>2}x ({time.perf_counter()-t_d:4.1f}s) -> {summary_str}")

    pd.DataFrame(depth_rows).to_csv(results_dir / "v6_giab2d_depth_sweep_logs.csv", index=False)

    # -----------------------------------------------------------------
    # Part 3: 1D k-mer Unit-L2 Normalized vs. Global Z-Score Control
    # -----------------------------------------------------------------
    print("\n" + "=" * 115)
    print("STAGE 3: 1D k-mer Coverage-Shortcut Dissection (Per-Sample Unit-L2 Norm vs. Heuristics)")
    print("=" * 115)
    d1 = np.load(kmer_npz, allow_pickle=True)
    X_1d_l2 = d1["X_unit_l2"]
    y_1d = d1["y"]

    X1_tr = torch.tensor(X_1d_l2[tr_idx], device=device)
    y1_tr = torch.tensor(y_1d[tr_idx], dtype=torch.float32, device=device)
    X1_va = torch.tensor(X_1d_l2[va_idx], device=device)
    X1_te = torch.tensor(X_1d_l2[te_idx], device=device)

    rows_1d = []
    max_bin_score = np.max(X_1d_l2[te_idx], axis=1)
    max_bin_auroc = roc_auc_score(y_1d[te_idx], max_bin_score)
    print(f"Max-Bin Heuristic Control (Unit-L2 Norm)  | Test AUROC: {max_bin_auroc:.4f}")

    configs_1d = [
        ("ReLU (C0)", lambda: PathwayA1DCNN(get_activation("relu"), get_activation("relu"))),
        ("Smooth RectLU (L=0.0, W=2.0)", lambda: PathwayA1DCNN(SmoothRectLU(0.0, 2.0), SmoothRectLU(0.0, 2.0))),
        ("Symm. sigma-AS-RectLU (k=0.8)", lambda: PathwayA1DCNN(PerSampleSigmaASRectLU(0.8, True), PerSampleSigmaASRectLU(0.8, True))),
    ]
    for cfg_name, builder in configs_1d:
        t_1d = time.perf_counter()
        for seed in seeds:
            val_p, test_p, dead_pct = train_and_eval_gpu_resident(
                builder, X1_tr, y1_tr, X1_va, X1_te,
                seed=seed, epochs=epochs, device=device
            )
            m = compute_comprehensive_metrics(y_1d[te_idx], test_p, y_1d[va_idx], val_p)
            rows_1d.append({"Configuration": cfg_name, "Seed": seed, "Dead_Pct": dead_pct, **m})

        df_c1 = pd.DataFrame([r for r in rows_1d if r["Configuration"] == cfg_name])
        print(f"{cfg_name:<35} ({time.perf_counter()-t_1d:4.1f}s) | Test AUROC: {df_c1['AUROC'].mean():.4f}±{df_c1['AUROC'].std():.4f} | "
              f"Calibrated MCC(tau*): {df_c1['MCC_Cal'].mean():.4f}")

    out_1d_csv = results_dir / "v5_real_giab1d_l2norm_multiseed.csv"
    pd.DataFrame(rows_1d).to_csv(out_1d_csv, index=False)
    print(f"[Saved] บันทึกผลลัพธ์ 1D Unit-L2 k-mer ลงใน: {out_1d_csv}")


if __name__ == "__main__":
    run_step07_pipeline()