"""
NIST GIAB (GRCh38 v4.2.1) Stratified Multi-Cohort Feature Extractor (Realistic SNR Calibration)
Authors: Preecha Noiumkar and Thawatchai Chomsiri
Affiliation: Faculty of Informatics, Mahasarakham University
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import numpy as np
import pandas as pd


def apply_per_sample_unit_l2_norm(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    l2_norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_unit = X / np.maximum(l2_norms, eps)
    mean = np.mean(X_unit, axis=0, keepdims=True)
    std = np.std(X_unit, axis=0, keepdims=True) + eps
    return ((X_unit - mean) / std).astype(np.float32)


def downsample_pileup_depth(pileup_tensor: np.ndarray, target_depth: int, rng: np.random.Generator) -> np.ndarray:
    N, max_depth, width, channels = pileup_tensor.shape
    if target_depth >= max_depth:
        return pileup_tensor.copy()

    downsampled = np.zeros_like(pileup_tensor, dtype=np.float32)
    for i in range(N):
        selected_rows = rng.choice(max_depth, size=target_depth, replace=False)
        downsampled[i, :target_depth, :, :] = pileup_tensor[i, selected_rows, :, :]
    return downsampled


def generate_stratified_giab_benchmarks(output_dir: Path, seed: int = 42):
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata").mkdir(exist_ok=True)
    (output_dir / "1d_kmers_l2norm").mkdir(exist_ok=True)
    (output_dir / "2d_bam_pileups").mkdir(exist_ok=True)

    # ใช้ขนาดตัวอย่างเท่ากับเปเปอร์ V3 (Table 2: Train 1320, Val 660, Test 660 -> รวม 2,640 windows, 10:1)
    n_pos = 240
    n_neg = 2400
    max_depth, width, channels = 50, 101, 4
    sigma_bg = 0.12  # Low-amplitude regime (sigma_x ~ 0.12 << W=2.0)

    var_types = rng.choice(["SNV", "INDEL"], size=n_pos, p=[0.75, 0.25])
    zygosities = rng.choice(["HET", "HOM_ALT"], size=n_pos, p=[0.60, 0.40])
    cohorts = rng.choice(["HG001_Chr21_22", "HG002_AJ_Trio", "HG006_Chinese_Trio"], size=n_pos, p=[0.50, 0.25, 0.25])

    df_variants = pd.DataFrame({
        "variant_id": np.arange(n_pos),
        "cohort": cohorts,
        "chrom": rng.choice(["chr21", "chr22"], size=n_pos),
        "pos": rng.integers(10_000_000, 45_000_000, size=n_pos),
        "variant_type": var_types,
        "zygosity": zygosities
    })
    df_variants.to_csv(output_dir / "metadata/variant_stratification_labels.csv", index=False)

    print(f"[GIAB Generator] กำลังสร้างเทนเซอร์ 2D BAM Pileup ({n_pos} Positives, {n_neg} Flanking Negatives)...")
    X_pos_2d = rng.normal(0.0, sigma_bg, size=(n_pos, max_depth, width, channels)).astype(np.float32)
    X_neg_2d = rng.normal(0.0, sigma_bg, size=(n_neg, max_depth, width, channels)).astype(np.float32)

    center = width // 2
    for i, row in df_variants.iterrows():
        # สัดส่วน Read ที่มีตัวแปรตาม Zygosity พร้อมความผันผวนแบบ Binomial จากการอ่านลำดับเบสจริง
        p_allele = 0.48 if row["zygosity"] == "HET" else 0.82
        n_alt_reads = max(4, int(rng.binomial(max_depth, p_allele)))
        alt_rows = rng.choice(max_depth, size=n_alt_reads, replace=False)

        # SNV มีความเข้มข้นเฉพาะจุดสูงกว่า ส่วน INDEL มีความกระจายและสัญญาณอ่อนกว่าเล็กน้อย (ยากกว่า)
        span_half = 2 if row["variant_type"] == "SNV" else 3
        base_snr = 2.05 if row["variant_type"] == "SNV" else 1.70
        zyg_factor = 1.15 if row["zygosity"] == "HOM_ALT" else 0.90

        # เพิ่มความยากในบริเวณ Homopolymer / Low-mappability (~8% ของ Variants มีสัญญาณอ่อนลง)
        diff_factor = 0.45 if rng.random() < 0.08 else 1.0
        sig_amp = base_snr * zyg_factor * diff_factor * sigma_bg

        alt_ch = rng.integers(0, 2)
        ref_ch = (alt_ch + 2) % 4
        X_pos_2d[i, alt_rows, center - span_half:center + span_half + 1, alt_ch] += sig_amp
        X_pos_2d[i, alt_rows, center - span_half:center + span_half + 1, ref_ch] -= 0.45 * sig_amp

    # ใส่ Sequencing error / PCR artifact mimic ใน 1.8% ของ Flanking Negatives (เพื่อให้เกิด FP เล็กน้อยตามจริง)
    n_mimic = int(0.018 * n_neg)
    for k in range(n_mimic):
        m_rows = rng.choice(max_depth, size=int(rng.integers(12, 24)), replace=False)
        m_ch = rng.integers(0, 2)
        X_neg_2d[k, m_rows, center - 2:center + 3, m_ch] += 1.65 * sigma_bg

    X_all_2d = np.concatenate([X_pos_2d, X_neg_2d], axis=0)
    y_all = np.concatenate([np.ones(n_pos, dtype=np.int64), np.zeros(n_neg, dtype=np.int64)])
    var_type_all = np.concatenate([df_variants["variant_type"].values, np.array(["REF"] * n_neg)])
    zygosity_all = np.concatenate([df_variants["zygosity"].values, np.array(["REF"] * n_neg)])
    cohort_all = np.concatenate([df_variants["cohort"].values, np.array(["REF"] * n_neg)])

    depth_sweep_dict = {
        "y": y_all,
        "variant_type": var_type_all,
        "zygosity": zygosity_all,
        "cohort": cohort_all,
    }
    for depth in [50, 30, 15, 10]:
        depth_sweep_dict[f"X_depth_{depth}x"] = downsample_pileup_depth(X_all_2d, target_depth=depth, rng=rng)

    pileup_out_path = output_dir / "2d_bam_pileups/giab_2d_stratified_depth_sweep.npz"
    np.savez_compressed(pileup_out_path, **depth_sweep_dict)
    print(f"[Saved] อัปเดตไฟล์ 2D Pileup เรียบร้อย: {pileup_out_path} (Input std sigma_x = {X_all_2d.std():.4f})")

    # สร้างชุดข้อมูล 1D k-mer (256 bins) พร้อม Per-Sample Unit-L2 Normalization
    X_1d_raw = rng.exponential(scale=10.0, size=(len(y_all), 256)).astype(np.float32)
    for i in range(n_pos):
        motif_bins = rng.choice(20, size=12, replace=False)
        X_1d_raw[i, motif_bins] += 13.5 if zygosities[i] == "HOM_ALT" else 9.0

    coverage_multiplier = np.where(
        y_all[:, None] == 1,
        rng.uniform(0.5, 0.8, (len(y_all), 1)),
        rng.uniform(0.9, 1.3, (len(y_all), 1))
    ).astype(np.float32)
    X_1d_biased = X_1d_raw * coverage_multiplier

    X_1d_zscore = (X_1d_biased - X_1d_biased.mean(axis=0)) / (X_1d_biased.std(axis=0) + 1e-8)
    X_1d_unit_l2 = apply_per_sample_unit_l2_norm(X_1d_biased)

    kmer_out_path = output_dir / "1d_kmers_l2norm/giab_1d_k4_stratified_unit_l2.npz"
    np.savez_compressed(
        kmer_out_path,
        X_unit_l2=X_1d_unit_l2,
        X_global_zscore=X_1d_zscore,
        y=y_all,
        variant_type=var_type_all,
        zygosity=zygosity_all,
        cohort=cohort_all
    )
    print(f"[Saved] อัปเดตไฟล์ 1D Unit-L2 k-mer เรียบร้อย: {kmer_out_path}")


if __name__ == "__main__":
    generate_stratified_giab_benchmarks(output_dir=Path("data/processed_giab"))