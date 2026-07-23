#!/usr/bin/env python
"""Evaluate a fine-tuned AlphaGenome model.

Supports three opt-in features (--metrics, --regions, --ism) and an optional
native-head comparison layer (--native-biosample) that enriches all outputs.

*MEMORY & SPEED OPTIMIZED VERSION* 
Computes test metrics batch-by-batch purely on the GPU to prevent Out Of 
Memory (OOM) errors and bypass the single-thread Python CPU bottleneck.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.extensions.finetuning.checkpointing import (
    is_delta_checkpoint,
    load_delta_checkpoint,
    load_finetuned_model as _load_finetuned_model,
)
from alphagenome_pytorch.extensions.finetuning.datasets import GenomicDataset
from alphagenome_pytorch.extensions.finetuning.heads import create_finetuning_head
from alphagenome_pytorch.extensions.finetuning.training import collate_genomic
from alphagenome_pytorch.extensions.finetuning.transfer import load_trunk
from alphagenome_pytorch.losses import multinomial_loss
from alphagenome_pytorch.named_outputs import TrackMetadataCatalog
from alphagenome_pytorch.metrics import profile_pearson_r

log = logging.getLogger(__name__)

NUM_SEGMENTS = 8


# =============================================================================
# Model loading
# =============================================================================

def load_finetuned_model(
    checkpoint_path: str,
    pretrained_weights: str,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    log.info("Loading checkpoint: %s", Path(checkpoint_path).name)
    model, meta = _load_finetuned_model(
        checkpoint_path=checkpoint_path,
        pretrained_weights=pretrained_weights,
        device=device,
        merge=True,
    )
    for p in model.parameters():
        p.requires_grad = False
    return model, meta


def load_native_model(
    pretrained_weights: str,
    native_biosample: str | None,
    native_track_index: int | None,
    modality: str,
    device: torch.device,
) -> tuple[nn.Module, int, str]:
    log.info("Loading native model for comparison...")
    model = AlphaGenome.from_pretrained(pretrained_weights, device=device)
    model.eval()
    model = torch.compile(model)
    for p in model.parameters():
        p.requires_grad = False

    catalog = TrackMetadataCatalog.load_builtin("human")
    model.set_track_metadata_catalog(catalog)

    if native_track_index is not None:
        tracks = catalog.get_tracks(modality, organism=0)
        track = tracks[native_track_index]
        display_name = track.get("biosample_name") or track.track_name
        return model, native_track_index, display_name

    tracks = catalog.get_tracks(modality, organism=0)
    query = native_biosample.lower()
    matches = [
        t for t in tracks
        if not t.is_padding and query in (t.get("biosample_name") or "").lower()
    ]

    track = matches[0]
    display_name = track.get("biosample_name") or track.track_name
    return model, track.track_index, display_name


# =============================================================================
# GPU-Accelerated Metric Helpers (Blazing Fast)
# =============================================================================

def jsd_per_region(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Calculates JSD instantly using PyTorch GPU tensors."""
    p = targets / (targets.sum(dim=1, keepdim=True) + eps)
    q = preds / (preds.sum(dim=1, keepdim=True) + eps)
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * torch.log((p + eps) / (m + eps)), dim=1)
    kl_qm = torch.sum(q * torch.log((q + eps) / (m + eps)), dim=1)
    return 0.5 * (kl_pm + kl_qm)

def _init_metrics(resolutions: tuple[int, ...]) -> dict:
    return {r: {
        'profile_rs': [], 'pred_counts': [], 'target_counts': [],
        'jsd_vals': [], 'mse_sum': 0.0, 'mse_count': 0, # these are bin/bp specific mse
        'region_mse_sum': 0.0, 'region_rmsle_sum': 0.0, 'region_x_tracks': 0, # these are region-specific mse metrics (sum of all bins/bps in the region)
        'global_p_sample': [], 'global_t_sample': [], 'n_regions': 0
    } for r in resolutions}

def _update_metrics_batch(m: dict, p_batch: torch.Tensor, t_batch: torch.Tensor):
    """Fast, memory-safe batch metric updater running entirely on the GPU."""
    b_size = p_batch.shape[0]
    m['n_regions'] += b_size
    
    # # 1. Profile Pearson R (Flatten to Seq * Tracks per region)
    # p_flat = p_batch.reshape(b_size, -1)
    # t_flat = t_batch.reshape(b_size, -1)
    
    # # Vectorized std calculations on GPU
    # std_p = p_flat.std(dim=1)
    # std_t = t_flat.std(dim=1)
    # valid_mask = (std_p > 1e-10) & (std_t > 1e-10)
    
    # # Vectorized Pearson Math on GPU
    # p_mean = p_flat.mean(dim=1, keepdim=True)
    # t_mean = t_flat.mean(dim=1, keepdim=True)
    
    # p_norm = p_flat - p_mean
    # t_norm = t_flat - t_mean
    
    # num = torch.sum(p_norm * t_norm, dim=1)
    # den = torch.sqrt(torch.sum(p_norm ** 2, dim=1) * torch.sum(t_norm ** 2, dim=1))
    
    # # Safely apply correlation only to valid regions
    # r_vals = torch.zeros(b_size, dtype=torch.float32, device=p_batch.device)
    # r_vals[valid_mask] = num[valid_mask] / den[valid_mask]
    
    # # Move the 4 summary numbers to CPU *after* doing the math
    # m['profile_rs'].extend(r_vals.cpu().tolist())

    batch_profile_r = profile_pearson_r(p_batch, t_batch)  # Shape: (batch_size, tracks)[cite: 2]
    m['profile_rs'].extend(batch_profile_r.cpu().flatten().tolist())

    # 2. Total Count Accumulation
    m['pred_counts'].extend(p_batch.sum(dim=1).flatten().cpu().tolist())
    m['target_counts'].extend(t_batch.sum(dim=1).flatten().cpu().tolist())
    
    # 3. JSD Mean
    jsd_batch = jsd_per_region(p_batch, t_batch)
    m['jsd_vals'].extend(jsd_batch.mean(dim=1).cpu().tolist())

    # 4. MSE Tracker
    m['mse_sum'] += torch.sum((p_batch - t_batch) ** 2).item()
    m['mse_count'] += p_batch.numel()

    pred_region_sums = p_batch.sum(dim=1)  # Shape: (Batch, Tracks)
    target_region_sums = t_batch.sum(dim=1)

    # Count RMSLE (log1p to handle zeros safely)
    region_mse_batch = torch.sum((pred_region_sums - target_region_sums) ** 2).item()
    m['region_mse_sum'] += region_mse_batch
    log_pred_counts = torch.log1p(pred_region_sums)
    log_true_counts = torch.log1p(target_region_sums)
    count_rmsle_batch = torch.sum((log_pred_counts - log_true_counts) ** 2).item()
    m['region_rmsle_sum'] += count_rmsle_batch
    
    m['region_x_tracks'] += pred_region_sums.numel() # n_regions x tracks
    
    # 5. Stratified Subsampling 
    # grabs a stratified sample of up to 1,000 random base-pair (or bin) predictions 
    # from every single region. If evaluating 2,000 regions, this creates a pool of ~2,000,000 points.
    # Batch 0 Processing:
    # i=0 (Reg 0): Adds 1,000 points. (List length is now 1,000)
    # i=1 (Reg 1): Adds 1,000 points. (List length is now 2,000)
    # i=2 (Reg 2): Adds 1,000 points. (List length is now 3,000)
    # i=3 (Reg 3): Adds 1,000 points. (List length is now 4,000)
    # Batch 1 Processing:
    # i=0 (Reg 4): Adds 1,000 points. (List length is now 5,000)
    # i=1 (Reg 5): Adds 1,000 points. (List length is now 6,000)

    p_flat = p_batch.reshape(b_size, -1)
    t_flat = t_batch.reshape(b_size, -1)

    n_samples_per_region = min(p_flat.shape[1], 1000)
    idx = torch.randint(0, p_flat.shape[1], (n_samples_per_region,), device=p_batch.device)
    # each batch is a region => draw 1000 points from each region
    for i in range(b_size):
        m['global_p_sample'].extend(p_flat[i, idx].cpu().tolist()) # pull out the ith region in the current batch of 4 regions, extract 1000 from that region
        m['global_t_sample'].extend(t_flat[i, idx].cpu().tolist())

def _finalize_metrics(m: dict) -> dict:
    if m['n_regions'] == 0:
        return {}
    p_counts = np.array(m['pred_counts'])
    t_counts = np.array(m['target_counts'])
    
    count_r = float(stats.pearsonr(p_counts, t_counts)[0]) if (np.std(p_counts) > 1e-10 and np.std(t_counts) > 1e-10) else 0.0
    spearman_global = float(stats.spearmanr(m['global_p_sample'], m['global_t_sample'])[0])
    mse = float(m['mse_sum'] / max(1, m['mse_count']))

    region_mse = float(m['region_mse_sum'] / max(1, m['count_n']))
    region_rmsle = float(np.sqrt(m['region_rmsle_sum'] / max(1, m['region_x_tracks'])))
    
    return {
        "profile_pearson_r_all": np.array(m['profile_rs']),
        "profile_pearson_r_mean": float(np.mean(m['profile_rs'])),
        "profile_pearson_r_median": float(np.median(m['profile_rs'])),
        "count_pearson_r": count_r,
        "jsd_all": np.array(m['jsd_vals']),
        "jsd_mean": float(np.mean(m['jsd_vals'])),
        "jsd_median": float(np.median(m['jsd_vals'])),
        "mse": mse,
        "count_mse": region_mse,            # region-level mse, matches validation
        "count_rmsle": region_rmsle,        # region-level RMSLE, matches validation
        "spearman_global": spearman_global,
        "n_regions": m['n_regions'],
        "_scatter_p": np.array(m['global_p_sample']),
        "_scatter_t": np.array(m['global_t_sample']),
        "_pred_counts": p_counts,
        "_target_counts": t_counts,
    }

# =============================================================================
# Inference (Memory Optimized Streaming)
# =============================================================================

@torch.no_grad()
def evaluate_test_split(
    model: nn.Module, modality: str, loader: DataLoader,
    device: torch.device, resolutions: tuple[int, ...], positional_weight: float = 5.0,
) -> tuple[dict[int, dict], float]:
    """Evaluates and computes metrics batch-by-batch on the GPU."""
    model.eval()
    model = torch.compile(model)
    head = model.heads[modality]

    total_loss = 0.0
    n_batches = 0
    metrics = _init_metrics(resolutions)

    for sequences, targets_dict in tqdm(loader, desc="Evaluating (finetuned)"):
        sequences = sequences.to(device)
        organism_idx = torch.zeros(sequences.shape[0], dtype=torch.long, device=device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs = model(sequences, organism_idx, return_embeddings=True, resolutions=resolutions, channels_last=False)
            embeddings_dict = {res: outputs[f"embeddings_{res}bp"] for res in resolutions if f"embeddings_{res}bp" in outputs}
            scaled_preds = head(embeddings_dict, organism_idx, return_scaled=True, channels_last=True)
            exp_preds = head(embeddings_dict, organism_idx, return_scaled=False, channels_last=True)

        loss = torch.tensor(0.0, device=device)
        for res in resolutions:
            if res not in scaled_preds or res not in targets_dict:
                continue
            pred = scaled_preds[res]
            targets = targets_dict[res].to(device)
            targets_scaled = head.scale(targets, organism_idx, resolution=res)
            mask = torch.ones(pred.shape[0], 1, pred.shape[-1], dtype=torch.bool, device=device)
            
            seq_len = pred.shape[-2]
            mn_res = max(1, seq_len // NUM_SEGMENTS)
            while mn_res > 1 and seq_len % mn_res != 0:
                mn_res -= 1
            ld = multinomial_loss(
                y_pred=pred, y_true=targets_scaled, mask=mask,
                multinomial_resolution=mn_res, positional_weight=positional_weight,
            )
            loss = loss + ld["loss"]

            if res in exp_preds:
                # Retain the massive arrays on the GPU!
                p_batch = exp_preds[res].float()
                t_batch = targets.float()
                _update_metrics_batch(metrics[res], p_batch, t_batch)
                print(f"DEBUG: p_batch shape = {p_batch.shape}")
        total_loss += loss.item()
        n_batches += 1

    final_metrics = {r: _finalize_metrics(m) for r, m in metrics.items() if m['n_regions'] > 0}
    return final_metrics, total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate_native_test_split(
    model: nn.Module, modality: str, track_index: int, loader: DataLoader,
    device: torch.device, resolutions: tuple[int, ...],
) -> dict[int, dict]:
    """Evaluates native head batch-by-batch on the GPU."""
    model.eval()
    metrics = _init_metrics(resolutions)

    for sequences, targets_dict in tqdm(loader, desc="Evaluating (native)"):
        sequences = sequences.to(device)
        organism_idx = torch.zeros(sequences.shape[0], dtype=torch.long, device=device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs = model(sequences, organism_idx)

        if modality not in outputs:
            continue
        head_outputs = outputs[modality]
        for res in resolutions:
            if res not in head_outputs or res not in targets_dict:
                continue
            pred_track = head_outputs[res][:, :, track_index : track_index + 1]
            p_batch = pred_track.float()
            t_batch = targets_dict[res].to(device).float()
            _update_metrics_batch(metrics[res], p_batch, t_batch)

    return {r: _finalize_metrics(m) for r, m in metrics.items() if m['n_regions'] > 0}


# =============================================================================
# Small-Scale Extraction (Strictly for Region Plotting)
# =============================================================================

@torch.no_grad()
def predict_regions_split(
    model: nn.Module, modality: str, loader: DataLoader, device: torch.device, resolutions: tuple[int, ...]
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Keeps the original array-hoarding logic STRICTLY for the tiny --regions dataset."""
    model.eval()
    model = torch.compile(model)
    head = model.heads[modality]
    
    preds_by_res = {r: [] for r in resolutions}
    targets_by_res = {r: [] for r in resolutions}

    for sequences, targets_dict in tqdm(loader, desc="Extracting Region Preds"):
        sequences = sequences.to(device)
        organism_idx = torch.zeros(sequences.shape[0], dtype=torch.long, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs = model(sequences, organism_idx, return_embeddings=True, resolutions=resolutions, channels_last=False)
            embeddings_dict = {res: outputs[f"embeddings_{res}bp"] for res in resolutions if f"embeddings_{res}bp" in outputs}
            exp_preds = head(embeddings_dict, organism_idx, return_scaled=False)

        for res in resolutions:
            if res in exp_preds:
                preds_by_res[res].append(exp_preds[res].float().cpu().numpy())
            if res in targets_dict:
                targets_by_res[res].append(targets_dict[res].numpy())

    return (
        {r: np.concatenate(v, axis=0) for r, v in preds_by_res.items() if v},
        {r: np.concatenate(v, axis=0) for r, v in targets_by_res.items() if v}
    )

@torch.no_grad()
def predict_native_regions_split(
    model: nn.Module, modality: str, track_index: int, loader: DataLoader, device: torch.device, resolutions: tuple[int, ...]
) -> dict[int, np.ndarray]:
    model.eval()
    preds_by_res = {r: [] for r in resolutions}
    for sequences, _ in tqdm(loader, desc="Extracting Native Region Preds"):
        sequences = sequences.to(device)
        organism_idx = torch.zeros(sequences.shape[0], dtype=torch.long, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs = model(sequences, organism_idx)
        if modality in outputs:
            for res in resolutions:
                if res in outputs[modality]:
                    pred_track = outputs[modality][res][:, :, track_index : track_index + 1]
                    preds_by_res[res].append(pred_track.float().cpu().numpy())
    return {r: np.concatenate(v, axis=0) for r, v in preds_by_res.items() if v}

# =============================================================================
# Plotting
# =============================================================================
# 1,000,000 random predictions
def plot_scatter(p_flat: np.ndarray, t_flat: np.ndarray, out_path: Path, title_suffix: str = "") -> None:
    if len(p_flat) > 1_000_000:
        idx = np.random.choice(len(p_flat), 1_000_000, replace=False)
        p, t = p_flat[idx], t_flat[idx]
    else:
        p, t = p_flat, t_flat
        
    r = stats.pearsonr(p, t)[0] if np.std(t) > 1e-10 else 0.0
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(t, p, alpha=0.05, s=1, color="steelblue", rasterized=True)
    lim = max(t.max(), p.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", alpha=0.5, linewidth=0.8)
    ax.set_xlabel("Observed signal")
    ax.set_ylabel("Predicted signal")
    ax.set_title(f"Pred vs Obs {title_suffix} (r={r:.3f})")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# n_regions * n_tracks number of predictions, each a sum of count across the sequence length
def plot_scatter_counts(pred_sums: np.ndarray, target_sums: np.ndarray, out_path: Path, title_suffix: str = "") -> None:
    r = stats.pearsonr(pred_sums, target_sums)[0] if np.std(target_sums) > 1e-10 else 0.0
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(target_sums, pred_sums, alpha=0.15, s=5, color="steelblue", rasterized=True)
    lim = max(target_sums.max(), pred_sums.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", alpha=0.5, linewidth=0.8)
    ax.set_xlabel("Observed total count")
    ax.set_ylabel("Predicted total count")
    ax.set_title(f"Count correlation {title_suffix} (r={r:.3f})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_correlation_histogram(
    ft_values: np.ndarray, out_path: Path, native_values: np.ndarray | None = None,
    xlabel: str = "Pearson r (per region)", title: str = "Per-region correlation distribution",
    ft_label: str = "Finetuned", native_label: str = "Native",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(
        min(ft_values.min(), native_values.min() if native_values is not None else ft_values.min()),
        max(ft_values.max(), native_values.max() if native_values is not None else ft_values.max()), 51,
    )
    ax.hist(ft_values, bins=bins, alpha=0.6, color="steelblue", edgecolor="white", label=f"{ft_label} (med={np.median(ft_values):.3f})")
    if native_values is not None:
        ax.hist(native_values, bins=bins, alpha=0.5, color="forestgreen", edgecolor="white", label=f"{native_label} (med={np.median(native_values):.3f})")
    ax.axvline(np.median(ft_values), color="steelblue", linestyle="--", linewidth=1.2)
    if native_values is not None:
        ax.axvline(np.median(native_values), color="forestgreen", linestyle="--", linewidth=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_region_tracks(
    ft_pred: np.ndarray, target: np.ndarray, region_name: str, out_path: Path, res: int,
    native_pred: np.ndarray | None = None, ft_r: float | None = None, native_r: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 3))
    x = np.arange(len(target))
    ax.fill_between(x, target, alpha=0.3, color="steelblue", label="Observed")
    label_ft = "Finetuned"
    if ft_r is not None: label_ft += f" (r={ft_r:.3f})"
    ax.plot(x, ft_pred, color="crimson", linewidth=0.8, alpha=0.8, label=label_ft)
    if native_pred is not None:
        label_nat = "Native"
        if native_r is not None: label_nat += f" (r={native_r:.3f})"
        ax.plot(x, native_pred, color="forestgreen", linewidth=0.8, alpha=0.7, linestyle="--", label=label_nat)
    ax.set_title(f"{region_name} ({res}bp)", fontsize=10)
    ax.set_xlabel(f"Position ({res}bp bins)" if res > 1 else "Position (bp)")
    ax.set_ylabel("Signal")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_ism_heatmap(ism_matrix: np.ndarray, region_name: str, out_path: Path, center_pos: int) -> None:
    fig, ax = plt.subplots(figsize=(max(6, ism_matrix.shape[0] * 0.4), 3))
    vmax = np.abs(ism_matrix).max()
    im = ax.imshow(ism_matrix.T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(4))
    ax.set_yticklabels(["A", "C", "G", "T"])
    ax.set_xlabel(f"Position (centered on {center_pos})")
    ax.set_title(f"ISM: {region_name}")
    plt.colorbar(im, ax=ax, label="Effect size")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Region parsing & ISM
# =============================================================================

def parse_regions_bed(bed_path: str) -> list[dict]:
    regions = []
    with open(bed_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("track"): continue
            parts = line.split("\t")
            chrom = parts[0]
            start, end = int(parts[1]), int(parts[2])
            name = parts[3] if len(parts) > 3 else f"{chrom}:{start}-{end}"
            regions.append({"chrom": chrom, "start": start, "end": end, "name": name, "midpoint": (start + end) // 2})
    log.info("Loaded %d regions from %s", len(regions), bed_path)
    return regions

def run_ism_for_regions(model: nn.Module, genome_path: str, regions: list[dict], modality: str, ism_window_size: int, device: torch.device, out_dir: Path) -> None:
    from alphagenome_pytorch.variant_scoring.aggregations import AggregationType
    from alphagenome_pytorch.variant_scoring.inference import VariantScoringModel
    from alphagenome_pytorch.variant_scoring.scorers.center_mask import CenterMaskScorer
    from alphagenome_pytorch.variant_scoring.types import Interval, OutputType

    out_dir.mkdir(parents=True, exist_ok=True)
    scoring_model = VariantScoringModel(model=model, fasta_path=genome_path, device=device)
    output_type = OutputType(modality)
    scorer = CenterMaskScorer(output_type, width=501, aggregation_type=AggregationType.DIFF_LOG2_SUM)

    for region in tqdm(regions, desc="ISM"):
        name = region["name"]
        center_1based = region["midpoint"] + 1
        interval = Interval.centered_on(region["chrom"], center_1based, width=131072)
        try:
            ism_results = scoring_model.score_ism_variants(interval=interval, center_position=center_1based, scorers=[scorer], window_size=ism_window_size, nucleotides="ACGT", progress=False)
        except Exception as e:
            log.warning("ISM failed for %s: %s", name, e)
            continue
            
        scores = [vr[0].scores.mean().item() for vr in ism_results]
        variants = [vr[0].variant for vr in ism_results]
        if not scores: continue
        
        matrix = scoring_model.ism_matrix(variant_scores=scores, variants=variants, interval=Interval.centered_on(region["chrom"], center_1based, width=ism_window_size), multiply_by_sequence=True)
        plot_ism_heatmap(matrix.numpy(), name, out_dir / f"{name}_ism.png", center_1based)
    log.info("ISM heatmaps saved to %s", out_dir)


# =============================================================================
# Summary
# =============================================================================

def format_summary_table(ft_metrics: dict | None, native_metrics: dict | None, native_display_name: str | None, resolutions: tuple[int, ...]) -> str:
    lines = []
    for res in resolutions:
        ft = ft_metrics.get(res) if ft_metrics else None
        nat = native_metrics.get(res) if native_metrics else None
        if ft is None: continue

        lines.append(f"\n--- {res}bp resolution ---")
        header = f"{'Metric':<28}{'Finetuned':>12}"
        if nat is not None: header += f"  {'Native(' + (native_display_name or '?') + ')':>20}"
        lines.append(header)
        lines.append("-" * len(header))

        rows = [("Profile r (mean)", "profile_pearson_r_mean"), ("Profile r (median)", "profile_pearson_r_median"),
                ("Count r", "count_pearson_r"), ("JSD (mean)", "jsd_mean"), ("JSD (median)", "jsd_median"),
                ("MSE", "mse"), ("Spearman (global)", "spearman_global")]
        for label, key in rows:
            line = f"{label:<28}{ft[key]:>12.4f}"
            if nat is not None: line += f"  {nat[key]:>20.4f}"
            lines.append(line)
        lines.append(f"{'N regions':<28}{ft['n_regions']:>12d}")
    return "\n".join(lines)

def save_summary_json(ft_metrics: dict | None, native_metrics: dict | None, checkpoint_meta: dict, native_info: dict | None, loss: float | None, out_path: Path) -> None:
    def _clean(m: dict) -> dict: return {k: v for k, v in m.items() if not isinstance(v, np.ndarray)}
    data: dict = {"checkpoint": checkpoint_meta}
    if loss is not None: data["loss"] = loss
    if ft_metrics: data["finetuned"] = {str(res): _clean(m) for res, m in ft_metrics.items()}
    if native_metrics: data["native"] = {str(res): _clean(m) for res, m in native_metrics.items()}
    if native_info: data["native_track"] = native_info
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate fine-tuned AlphaGenome models")
    p.add_argument("--checkpoint", required=True, help="Finetuned checkpoint path")
    p.add_argument("--pretrained-weights", required=True, help="Pretrained trunk weights")
    p.add_argument("--output-dir", required=True, help="Output directory")
    p.add_argument("--metrics", action="store_true", help="Compute test set metrics")
    p.add_argument("--regions", action="store_true", help="Plot predefined regions")
    p.add_argument("--ism", action="store_true", help="Run ISM on predefined regions")
    p.add_argument("--genome", help="Reference genome FASTA")
    p.add_argument("--bigwig", nargs="+", help="BigWig signal file(s)")
    p.add_argument("--test-bed", help="Test split BED file")
    p.add_argument("--regions-bed", help="Named regions BED4 file")
    p.add_argument("--native-biosample", help="Biosample name for native head comparison")
    p.add_argument("--native-track-index", type=int, help="Direct track index for native head")
    p.add_argument("--sequence-length", type=int, default=131072)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-regions", type=int, default=2000, help="Max regions for metrics (0=all)")
    p.add_argument("--ism-window-size", type=int, default=21)
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--device", default=None, help="Device (default: auto)")
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output: %s", out_dir)
    log.info("Device: %s", device)

    any_explicit = args.metrics or args.regions or args.ism
    run_metrics = args.metrics or (not any_explicit and args.test_bed is not None)
    run_regions = args.regions or (not any_explicit and args.regions_bed is not None)
    run_ism = args.ism

    if run_metrics and (not args.test_bed or not args.bigwig): sys.exit("Error: --metrics requires --test-bed and --bigwig")
    if run_regions and (not args.regions_bed or not args.bigwig): sys.exit("Error: --regions requires --regions-bed and --bigwig")
    if run_ism and (not args.regions_bed or not args.genome): sys.exit("Error: --ism requires --regions-bed and --genome")

    want_native = bool(args.native_biosample or args.native_track_index is not None)

    # ---- Load finetuned model ----
    model, ckpt_meta = load_finetuned_model(args.checkpoint, args.pretrained_weights, device)
    modality = ckpt_meta["modality"][0] if isinstance(ckpt_meta["modality"], list) else ckpt_meta["modality"]
    resolutions = tuple(ckpt_meta["resolutions"])

    if any(isinstance(r, str) for r in resolutions):
        log.warning(f"Corrupted string detected in resolutions {resolutions}. Overriding to (1,).")
        resolutions = (1, 128)

    track_names = ckpt_meta["track_names"].get(modality, []) if isinstance(ckpt_meta["track_names"], dict) else ckpt_meta["track_names"]
    log.info("Modality: %s, Tracks: %s, Resolutions: %s", modality, track_names, resolutions)

    native_model, native_track_idx, native_display_name = None, None, None
    if want_native:
        native_model, native_track_idx, native_display_name = load_native_model(args.pretrained_weights, args.native_biosample, args.native_track_index, modality, device)

    ft_metrics_by_res, native_metrics_by_res, loss = {}, {}, None

    # ---- Feature: metrics (Memory Optimized Streaming) ----
    if run_metrics:
        log.info("=" * 60)
        log.info("Computing test set metrics (Memory Optimized)")
        log.info("=" * 60)
        metrics_dir = out_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)

        dataset = GenomicDataset(genome_fasta=args.genome, bigwig_files=args.bigwig, bed_file=args.test_bed, resolutions=resolutions, sequence_length=args.sequence_length)
        if args.max_regions > 0 and len(dataset) > args.max_regions:
            indices = np.random.default_rng(42).choice(len(dataset), args.max_regions, replace=False).tolist()
            dataset_sub = Subset(dataset, indices)
        else:
            dataset_sub = dataset

        loader = DataLoader(dataset_sub, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_genomic)

        # Finetuned evaluation
        ft_metrics_by_res, loss = evaluate_test_split(model, modality, loader, device, resolutions)
        log.info("Loss: %.4f", loss)

        for res, ft_m in ft_metrics_by_res.items():
            log.info("  Resolution %dbp", res)
            log.info("  Profile r (mean):  %.4f", ft_m["profile_pearson_r_mean"])
            log.info("  Profile r (median):%.4f", ft_m["profile_pearson_r_median"])
            log.info("  Count r:           %.4f", ft_m["count_pearson_r"])
            log.info("  JSD (mean):        %.4f", ft_m["jsd_mean"])

            plot_scatter(ft_m["_scatter_p"], ft_m["_scatter_t"], metrics_dir / f"scatter_test_{res}bp.png", title_suffix=f"({res}bp)")
            plot_scatter_counts(ft_m["_pred_counts"], ft_m["_target_counts"], metrics_dir / f"scatter_counts_test_{res}bp.png", title_suffix=f"({res}bp)")

        # Native evaluation
        if native_model is not None:
            model.cpu()
            native_model = native_model.to(device)
            native_metrics_by_res = evaluate_native_test_split(native_model, modality, native_track_idx, loader, device, resolutions)
            
            for res in resolutions:
                if res in native_metrics_by_res and native_metrics_by_res[res]:
                    nat_m = native_metrics_by_res[res]
                    log.info("Native %dbp — Profile r: %.4f, Count r: %.4f, JSD: %.4f", res, nat_m["profile_pearson_r_mean"], nat_m["count_pearson_r"], nat_m["jsd_mean"])
                    # for later
                    # plot_scatter(nat_m["_scatter_p"], nat_m["_scatter_t"], metrics_dir / f"native_scatter_test_{res}bp.png", title_suffix=f"(test, {res}bp)")
                    # plot_scatter_counts(nat_m["_pred_counts"], nat_m["_target_counts"], metrics_dir / f"native_scatter_counts_test_{res}bp.png", title_suffix=f"(test, {res}bp)")
            native_model.cpu()
            model = model.to(device)

        # Plot Overlays
        for res in resolutions:
            ft_m = ft_metrics_by_res.get(res)
            nat_m = native_metrics_by_res.get(res)
            if ft_m is None: continue

            plot_correlation_histogram(
                ft_m["profile_pearson_r_all"], metrics_dir / f"correlation_hist_test_{res}bp.png",
                native_values=nat_m["profile_pearson_r_all"] if nat_m else None, xlabel="Pearson r (per region)",
                title=f"Profile correlation distribution ({res}bp)", native_label=f"Native ({native_display_name})" if native_display_name else "Native",
            )
            plot_correlation_histogram(
                ft_m["jsd_all"], metrics_dir / f"jsd_hist_test_{res}bp.png",
                native_values=nat_m["jsd_all"] if nat_m else None, xlabel="JSD (per region)",
                title=f"JSD distribution ({res}bp)", native_label=f"Native ({native_display_name})" if native_display_name else "Native",
            )

        if args.save_predictions:
            log.warning("Skipping --save-predictions: Array caching disabled in memory-optimized mode to prevent OOM errors.")

    # ---- Feature: region exploration ----
    if run_regions:
        log.info("=" * 60)
        log.info("Region exploration")
        log.info("=" * 60)
        regions_dir = out_dir / "regions"
        regions_dir.mkdir(exist_ok=True)
        regions = parse_regions_bed(args.regions_bed)

        region_dataset = GenomicDataset(genome_fasta=args.genome, bigwig_files=args.bigwig, bed_file=args.regions_bed, resolutions=resolutions, sequence_length=args.sequence_length)
        region_loader = DataLoader(region_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_genomic)

        ft_region_preds, region_targets = predict_regions_split(model, modality, region_loader, device, resolutions)

        native_region_preds = None
        if native_model is not None:
            model.cpu()
            native_model = native_model.to(device)
            native_region_preds = predict_native_regions_split(native_model, modality, native_track_idx, region_loader, device, resolutions)
            native_model.cpu()
            model = model.to(device)

        for res in resolutions:
            if res not in ft_region_preds: continue
            ft_p = ft_region_preds[res]
            t = region_targets[res]
            nat_p = native_region_preds[res] if native_region_preds and res in native_region_preds else None

            for i, region in enumerate(regions):
                if i >= len(ft_p): break
                ft_track = ft_p[i, :, 0]
                obs_track = t[i, :, 0]
                nat_track = nat_p[i, :, 0] if nat_p is not None else None

                ft_r = stats.pearsonr(ft_track, obs_track)[0] if np.std(obs_track) > 1e-10 else 0.0
                nat_r = stats.pearsonr(nat_track, obs_track)[0] if (nat_track is not None and np.std(obs_track) > 1e-10) else None

                safe_name = region["name"].replace("/", "_").replace(" ", "_")
                plot_region_tracks(ft_track, obs_track, region["name"], regions_dir / f"{safe_name}_{res}bp.png", res=res, native_pred=nat_track, ft_r=ft_r, native_r=nat_r)
        log.info("Region plots saved to %s", regions_dir)

    # ---- Feature: ISM ----
    if run_ism:
        log.info("=" * 60)
        log.info("In-silico mutagenesis")
        log.info("=" * 60)
        run_ism_for_regions(model, args.genome, parse_regions_bed(args.regions_bed), modality, args.ism_window_size, device, out_dir / "ism")

    # ---- Summary ----
    summary_text = format_summary_table(ft_metrics_by_res if ft_metrics_by_res else None, native_metrics_by_res if native_metrics_by_res else None, native_display_name, resolutions)
    if summary_text:
        print(summary_text)
        with open(out_dir / "summary.txt", "w") as f:
            f.write(summary_text)

    native_info = {"biosample": native_display_name, "track_index": native_track_idx} if native_display_name is not None else None
    save_summary_json(
        ft_metrics_by_res if ft_metrics_by_res else None,
        native_metrics_by_res if native_metrics_by_res else None,
        {k: v for k, v in ckpt_meta.items() if not isinstance(v, (list, dict)) or k in ("modality", "track_names")},
        native_info, loss, out_dir / "summary.json"
    )

if __name__ == "__main__":
    main()