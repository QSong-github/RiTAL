#!/usr/bin/env python3
"""RiTAL — seed-rotation training.

Changes from v31
----------------
- Seed rotation replaces reload_patience:
    After `seed_rotation.patience` (default 3) consecutive epochs without
    val_RP improvement, the trainer:
      1. Reloads best.pt model weights
      2. Increments the random seed (base_seed + N * 1009)
      3. Re-seeds random / numpy / torch / cuda
      4. Creates a fresh AdamW optimizer at the original LR
      5. Creates a fresh OneCycleLR scheduler covering remaining epochs
      6. Resets no_improve_count to 0
- seed_counter and current_seed are saved in every checkpoint so that
  auto-resubmitted jobs resume the exact same seed rotation state.

Usage (same CLI as v31):
    python scripts/train.py \\
        --config configs/hariboss5fd.yaml \\
        --fold 0
    # resume:
    python scripts/train.py \\
        --config configs/hariboss5fd.yaml \\
        --fold 0 --resume checkpoints/hariboss5fd/fold0/last.pt
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import math
import random
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

# ── Project path ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# ── Load helper functions from v31 to avoid duplication ──────────────────────
# We exec the v31 script as a sub-module; its sys.path setup runs first which
# guarantees rital.* are importable for both modules.
_TRAINER_LIB = Path(__file__).parent / "_trainer_lib.py"
_spec = importlib.util.spec_from_file_location("_rital_trainer_lib", _TRAINER_LIB)
_lib = importlib.util.module_from_spec(_spec)
sys.modules["_rital_trainer_lib"] = _lib
_spec.loader.exec_module(_lib)

# Re-export the pieces we need
load_pkl_records        = _lib.load_pkl_records
build_rna_ligand_map    = _lib.build_rna_ligand_map
eval_rank_percentile    = _lib.eval_rank_percentile
_rank_loss_step         = _lib._rank_loss_step
_refresh_hard_neg_cache = _lib._refresh_hard_neg_cache
_to_device              = _lib._to_device
smiles_list_to_fp2_tensor = _lib.smiles_list_to_fp2_tensor
_DEFAULT_PKL            = _lib._DEFAULT_PKL
_DEFAULT_DECOY_LOOKUP   = _lib._DEFAULT_DECOY_LOOKUP
_DEFAULT_RNA3D_DIRS     = _lib._DEFAULT_RNA3D_DIRS

# Model classes (same underlying module thanks to sys.modules cache)
from rital.model import RiTALModel
from rital.paths import expand_config, data_root
from rital.featurizers      import build_rna_graph_cached, build_lig_graph_cached
from rital.graph_data       import batch_rna_graphs, batch_ligand_graphs


# ═════════════════════════════════════════════════════════════════════════════
# Per-epoch DeepCoy eval (DUD-E + DEKOIS) — inline, reuses live model
# ═════════════════════════════════════════════════════════════════════════════

_DEEPCOY_BASE = data_root() / "deepcoy"
_DEEPCOY_FILES = {
    "dude":   _DEEPCOY_BASE / "molecules_ligands_DUDE_smiles.pickle",
    "dekois": _DEEPCOY_BASE / "molecules_ligands_DEKOIS_smiles.pickle",
}
_DEEPCOY_CACHE: dict[str, dict] = {}


def _load_deepcoy_dict(name: str) -> dict:
    if name not in _DEEPCOY_CACHE:
        import pickle
        with open(_DEEPCOY_FILES[name], "rb") as f:
            _DEEPCOY_CACHE[name] = pickle.load(f)
    return _DEEPCOY_CACHE[name]


def _deepcoy_eval_one_set(
    model: torch.nn.Module,
    test_recs: list[dict],
    decoy_dict: dict,
    device: torch.device,
    chunk: int = 64,
) -> tuple[float, int]:
    """Run DeepCoy rank-percentile eval on test_recs. Returns (mean_pct, n_used)."""
    model.eval()
    pcts: list[float] = []
    with torch.no_grad():
        for rec in test_recs:
            lig_id  = rec.get("ligand_id", "")
            nat_smi = rec.get("native_smiles", "")
            dec_smiles = decoy_dict.get(lig_id, [])
            if not dec_smiles or not nat_smi:
                continue
            try:
                rg = build_rna_graph_cached(rec["pkl_key"], str(rec["json_path"]))
            except Exception:
                continue
            all_smi = [nat_smi] + list(dec_smiles)
            all_scores: list[float] = []
            ok = True
            for i in range(0, len(all_smi), chunk):
                sub = all_smi[i : i + chunk]
                try:
                    lig_gs = [build_lig_graph_cached(s) for s in sub]
                except Exception:
                    ok = False; break
                rna_bg = _to_device(batch_rna_graphs([rg] * len(sub)), device)
                lig_bg = _to_device(batch_ligand_graphs(lig_gs), device)
                sc = model(rna_bg, lig_bg)["rank_score"].squeeze().detach().cpu()
                if sc.dim() == 0: sc = sc.unsqueeze(0)
                all_scores.extend(sc.tolist())
            if not ok or len(all_scores) < 2:
                continue
            pcts.append(float(np.mean([s < all_scores[0] for s in all_scores[1:]])))
    model.train()
    if not pcts:
        return float("nan"), 0
    return float(np.mean(pcts)), len(pcts)


def _per_epoch_deepcoy_eval(
    model: torch.nn.Module,
    test_recs: list[dict],
    device: torch.device,
    out_dir: Path,
    epoch: int,
    sets: list[str],
    max_pockets: int = 0,
) -> dict:
    """Run DeepCoy eval on selected sets, append jsonl, return result dict."""
    import json, time
    t0 = time.time()
    recs = test_recs
    if max_pockets > 0 and len(recs) > max_pockets:
        recs = random.sample(recs, max_pockets)
    out: dict[str, Any] = {"epoch": epoch}
    for s in sets:
        try:
            dd = _load_deepcoy_dict(s)
        except Exception as e:
            print(f"[ep{epoch:02d}/deepcoy:{s}] load failed: {e}", flush=True)
            continue
        mean_pct, n = _deepcoy_eval_one_set(model, recs, dd, device)
        out[f"{s}_mean"] = mean_pct
        out[f"{s}_n"]    = n
        print(f"[ep{epoch:02d}/deepcoy:{s}] mean_RP={mean_pct:.4f}  n={n}", flush=True)
    out["elapsed_s"] = round(time.time() - t0, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "per_epoch_deepcoy.jsonl", "a") as f:
        f.write(json.dumps(out) + "\n")
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Seed + optimizer + scheduler helpers
# ═════════════════════════════════════════════════════════════════════════════

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_optimizer(model: torch.nn.Module, lr: float, wd: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(wd))


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    lr: float,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.OneCycleLR | None:
    if total_steps <= 0:
        return None
    pct_start = min(float(warmup_steps) / total_steps, 0.30)
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(lr),
        total_steps=total_steps,
        pct_start=pct_start,
        final_div_factor=1.0 / max(min_lr_ratio, 1e-9),
        anneal_strategy="cos",
    )


# ═════════════════════════════════════════════════════════════════════════════
# Main training function (seed rotation)
# ═════════════════════════════════════════════════════════════════════════════

def train(
    cfg: dict[str, Any],
    fold_idx: int,
    resume: Path | None = None,
    init: Path | None = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[rital] fold={fold_idx}  device={device}", flush=True)

    # ── Seed rotation config ────────────────────────────────────────────────
    base_seed              = int(cfg.get("seed", 42))
    seed_rotation_patience = int(cfg.get("seed_rotation", {}).get("patience", 3))
    seed_counter           = 0
    current_seed           = base_seed
    _set_seed(current_seed)
    print(
        f"[seed] base_seed={base_seed}  rotation_patience={seed_rotation_patience}",
        flush=True,
    )

    # ── Data ───────────────────────────────────────────────────────────────
    data_cfg          = cfg["data"]
    pkl_path          = Path(data_cfg.get("pkl_path", str(_DEFAULT_PKL)))
    rna3d_dirs        = data_cfg.get("rna3d_dirs", _DEFAULT_RNA3D_DIRS)
    decoy_lookup_path = Path(data_cfg.get("decoy_lookup_path", str(_DEFAULT_DECOY_LOOKUP)))

    records     = load_pkl_records(pkl_path, rna3d_dirs, decoy_lookup_path)
    rna_lig_map = build_rna_ligand_map(records)
    print(f"[data] rna_lig_map: {len(rna_lig_map)} RNA systems", flush=True)

    train_recs = [
        r for r in records
        if len(r["train_split"]) > fold_idx and bool(r["train_split"][fold_idx])
    ]
    test_recs = [
        r for r in records
        if len(r["train_split"]) > fold_idx and not bool(r["train_split"][fold_idx])
    ]
    print(f"[split] fold {fold_idx}: train={len(train_recs)}, test={len(test_recs)}", flush=True)
    if not train_recs:
        raise RuntimeError(f"No training records for fold {fold_idx}")

    # ── Model ──────────────────────────────────────────────────────────────
    model_cfg      = cfg["model"]
    use_pretrained = bool(model_cfg.get("use_pretrained_ligand", False))
    use_fp2_ligand = bool(model_cfg.get("use_fp2_ligand", False))
    model          = RiTALModel(model_cfg=model_cfg).to(device)
    if use_pretrained or use_fp2_ligand:
        use_pretrained = False  # RiTAL uses GIN by default

    # ── Restore from checkpoint ────────────────────────────────────────────
    start_epoch = 0
    best_rp     = 0.0
    if resume is not None and Path(resume).exists():
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        start_epoch  = int(ckpt.get("epoch", 0)) + 1
        best_rp      = float(ckpt.get("best_rp", 0.0))
        seed_counter = int(ckpt.get("seed_counter", 0))
        current_seed = int(ckpt.get("current_seed", base_seed))
        _set_seed(current_seed)
        print(
            f"[resume] epoch={start_epoch}  best_rp={best_rp:.4f}"
            f"  seed={current_seed} (rotation #{seed_counter})",
            flush=True,
        )
    elif init is not None and Path(init).exists():
        ckpt = torch.load(init, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"[init] warm-started from {init}", flush=True)

    # ── Training hyper-parameters ──────────────────────────────────────────
    train_cfg       = cfg["train"]
    lr              = float(train_cfg["lr"])
    wd              = float(train_cfg.get("weight_decay", 1e-4))
    grad_clip       = float(train_cfg.get("grad_clip", 1.0))
    epochs          = int(train_cfg["epochs"])
    steps_per_epoch = int(train_cfg["steps_per_epoch"])
    warmup_steps    = int(train_cfg.get("lr_scheduler", {}).get("warmup_steps", 100))
    min_lr_ratio    = float(train_cfg.get("lr_scheduler", {}).get("min_lr_ratio", 0.02))

    optimizer    = _make_optimizer(model, lr, wd)
    total_steps  = epochs * steps_per_epoch
    scheduler    = _make_scheduler(optimizer, lr, total_steps, warmup_steps, min_lr_ratio)

    # Fast-forward scheduler to match current start_epoch
    if start_epoch > 0 and scheduler is not None:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for _ in range(start_epoch * steps_per_epoch):
                scheduler.step()

    # ── Loss config ────────────────────────────────────────────────────────
    loss_cfg              = cfg.get("loss", {})
    margin_start          = float(loss_cfg.get("ranking_margin", 0.6))
    tau                   = float(loss_cfg.get("contrastive_temperature", 0.2))
    lambda_contrast       = float(loss_cfg.get("lambda_contrast", 0.1))
    lambda_rank           = float(loss_cfg.get("lambda_rank", 0.5))
    lambda_rna_contrast   = float(loss_cfg.get("lambda_rna_contrast", 0.3))
    lambda_soft_rp        = float(loss_cfg.get("lambda_soft_rp", 2.0))
    soft_rp_temp          = float(loss_cfg.get("soft_rp_temp", 0.07))
    lambda_site           = float(loss_cfg.get("lambda_site", 0.3))
    site_start_ep         = int(loss_cfg.get("site_start_epoch", 7))
    margin_restart_epochs = int(loss_cfg.get("margin_restart_epochs", 8))
    hard_margin_topk      = int(loss_cfg.get("hard_margin_topk", 0))
    use_circle_loss       = bool(loss_cfg.get("use_circle_loss", False))
    adaptive_margin       = bool(loss_cfg.get("adaptive_margin", False))
    circle_gamma          = float(loss_cfg.get("circle_gamma", 64.0))

    # ── Hard negative mining ───────────────────────────────────────────────
    hn_cfg         = cfg.get("hard_negative_mining", {})
    hn_enabled     = bool(hn_cfg.get("enabled", False))
    hn_start_ep    = int(hn_cfg.get("start_epoch", 0))
    hn_refresh_ev  = int(hn_cfg.get("refresh_every", 2))
    hn_n_cand      = int(hn_cfg.get("n_candidates", 40))
    hn_keep_frac   = float(hn_cfg.get("keep_frac", 0.5))
    hard_neg_cache: dict[str, list[str]] = {}

    # ── Curriculum ─────────────────────────────────────────────────────────
    curriculum  = train_cfg.get("decoy_curriculum", {})
    K_s1        = int(curriculum.get("stage1_negatives_per_positive", 48))
    K_s2        = int(curriculum.get("stage2_negatives_per_positive", 64))
    K_s3        = int(curriculum.get("stage3_negatives_per_positive", 72))
    cross_ratio = float(curriculum.get("cross_pocket_ratio", 0.667))
    sched_cfg   = cfg.get("schedule", {})
    stage1_ep   = int(sched_cfg.get("stage1_epochs", 1))
    stage2_ep   = int(sched_cfg.get("stage2_epochs", 3))

    batch_size = int(train_cfg["batch_size"])
    log_every  = int(cfg.get("output", {}).get("log_every", 50))
    val_max    = int(cfg.get("output", {}).get("val_max_pockets", 0))

    # ── Output directory ───────────────────────────────────────────────────
    out_dir = Path(
        cfg["output"]["checkpoint_dir"].replace("fold0", f"fold{fold_idx}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] checkpoints → {out_dir}", flush=True)

    no_improve_count = 0

    # ══════════════════════════════════════════════════════════════════════
    # Epoch loop
    # ══════════════════════════════════════════════════════════════════════
    for epoch in range(start_epoch, epochs):

        # Dynamic margin (SMARTBind tanh-decay with periodic restart)
        if margin_restart_epochs > 0:
            t_in_cycle = epoch % margin_restart_epochs
            margin = margin_start * (
                1.0 - math.tanh(2.0 * t_in_cycle / max(margin_restart_epochs, 1))
            )
        else:
            margin = margin_start

        # K curriculum
        if epoch < stage1_ep:
            K = K_s1
        elif epoch < stage1_ep + stage2_ep:
            K = K_s2
        else:
            K = K_s3
        n_cross = max(1, int(K * cross_ratio))
        n_hard  = K - n_cross

        model.train()
        running_loss = 0.0
        n_steps_ok   = 0

        for step in range(steps_per_epoch):
            idx   = np.random.choice(len(train_recs), batch_size,
                                     replace=len(train_recs) < batch_size)
            batch = [train_recs[i] for i in idx]
            optimizer.zero_grad()
            try:
                _hn = hard_neg_cache if (hn_enabled and epoch >= hn_start_ep) else None
                infonce, margin_l, aux_loss = _rank_loss_step(
                    model, batch, device, margin, tau, hard_margin_topk,
                    use_pretrained,
                    n_cross=n_cross, n_hard=n_hard,
                    all_train_recs=train_recs, rna_lig_map=rna_lig_map,
                    hard_neg_cache=_hn,
                    lambda_site=lambda_site if epoch >= site_start_ep else 0.0,
                    lambda_dock=0.0,
                    lambda_rna_contrast=lambda_rna_contrast,
                    use_fp2_ligand=use_fp2_ligand,
                    use_circle_loss=use_circle_loss,
                    adaptive_margin=adaptive_margin,
                    circle_gamma=circle_gamma,
                    lambda_soft_rp=lambda_soft_rp,
                    soft_rp_temp=soft_rp_temp,
                    moco_queue=None,
                    queue_in_smol_cts=False,
                )
                loss = lambda_contrast * infonce + lambda_rank * margin_l + aux_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                running_loss += loss.item()
                n_steps_ok   += 1
            except Exception as e:
                print(f"[ep{epoch:02d} step{step+1:04d}] err: {e}", flush=True)
                continue

            if (step + 1) % log_every == 0:
                avg    = running_loss / max(n_steps_ok, 1)
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"[ep{epoch:02d} step{step+1:04d}/{steps_per_epoch}]"
                    f"  loss={avg:.4f}  K={K}(x={n_cross}+h={n_hard})"
                    f"  M={margin:.3f}  lr={lr_now:.2e}  seed#{seed_counter}",
                    flush=True,
                )

        # ── Validation ────────────────────────────────────────────────────
        val_recs = (
            random.sample(test_recs, val_max)
            if val_max > 0 and len(test_recs) > val_max
            else test_recs
        )
        rp       = eval_rank_percentile(
            model, val_recs, device, use_pretrained,
            rna_lig_map=rna_lig_map,
            use_fp2_ligand=use_fp2_ligand,
        )
        avg_loss = running_loss / max(n_steps_ok, 1)
        print(
            f"[ep{epoch:02d}]  loss={avg_loss:.4f}"
            f"  val_RP={rp:.4f}  K={K}  M={margin:.3f}"
            f"  seed={current_seed} (rot#{seed_counter})",
            flush=True,
        )

        # ── Per-epoch DeepCoy eval (DUD-E + DEKOIS) ──────────────────────
        pe_cfg = cfg.get("train", {}).get("per_epoch_deepcoy", {}) or {}
        if bool(pe_cfg.get("enabled", False)) and (epoch + 1) % int(pe_cfg.get("every_n_epochs", 1)) == 0:
            try:
                _per_epoch_deepcoy_eval(
                    model, test_recs, device, out_dir, epoch,
                    sets=list(pe_cfg.get("sets", ["dude", "dekois"])),
                    max_pockets=int(pe_cfg.get("max_pockets", 0)),
                )
            except Exception as e:
                print(f"[ep{epoch:02d}/deepcoy] FAILED: {e}", flush=True)

        torch.cuda.empty_cache()
        gc.collect()

        # ── Hard negative mining ──────────────────────────────────────────
        if (
            hn_enabled and epoch >= hn_start_ep
            and (epoch + 1) % hn_refresh_ev == 0
        ):
            hard_neg_cache = _refresh_hard_neg_cache(
                model, train_recs, device, use_pretrained,
                n_candidates=hn_n_cand, keep_frac=hn_keep_frac,
                rna_lig_map=rna_lig_map,
            )

        # ── Checkpoint ───────────────────────────────────────────────────
        ckpt_data = {
            "epoch":                epoch,
            "fold":                 fold_idx,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_rank_percentile":  rp,
            "avg_loss":             avg_loss,
            "best_rp":              best_rp,
            "seed_counter":         seed_counter,
            "current_seed":         current_seed,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt_data, out_dir / f"ep{epoch:03d}.pt")
        torch.save(ckpt_data, out_dir / "last.pt")

        if rp > best_rp:
            best_rp = rp
            no_improve_count = 0
            ckpt_data["best_rp"] = best_rp
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(ckpt_data, out_dir / "best.pt")
            print(
                f"[ep{epoch:02d}] ★ new best RP={best_rp:.4f} → best.pt"
                f"  (seed={current_seed} rot#{seed_counter})",
                flush=True,
            )
        else:
            no_improve_count += 1
            print(
                f"[ep{epoch:02d}] no improvement"
                f" ({no_improve_count}/{seed_rotation_patience})"
                f"  best_rp={best_rp:.4f}",
                flush=True,
            )

            # ── Seed rotation ─────────────────────────────────────────────
            if no_improve_count >= seed_rotation_patience:
                seed_counter += 1
                current_seed  = base_seed + seed_counter * 1009   # prime step
                _set_seed(current_seed)

                # Reload best model weights
                best_ckpt_path = out_dir / "best.pt"
                if best_ckpt_path.exists():
                    _bc = torch.load(
                        best_ckpt_path, map_location=device, weights_only=False
                    )
                    model.load_state_dict(_bc["model_state_dict"])
                    print(
                        f"[seed_rot #{seed_counter}] reloaded best.pt"
                        f" RP={best_rp:.4f}",
                        flush=True,
                    )
                else:
                    print(
                        f"[seed_rot #{seed_counter}] no best.pt yet,"
                        " keeping current weights",
                        flush=True,
                    )

                # Fresh optimizer + warmup scheduler for remaining epochs
                remaining_steps = (epochs - epoch - 1) * steps_per_epoch
                optimizer  = _make_optimizer(model, lr, wd)
                scheduler  = _make_scheduler(
                    optimizer, lr, remaining_steps, warmup_steps, min_lr_ratio
                )
                no_improve_count = 0
                print(
                    f"[seed_rot #{seed_counter}] new seed={current_seed},"
                    f" fresh warmup, {remaining_steps} steps remaining"
                    f" ({epochs - epoch - 1} epochs left)",
                    flush=True,
                )

    print(
        f"\n[done] fold={fold_idx}  best_val_RP={best_rp:.4f}"
        f"  total_seed_rotations={seed_counter}"
    )


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RiTAL — seed rotation training"
    )
    p.add_argument("--config", type=Path, required=True, help="YAML config file")
    p.add_argument("--fold",   type=int, default=0,  help="Fold index (0-4)")
    p.add_argument("--resume", type=Path, default=None, help="Resume from .pt checkpoint")
    p.add_argument("--init",   type=Path, default=None, help="Warm-start weights only")
    p.add_argument("--dry-run", action="store_true", help="Data + model check only")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        cfg = expand_config(yaml.safe_load(f))

    if args.dry_run:
        data_cfg = cfg["data"]
        records  = load_pkl_records(
            Path(data_cfg.get("pkl_path", str(_DEFAULT_PKL))),
            data_cfg.get("rna3d_dirs", _DEFAULT_RNA3D_DIRS),
            Path(data_cfg.get("decoy_lookup_path", str(_DEFAULT_DECOY_LOOKUP))),
        )
        train_recs = [
            r for r in records
            if len(r["train_split"]) > args.fold and r["train_split"][args.fold]
        ]
        test_recs = [
            r for r in records
            if len(r["train_split"]) > args.fold and not r["train_split"][args.fold]
        ]
        model = RiTALModel(cfg["model"])
        print(
            f"[dry-run] fold={args.fold}: train={len(train_recs)}, test={len(test_recs)}"
        )
        print(
            f"[dry-run] model params: {sum(p.numel() for p in model.parameters()):,}"
        )
        print(
            f"[dry-run] seed_rotation.patience="
            f"{cfg.get('seed_rotation', {}).get('patience', 3)}"
        )
        print("[dry-run] OK — all checks passed")
        return

    train(cfg, fold_idx=args.fold, resume=args.resume, init=args.init)


if __name__ == "__main__":
    main()
