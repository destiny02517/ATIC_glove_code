#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Tactile evaluation — Analysis 4 only (Explainability: Integrated Gradients + 1D Grad‑CAM)
# Simplified from a multi‑analysis script. All analyses except (4) have been removed.

import os, pathlib, warnings
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------
# IO utils
# ----------------------------
def _ensure_parent(p):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); return p

def save_df_csv(path, df):
    p = _ensure_parent(path)
    df.to_csv(p, index=False, encoding="utf-8-sig")

# ----------------------------
# Colors
# ----------------------------
_BASE_PALETTE = ["#6AF07A", "#F7E364", "#FFAD33", "#FF99C8", "#F2545B", "#A0CED9", "#3A6EA5", "#796465"]

def _hex_to_rgb01(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16)/255.0 for i in (0,2,4))

def _rgb01_to_hex(rgb):
    r,g,b = [int(max(0,min(1,x))*255) for x in rgb]
    return f"#{r:02X}{g:02X}{b:02X}"

def _interp_rgb(a, b, t: float):
    return (a[0]*(1-t)+b[0]*t, a[1]*(1-t)+b[1]*t, a[2]*(1-t)+b[2]*t)

def make_class_palette(n_classes: int):
    base = [_hex_to_rgb01(h) for h in _BASE_PALETTE]
    k = len(base)
    if n_classes <= k:
        idxs = np.linspace(0, k-1, n_classes)
        out = []
        for t in idxs:
            i = int(np.floor(t))
            if i >= k-1:
                out.append(base[-1])
            else:
                alpha = t - i
                out.append(_interp_rgb(base[i], base[i+1], alpha))
        return [_rgb01_to_hex(c) for c in out]
    else:
        out = []
        for j in range(n_classes):
            u = j/(n_classes-1) if n_classes>1 else 0.0
            s = u*(k-1)
            i = min(int(np.floor(s)), k-2)
            alpha = s - i
            out.append(_interp_rgb(base[i], base[i+1], alpha))
        return [_rgb01_to_hex(c) for c in out]

def class_color_list(labels: list):
    return make_class_palette(len(labels))

# ----------------------------
# Models
# ----------------------------
class BranchCNN(nn.Module):
    def __init__(self, length: int, hidden_dim: int = 512, conv_channels=(32, 64, 128)):
        super().__init__()
        c1, c2, c3 = conv_channels
        self.net = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=5, padding=2), nn.ReLU(inplace=True), nn.MaxPool1d(2, 2),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2), nn.ReLU(inplace=True), nn.MaxPool1d(2, 2),
            nn.Conv1d(c2, c3, kernel_size=5, padding=2), nn.ReLU(inplace=True), nn.MaxPool1d(2, 2),
            nn.Dropout(0.5),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, length)
            feat = self.net(dummy)
            flat = feat.view(1, -1).shape[1]
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        return self.fc(self.net(x))

class SingleModalityModel(nn.Module):
    def __init__(self, length: int, num_classes: int, hidden_dim: int = 512):
        super().__init__()
        self.branch = BranchCNN(length, hidden_dim=hidden_dim)
        self.cls = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        pen = self.branch(x)
        out = self.cls(pen)
        return out, pen

# Small helpers for model loading

def disable_inplace_relu(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.ReLU) and getattr(m, "inplace", False):
            m.inplace = False

def safe_load_weights(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)

# ----------------------------
# Data loading (single‑modality only)
# ----------------------------

def read_matrix_any(path: pathlib.Path, sheet="Sheet1"):
    suf = path.suffix.lower()
    if suf in (".xls", ".xlsx"):
        arr = np.asarray(pd.read_excel(path, sheet_name=sheet))
    elif suf == ".csv":
        arr = np.asarray(pd.read_csv(path, header=None))
    else:
        raise ValueError(f"Unsupported file: {path}")
    return arr


def ensure_length(A: np.ndarray, target_len: int):
    L, N = A.shape
    if L == target_len: return A
    if L > target_len:
        s = (L - target_len)//2
        return A[s:s+target_len, :]
    pad = target_len - L
    t = pad//2; b = pad - t
    return np.pad(A, ((t,b),(0,0)), mode="constant")


def find_file_for_class(base: pathlib.Path, cls_idx: int):
    for ext in (".xls", ".xlsx", ".csv"):
        p = base / f"{cls_idx}{ext}"
        if p.exists(): return p
    raise FileNotFoundError(f"No file for class {cls_idx} in {base}")


def split_counts(total_n: int, train_spec, val_spec, test_spec, shuffle=False, rng=None):
    def parse(v):
        return (v, True) if (isinstance(v, float) and v <= 1.0) else (int(v), False)
    tr_raw, tr_is_ratio = parse(train_spec)
    va_raw, va_is_ratio = parse(val_spec)
    te_raw, te_is_ratio = parse(test_spec)
    if tr_is_ratio or va_is_ratio or te_is_ratio:
        s = (tr_raw if tr_is_ratio else 0) + (va_raw if va_is_ratio else 0) + (te_raw if te_is_ratio else 0)
        if s <= 0: raise ValueError("ratio sum is 0")
        n_tr = int(np.floor((tr_raw if tr_is_ratio else 0)/s * total_n))
        n_va = int(np.floor((va_raw if va_is_ratio else 0)/s * total_n))
        n_te = total_n - n_tr - n_va
    else:
        n_tr, n_va, n_te = int(tr_raw), int(va_raw), int(te_raw)
    n_tr = max(0, min(n_tr, total_n))
    n_va = max(0, min(n_va, total_n - n_tr))
    n_te = max(0, min(n_te, total_n - n_va - n_tr))
    idx = np.arange(total_n)
    if shuffle:
        assert rng is not None
        rng.shuffle(idx)
    tr = idx[:n_tr]
    va = idx[n_tr:n_tr+n_va]
    te = idx[n_tr+n_va:n_tr+n_va+n_te]
    return tr, va, te


@dataclass
class EvalData:
    X_test: np.ndarray
    X_raw_test: np.ndarray
    y_test: np.ndarray
    labels: List[str]


def load_dataset(modality: str, num_classes: int, length: int,
                 train_spec, val_spec, test_spec,
                 fast_dir: pathlib.Path, slow_dir: pathlib.Path,
                 seed=2025, zscore=True, shuffle=False) -> EvalData:
    assert modality in ("fast", "slow"), "Only single‑modality ('fast' or 'slow') is supported in analysis 4."
    rng = np.random.default_rng(seed)
    labels = [str(i) for i in range(1, num_classes+1)]

    base = fast_dir if modality == "fast" else slow_dir

    Xte, Xrawte, yte = [], [], []
    for ci in range(1, num_classes+1):
        M = ensure_length(read_matrix_any(find_file_for_class(base, ci)), length)  # [L,N]
        N = M.shape[1]
        _, _, te = split_counts(N, train_spec, val_spec, test_spec, shuffle=shuffle, rng=rng)
        if len(te) == 0:
            continue
        arr_raw = M[:, te].T[:, None, :]  # [k,1,L] raw
        arr = arr_raw.copy()
        if zscore:
            m = arr.mean(axis=2, keepdims=True); s = arr.std(axis=2, keepdims=True) + 1e-8
            arr = (arr - m)/s
        Xrawte.append(arr_raw.astype(np.float32))
        Xte.append(arr.astype(np.float32))
        yte += [ci-1] * arr.shape[0]

    Xte = np.concatenate(Xte, axis=0) if Xte else np.zeros((0,1,length), dtype=np.float32)
    Xrawte = np.concatenate(Xrawte, axis=0) if Xrawte else np.zeros((0,1,length), dtype=np.float32)
    return EvalData(X_test=Xte, X_raw_test=Xrawte, y_test=np.array(yte), labels=labels)

# ----------------------------
# (4) Explainability (IG + 1D Grad‑CAM) with shading
# ----------------------------

def moving_average(x: np.ndarray, k: int):
    if k <= 1: return x
    k = int(k)
    if k < 1: return x
    pad = k//2
    xp = np.pad(x, (pad, k-1-pad), mode="edge")
    w = np.ones(k)/k
    return np.convolve(xp, w, mode="valid")


def make_baseline(x: torch.Tensor, mode: str, noise_std: float = 0.1):
    if mode == "zero":
        return torch.zeros_like(x)
    elif mode == "mean":
        m = x.mean(dim=-1, keepdim=True)
        return m.expand_as(x).clone()
    elif mode.startswith("noise"):
        try:
            parts = mode.split("_")
            std = float(parts[1]) if len(parts) > 1 else noise_std
        except Exception:
            std = noise_std
        return std * torch.randn_like(x)
    else:
        return torch.zeros_like(x)


def integrated_gradients(model, x: torch.Tensor, target: int, steps: int=64, baseline_mode: str="zero"):
    model.eval()
    baseline = make_baseline(x, baseline_mode)
    grads = []
    for i in range(steps+1):
        xi = baseline + (i/steps)*(x-baseline)
        xi.requires_grad_(True)
        logits, _ = model(xi)
        loss = logits[0, target]
        model.zero_grad(set_to_none=True)
        loss.backward()
        grads.append(xi.grad.detach().clone())
    avg_grads = torch.stack(grads[:-1]).mean(dim=0)
    ig = (x - baseline) * avg_grads
    return ig[0,0].cpu().numpy()


def grad_cam_1d(model: SingleModalityModel, x: torch.Tensor, target: int):
    feats = {}
    conv_last = model.branch.net[6]  # the last Conv1d before ReLU/Pool
    def fwd_hook(m, i, o): feats['A'] = o.detach().clone()
    def bwd_hook(m, gi, go): feats['dA'] = go[0].detach().clone()
    h1 = conv_last.register_forward_hook(fwd_hook)
    h2 = conv_last.register_full_backward_hook(bwd_hook)
    model.eval()
    x = x.requires_grad_(True)
    logits, _ = model(x)
    loss = logits[0, target]
    model.zero_grad(set_to_none=True); loss.backward()
    A = feats['A'][0]; dA = feats['dA'][0]
    w = dA.mean(dim=1)
    cam = (w[:,None] * A).sum(0)
    cam = torch.relu(cam)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    h1.remove(); h2.remove()
    Lp = cam.shape[0]; L = x.shape[-1]
    cam_up = torch.nn.functional.interpolate(cam[None,None,:], size=L, mode="linear", align_corners=False)[0,0]
    return cam_up.cpu().numpy()


def contiguous_spans(mask: np.ndarray) -> List[Tuple[int,int]]:
    if mask.size == 0: return []
    m = mask.astype(np.int8)
    diff = np.diff(np.pad(m, (1,1), mode='constant'))
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]
    return list(zip(starts, ends))


def plot_explainability(model,
                        X: np.ndarray,
                        y: np.ndarray,
                        labels: List[str],
                        out_dir: pathlib.Path,
                        X_raw: Optional[np.ndarray] = None,
                        n_samples: int = 8,
                        ig_steps: int = 64,
                        ig_baseline: str = "zero",
                        target_mode: str = "true",
                        th_ig: float = 0.4,
                        th_cam: float = 0.6,
                        use_abs_ig: bool = False,
                        smooth_k: int = 0,
                        fixed_indices: Optional[List[int]] = None, highlight: bool = True):
    dev = next(model.parameters()).device
    N = len(y)

    if fixed_indices is not None and len(fixed_indices) > 0:
        sel = np.array([i for i in fixed_indices if 0 <= i < N], dtype=int)
    else:
        sel = np.random.choice(N, size=min(n_samples, N), replace=False)

    for i in sel:
        xi = torch.from_numpy(X[i:i+1]).to(dev)
        with torch.no_grad():
            logits, _ = model(xi)
            pred = int(torch.argmax(logits, dim=1)[0].cpu().item())
            pr = torch.softmax(logits, dim=1)[0].cpu().numpy()
        yi = int(y[i])

        if target_mode == "true":
            tgt = yi
        elif target_mode == "pred":
            tgt = pred
        else:
            order = np.argsort(pr)
            tgt = int(order[-2]) if pr.size > 1 else (pred + 1) % len(labels)

        ig = integrated_gradients(model, xi, tgt, steps=ig_steps, baseline_mode=ig_baseline)
        cam = grad_cam_1d(model, xi, tgt)
        sig = X[i,0]
        sig_raw = X_raw[i,0] if (X_raw is not None) else sig

        ig_norm = ig / (np.abs(ig).max() + 1e-8)
        cam_norm = cam

        if smooth_k and smooth_k > 1:
            ig_norm = moving_average(ig_norm, smooth_k)
            cam_norm = moving_average(cam_norm, smooth_k)

        if highlight:
            if use_abs_ig:
                ig_mask = np.abs(ig_norm) > th_ig
            else:
                ig_mask = ig_norm > th_ig
            cam_mask = cam_norm > th_cam
            joint_mask = ig_mask & cam_mask
            spans = contiguous_spans(joint_mask)
        else:
            joint_mask = np.zeros_like(ig_norm, dtype=bool)
            spans = []

        df = pd.DataFrame({
            "t": np.arange(len(sig)),
            "signal": sig,
            "signal_raw": sig_raw,
            "ig": ig,
            "ig_norm": ig_norm,
            "gradcam": cam_norm,
            "mask_joint": joint_mask.astype(int)
        })
        save_df_csv(out_dir / "csv" / f"explain_{i}.csv", df)

        if highlight:
            span_rows = [{"idx": i, "start": l, "end": r, "length": r-l} for (l,r) in spans]
            save_df_csv(out_dir / "csv" / f"explain_{i}_spans.csv", pd.DataFrame(span_rows))

        cols = class_color_list(labels); col = cols[int(yi) % len(cols)]
        plt.figure(figsize=(9, 3.4), dpi=130)
        ax = plt.gca()

        if highlight:
            for (l, r) in spans:
                ax.axvspan(l, r, alpha=0.25, color=col)

        ax.plot(sig, lw=0.9, color=col, label=f"signal (true={labels[yi]}, tgt={labels[tgt]})")
        ax.plot(ig_norm, lw=0.9, color=col, linestyle="--", alpha=0.95, label=f"IG{'|abs' if use_abs_ig else ''} (norm)")
        ax.plot(cam_norm, lw=0.9, color=col, linestyle=":", alpha=0.95, label="Grad-CAM (0..1)")

        if highlight:
            # threshold guides
            if not use_abs_ig:
                ax.axhline(th_ig, color="k", lw=0.6, ls="--", alpha=0.4)
            else:
                ax.axhline(th_ig,  color="k", lw=0.6, ls="--", alpha=0.4)
                ax.axhline(-th_ig, color="k", lw=0.6, ls="--", alpha=0.4)
            ax.axhline(th_cam, color="k", lw=0.6, ls=":", alpha=0.4)

        ax.set_title(f"Explainability for sample #{i}")
        ax.legend(fontsize=8, ncol=3)
        plt.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / f"explain_{i}.png", dpi=200)
        plt.close()

# ----------------------------
# Main (analysis 4 only)
# ----------------------------

def build_parser():
    import argparse
    p = argparse.ArgumentParser(description="Tactile explainability (analysis 4 only)")
    # data/model
    p.add_argument("--modality", type=str, required=True, choices=["fast","slow"], help="single modality for analysis 4")
    p.add_argument("--num-classes", type=int, default=8)
    p.add_argument("--length", type=int, default=3200)
    p.add_argument("--train", type=float, default=70)
    p.add_argument("--val", type=float, default=25)
    p.add_argument("--test", type=float, default=25)
    p.add_argument("--fast-dir", type=str, default="fast")
    p.add_argument("--slow-dir", type=str, default="slow")
    p.add_argument("--out-dir",  type=str, default="outputs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--ckpt", type=str, default=None, help="path to model weights; default: outputs/best_<modality>.pt")
    # analysis 4 params
    p.add_argument("--exp-n-samples", type=int, default=8, help="number of samples to visualize")
    p.add_argument("--exp-ig-steps", type=int, default=64, help="IG path integral steps")
    p.add_argument("--exp-ig-baseline", type=str, default="zero",
                   choices=["zero","mean","noise","noise_0.1","noise_0.2","noise_0.5"], help="IG baseline")
    p.add_argument("--exp-target", type=str, default="true", choices=["true","pred","second"], help="target class for attribution")
    p.add_argument("--exp-th-ig", type=float, default=0.4, help="threshold on normalized IG (or abs(IG) if --exp-abs-ig)")
    p.add_argument("--exp-th-cam", type=float, default=0.6, help="threshold on normalized Grad-CAM (0..1)")
    p.add_argument("--exp-abs-ig", action="store_true", help="use |IG| > th_ig")
    p.add_argument("--exp-smooth", type=int, default=0, help="optional moving average kernel on IG/CAM (0=off)")
    p.add_argument("--exp-indices", type=str, default="", help="comma-separated test indices to visualize (overrides random sampling)")
    p.add_argument("--exp-no-highlight", action="store_true", help="disable span highlighting & threshold guides; still saves curves and per-sample CSV")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Reproducibility for sampling
    np.random.seed(args.seed)

    out_root = pathlib.Path(args.out_dir)
    out_eval = out_root / "eval"
    out_eval.mkdir(parents=True, exist_ok=True)

    ckpt = pathlib.Path(args.ckpt) if args.ckpt else out_root / f"best_{args.modality}.pt"
    if not ckpt.exists():
        print(f"[ERROR] 未找到权重文件: {ckpt}\n请提供 --ckpt 或在 {out_root} 下放置 best_{args.modality}.pt")
        print("已跳过分析 4。")
        print("Done. 输出目录：", out_eval)
        return

    # load test
    data = load_dataset(args.modality, args.num_classes, args.length,
                        args.train, args.val, args.test,
                        pathlib.Path(args.fast_dir), pathlib.Path(args.slow_dir),
                        seed=args.seed, zscore=True, shuffle=args.shuffle)

    if data.X_test.shape[0] == 0:
        print("[WARN] 测试集为空，无法执行解释性可视化。")
        print("Done. 输出目录：", out_eval)
        return

    # build model & load weights
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SingleModalityModel(args.length, args.num_classes, hidden_dim=args.hidden_dim).to(dev)
    sd = safe_load_weights(ckpt, dev)
    try:
        model.load_state_dict(sd)
    except RuntimeError:
        from collections import OrderedDict
        new_sd = OrderedDict()
        for k, v in sd.items():
            nk = k.replace("branch.conv", "branch.net").replace("branch.head", "branch.fc")
            new_sd[nk] = v
        model.load_state_dict(new_sd, strict=False)
    disable_inplace_relu(model)
    model.eval()

    fixed_indices = None
    if args.exp_indices.strip():
        try:
            fixed_indices = [int(s) for s in args.exp_indices.split(",") if s.strip()!=""]
        except Exception:
            fixed_indices = None

    labels = [f"H{i}" for i in range(1, args.num_classes + 1)]

    plot_explainability(
        model=model,
        X=data.X_test,
        y=data.y_test,
        labels=labels,
        out_dir=out_eval,
        X_raw=data.X_raw_test,
        n_samples=args.exp_n_samples,
        ig_steps=args.exp_ig_steps,
        ig_baseline=args.exp_ig_baseline,
        target_mode=args.exp_target,
        th_ig=args.exp_th_ig,
        th_cam=args.exp_th_cam,
        use_abs_ig=args.exp_abs_ig,
        smooth_k=args.exp_smooth,
        fixed_indices=fixed_indices,
        highlight=(not args.exp_no_highlight)
    )
    print("[4] saved: explain_*.png")
    print("Done. 输出目录：", out_eval)


if __name__ == "__main__":
    main()
