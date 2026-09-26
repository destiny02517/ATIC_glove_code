#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tactile_eval_tsne.py —— 仅用 t-SNE 可视化（透明底；可调尺寸/纵横比/字体）
输出：
  - PNG: outputs/eval/embed_tsne_ellipses.png
  - CSV: outputs/eval/csv/embed_tsne.csv / embed_tsne_ellipses.csv
"""

import os, pathlib, warnings
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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
# 颜色
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
# 模型（与训练一致）
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

class FusionModel(nn.Module):
    def __init__(self, length: int, num_classes: int, hidden_dim: int = 512):
        super().__init__()
        self.f = BranchCNN(length, hidden_dim=hidden_dim)
        self.s = BranchCNN(length, hidden_dim=hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim*2, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, xf, xs):
        pf, ps = self.f(xf), self.s(xs)
        cat = torch.cat([pf, ps], dim=1)
        out = self.head(cat)
        return out, cat

def safe_load_weights(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)

# ----------------------------
# 数据加载
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
    for ext in (".xls",".xlsx",".csv"):
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
    n_te = max(0, min(n_te, total_n - n_tr - n_va))
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
    X_test: Optional[np.ndarray]
    y_test: np.ndarray
    labels: List[str]
    Xf_test: Optional[np.ndarray] = None
    Xs_test: Optional[np.ndarray] = None

def load_dataset(modality: str, num_classes: int, length: int,
                 train_spec, val_spec, test_spec,
                 fast_dir: pathlib.Path, slow_dir: pathlib.Path,
                 seed=2025, zscore=True, shuffle=False) -> EvalData:
    rng = np.random.default_rng(seed)
    labels = [str(i) for i in range(1, num_classes+1)]

    def per_mod(base: pathlib.Path):
        Xte, yte = [], []
        for ci in range(1, num_classes+1):
            M = ensure_length(read_matrix_any(find_file_for_class(base, ci)), length)
            N = M.shape[1]
            _, _, te = split_counts(N, train_spec, val_spec, test_spec, shuffle=shuffle, rng=rng)
            if len(te) == 0: continue
            arr = M[:, te].T[:, None, :]
            if zscore:
                m = arr.mean(axis=2, keepdims=True); s = arr.std(axis=2, keepdims=True) + 1e-8
                arr = (arr - m)/s
            Xte.append(arr.astype(np.float32))
            yte += [ci-1] * arr.shape[0]
        Xte = np.concatenate(Xte, axis=0) if Xte else np.zeros((0,1,length), dtype=np.float32)
        return Xte, np.array(yte)

    if modality in ("fast", "slow"):
        base = fast_dir if modality == "fast" else slow_dir
        Xte, yte = per_mod(base)
        return EvalData(X_test=Xte, y_test=yte, labels=labels)
    else:
        state = rng.bit_generator.state
        Xf_te, yf = per_mod(fast_dir)
        rng.bit_generator.state = state
        Xs_te, ys = per_mod(slow_dir)
        assert np.array_equal(yf, ys), "FAST/SLOW splits misaligned"
        return EvalData(X_test=None, y_test=yf, labels=labels, Xf_test=Xf_te, Xs_test=Xs_te)

# ----------------------------
# 特征抽取
# ----------------------------
@torch.no_grad()
def extract_features(modality: str, length: int, num_classes: int, hidden_dim: int,
                     ckpt_path: pathlib.Path, data: EvalData, device: str = "cuda",
                     feature_space: str = "penultimate") -> np.ndarray:
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    if modality in ("fast","slow"):
        model = SingleModalityModel(length, num_classes, hidden_dim=hidden_dim).to(dev)
        sd = safe_load_weights(ckpt_path, dev)
        try:
            model.load_state_dict(sd)
        except RuntimeError:
            from collections import OrderedDict
            new_sd = OrderedDict()
            for k,v in sd.items():
                nk = k.replace("branch.conv","branch.net").replace("branch.head","branch.fc")
                new_sd[nk] = v
            model.load_state_dict(new_sd, strict=False)
        model.eval()
        X = torch.from_numpy(data.X_test).to(dev)
        logits, pen = model(X)
        if feature_space == "penultimate":
            feats = pen.cpu().numpy()
        elif feature_space == "logits":
            feats = logits.cpu().numpy()
        else:
            feats = data.X_test.reshape(data.X_test.shape[0], -1)
        return feats

    model = FusionModel(length, num_classes, hidden_dim=hidden_dim).to(dev)
    sd = safe_load_weights(ckpt_path, dev)
    try:
        model.load_state_dict(sd)
    except RuntimeError:
        from collections import OrderedDict
        new_sd = OrderedDict()
        for k,v in sd.items():
            nk = (k.replace("f.conv","f.net").replace("f.head","f.fc")
                    .replace("s.conv","s.net").replace("s.head","s.fc"))
            new_sd[nk] = v
        model.load_state_dict(new_sd, strict=False)
    model.eval()
    Xf = torch.from_numpy(data.Xf_test).to(dev)
    Xs = torch.from_numpy(data.Xs_test).to(dev)
    logits, cat = model(Xf, Xs)
    if feature_space == "penultimate":
        feats = cat.cpu().numpy()
    elif feature_space == "logits":
        feats = logits.cpu().numpy()
    else:
        feats = np.concatenate([data.Xf_test.reshape(Xf.shape[0], -1),
                                data.Xs_test.reshape(Xs.shape[0], -1)], axis=1)
    return feats

# ----------------------------
# t-SNE（版本兼容）+ 绘图（透明底 + 字体可调）
# ----------------------------
def _build_tsne(args, perp, lr, metric):
    kw = dict(
        n_components=2,
        perplexity=perp,
        learning_rate=lr,
        init=args.tsne_init,
        random_state=args.seed,
        early_exaggeration=args.tsne_ee,
        metric=metric,
        angle=args.tsne_angle,
    )
    try:
        return TSNE(n_iter=args.tsne_niter, **kw)
    except TypeError:
        print("[WARN] TSNE() 在你当前的 sklearn 版本不接受 n_iter，已使用默认迭代数。")
        return TSNE(**kw)

def compute_tsne(X: np.ndarray, y: np.ndarray, labels: List[str], args):
    Xs = StandardScaler().fit_transform(X)
    p = min(args.pca_pre, Xs.shape[1])
    Xp = PCA(n_components=p, whiten=args.pca_whiten, random_state=args.seed).fit_transform(Xs)

    n = Xp.shape[0]
    max_safe = max(2, n//3 - 1)
    perp = min(args.tsne_perp, max_safe)

    lr = args.tsne_lr
    try:
        lr = float(lr) if str(lr).lower() != "auto" else "auto"
    except Exception:
        lr = "auto"

    metric = args.tsne_metric
    tsne = _build_tsne(args, perp, lr, metric)
    try:
        Z = tsne.fit_transform(Xp)
    except ValueError as e:
        if "metric" in str(e).lower():
            print(f"[WARN] metric='{metric}' 在当前版本不受支持，已回退为 'euclidean'。")
            tsne = _build_tsne(args, perp, lr, "euclidean")
            Z = tsne.fit_transform(Xp)
        else:
            raise
    return Z, "TSNE"

def plot_tsne_with_ellipses(Z: np.ndarray, y: np.ndarray, labels: List[str], args, out_png: pathlib.Path):
    fig = plt.figure(figsize=(args.embed_w, args.embed_h), dpi=args.embed_dpi)
    ax = plt.gca()
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    xmin, xmax = np.percentile(Z[:, 0], [1, 99])
    ymin, ymax = np.percentile(Z[:, 1], [1, 99])
    rx, ry = (xmax - xmin), (ymax - ymin)
    xmin -= 0.05 * rx; xmax += 0.05 * rx
    ymin -= 0.05 * ry; ymax += 0.05 * ry

    cols = class_color_list(labels)
    classes = np.unique(y)

    for c in classes:
        m = (y == c)
        ax.scatter(Z[m, 0], Z[m, 1], s=args.pt_size, alpha=args.pt_alpha,
                   label=labels[c], color=cols[int(c) % len(cols)], edgecolors="none")

    k50, k90 = 1.3863, 4.6052

    def add_ellipse(mu, cov, k, edgecolor, facecolor, alpha, lw=1.6):
        vals, vecs = np.linalg.eigh(cov + 1e-8*np.eye(2))
        vals = np.clip(vals, 1e-8, None)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        width, height = 2.0 * np.sqrt(vals * k)
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        ell = Ellipse(xy=mu, width=width, height=height, angle=angle,
                      edgecolor=edgecolor, facecolor=facecolor, lw=lw, alpha=alpha)
        ax.add_patch(ell)

    rows = []
    for c in classes:
        m = (y == c)
        if m.sum() < 3:
            continue
        Zi = Z[m]
        mu = Zi.mean(0)
        cov = np.cov(Zi.T)
        col = cols[int(c) % len(cols)]
        if args.ell50_alpha > 0:
            add_ellipse(mu, cov, k50, edgecolor=col, facecolor=col, alpha=args.ell50_alpha)
        if args.ell90_alpha > 0:
            add_ellipse(mu, cov, k90, edgecolor=col, facecolor=col, alpha=args.ell90_alpha)
        ax.scatter([mu[0]], [mu[1]], marker='x', s=60, color=col, linewidths=1.8)
        vals, vecs = np.linalg.eigh(cov + 1e-8*np.eye(2))
        vals = np.clip(vals, 1e-8, None); order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))
        w50, h50 = 2.0*np.sqrt(vals*k50); w90, h90 = 2.0*np.sqrt(vals*k90)
        rows.append({
            "class_id": int(c), "label": labels[int(c)],
            "mu_x": mu[0], "mu_y": mu[1],
            "cov_00": cov[0,0], "cov_01": cov[0,1], "cov_11": cov[1,1],
            "angle_deg": angle,
            "w50": w50, "h50": h50, "w90": w90, "h90": h90
        })

    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal" if args.embed_aspect=="equal" else "auto", adjustable="box")

    if args.legend_size is not None:
        leg = ax.legend(title="class", loc="best", frameon=True, fontsize=args.legend_size, title_fontsize=args.legend_size)
    else:
        leg = ax.legend(title="class", loc="best", frameon=True)
    if leg: leg.get_frame().set_alpha(0.6)

    if args.label_size is not None:
        ax.set_xlabel("dim 1", fontsize=args.label_size)
        ax.set_ylabel("dim 2", fontsize=args.label_size)
    else:
        ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")

    if args.tick_size is not None:
        ax.tick_params(labelsize=args.tick_size)

    if args.title_size is not None:
        ax.set_title("TSNE + class ellipses (50% / 90%)", fontsize=args.title_size)
    else:
        ax.set_title("TSNE + class ellipses (50% / 90%)")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", transparent=True)
    plt.close()

    dfZ = pd.DataFrame({"dim1": Z[:,0], "dim2": Z[:,1], "y": y.astype(int),
                        "label": [labels[i] for i in y.astype(int)]})
    save_df_csv(out_png.parent / "csv" / "embed_tsne.csv", dfZ)
    save_df_csv(out_png.parent / "csv" / "embed_tsne_ellipses.csv", pd.DataFrame(rows))

# ----------------------------
# 字体设置（新增功能）
# ----------------------------
def apply_font_from_args(args):
    # 优先使用 --font-path（可指向 .ttf/.otf），其次使用 --font-family
    if args.font_path:
        try:
            from matplotlib import font_manager as fm
            fm.fontManager.addfont(args.font_path)
            prop = fm.FontProperties(fname=args.font_path)
            family = prop.get_name()
            mpl.rcParams["font.family"] = family
            print(f"[font] loaded from file: {family}")
        except Exception as e:
            print("[WARN] load font from path failed:", e)
    elif args.font_family:
        mpl.rcParams["font.family"] = args.font_family
        print(f"[font] family = {args.font_family}")

    if args.font_size is not None:
        mpl.rcParams["font.size"] = args.font_size
    if args.label_size is not None:
        mpl.rcParams["axes.labelsize"] = args.label_size
    if args.title_size is not None:
        mpl.rcParams["axes.titlesize"] = args.title_size
    if args.tick_size is not None:
        mpl.rcParams["xtick.labelsize"] = args.tick_size
        mpl.rcParams["ytick.labelsize"] = args.tick_size
    if args.legend_size is not None:
        mpl.rcParams["legend.fontsize"] = args.legend_size

# ----------------------------
# 主流程
# ----------------------------
def build_parser():
    import argparse
    p = argparse.ArgumentParser(description="t-SNE evaluation & visualization (transparent background)")
    # 数据/模型
    p.add_argument("--modality", type=str, required=True, choices=["fast","slow","fusion"])
    p.add_argument("--num-classes", type=int, default=8)
    p.add_argument("--length", type=int, default=3200)
    p.add_argument("--train", type=float, default=70)
    p.add_argument("--val", type=float, default=25)
    p.add_argument("--test", type=float, default=25)
    p.add_argument("--fast-dir", type=str, default="fast")
    p.add_argument("--slow-dir", type=str, default="slow")
    p.add_argument("--out-dir",  type=str, default="outputs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shuffle", action="store_true", help="若训练切分是随机的，加此项并设定相同 seed")
    p.add_argument("--hidden-dim", type=int, default=512, help="倒数第二层维度（与训练一致）")
    p.add_argument("--ckpt", type=str, default=None, help="权重路径（若提供，将用模型特征）")
    p.add_argument("--feature-space", type=str, default="penultimate", choices=["penultimate","logits","raw"])
    # 预处理
    p.add_argument("--pca-pre", type=int, default=50, help="t-SNE 前的 PCA 维度")
    p.add_argument("--pca-whiten", action="store_true", help="PCA whiten")
    # t-SNE 参数
    p.add_argument("--tsne-perp", type=float, default=30.0)
    p.add_argument("--tsne-lr", type=str, default="auto")  # 或数字
    p.add_argument("--tsne-ee", type=float, default=12.0)
    p.add_argument("--tsne-niter", type=int, default=2500)
    p.add_argument("--tsne-metric", type=str, default="euclidean",
                   choices=["euclidean","cosine","manhattan","chebyshev"])
    p.add_argument("--tsne-init", type=str, default="pca", choices=["pca","random"])
    p.add_argument("--tsne-angle", type=float, default=0.5)
    # 绘图外观
    p.add_argument("--embed-w", type=float, default=7.0)
    p.add_argument("--embed-h", type=float, default=7.0)
    p.add_argument("--embed-dpi", type=int, default=120)
    p.add_argument("--embed-aspect", type=str, default="equal", choices=["equal","auto"])
    p.add_argument("--pt-size", type=float, default=18.0)
    p.add_argument("--pt-alpha", type=float, default=0.85)
    p.add_argument("--ell50-alpha", type=float, default=0.50)
    p.add_argument("--ell90-alpha", type=float, default=0.10)
    # 字体控制（新增）
    p.add_argument("--font-family", type=str, default=None, help="例：'Microsoft YaHei', 'SimHei', 'DejaVu Sans'")
    p.add_argument("--font-path", type=str, default=None, help="本地字体文件 .ttf/.otf 路径")
    p.add_argument("--font-size", type=float, default=None, help="全局基础字号（rcParams['font.size']）")
    p.add_argument("--title-size", type=float, default=None, help="标题字号")
    p.add_argument("--label-size", type=float, default=None, help="坐标轴标签字号")
    p.add_argument("--tick-size", type=float, default=None, help="坐标刻度字号")
    p.add_argument("--legend-size", type=float, default=None, help="图例字号（含标题）")
    return p

def main():
    parser = build_parser()
    args = parser.parse_args()

    # 应用字体设置（只影响图形外观）
    apply_font_from_args(args)

    out_root = pathlib.Path(args.out_dir)
    out_eval = out_root / "eval"
    out_eval.mkdir(parents=True, exist_ok=True)

    data = load_dataset(args.modality, args.num_classes, args.length,
                        args.train, args.val, args.test,
                        pathlib.Path(args.fast_dir), pathlib.Path(args.slow_dir),
                        seed=args.seed, zscore=True, shuffle=args.shuffle)
    y_true = data.y_test
    labels = [f"R{i}" for i in range(1, args.num_classes + 1)]

    feats = None
    if args.ckpt:
        ckpt = pathlib.Path(args.ckpt)
        if not ckpt.exists():
            print(f"[WARN] 未找到 {ckpt}，将改用原始信号展平特征。")
        else:
            feats = extract_features(args.modality, args.length, args.num_classes,
                                     args.hidden_dim, ckpt, data, feature_space=args.feature_space)
    if feats is None:
        if args.modality in ("fast","slow") and data.X_test is not None:
            feats = data.X_test.reshape(data.X_test.shape[0], -1)
        else:
            feats = np.concatenate([data.Xf_test.reshape(data.Xf_test.shape[0], -1),
                                    data.Xs_test.reshape(data.Xs_test.shape[0], -1)], axis=1)

    Z, _ = compute_tsne(feats, y_true, labels, args)
    out_png = out_eval / "embed_tsne_ellipses.png"
    plot_tsne_with_ellipses(Z, y_true, labels, args, out_png)
    print("Saved:", out_png)
    print("CSV:", out_eval / "csv" / "embed_tsne.csv")
    print("CSV:", out_eval / "csv" / "embed_tsne_ellipses.csv")
    print("Done. 输出目录：", out_eval)

if __name__ == "__main__":
    main()
