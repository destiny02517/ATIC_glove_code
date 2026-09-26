#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, pathlib, numpy as np, pandas as pd
import torch, torch.nn as nn, matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# ---------------- 通用：数据加载 ----------------
def read_matrix_any(p: pathlib.Path):
    suf=p.suffix.lower()
    if suf in [".xls",".xlsx"]: return np.asarray(pd.read_excel(p, sheet_name="Sheet1"))
    elif suf==".csv": return np.asarray(pd.read_csv(p, header=None))
    raise ValueError("bad file")

def ensure_length(A,L):
    l,n=A.shape
    if l==L: return A
    if l>L:
        s=(l-L)//2; return A[s:s+L,:]
    pad=L-l; top=pad//2; bot=pad-top
    return np.pad(A, ((top,bot),(0,0)), "constant")

def find_file_for_class(base,ci):
    for ext in (".xls",".xlsx",".csv"):
        p=base/f"{ci}{ext}"
        if p.exists(): return p
    raise FileNotFoundError

def split_counts(N,tr,va,te):
    def dec(x): return (x,True) if x<=1 else (int(x),False)
    t,tis=dec(tr); v,vis=dec(va); e,eis=dec(te)
    if tis or vis or eis:
        r=(t if tis else 0)+(v if vis else 0)+(e if eis else 0)
        nt=int(np.floor((t if tis else 0)/r*N))
        nv=int(np.floor((v if vis else 0)/r*N))
        ne=N-nt-nv
    else:
        nt,nv,ne=int(t),int(v),int(e)
    nt=max(0,min(nt,N)); nv=max(0,min(nv,N-nt)); ne=max(0,min(ne,N-nt-nv))
    return nt,nv,ne

def per_modality_dataset(d,C,L,tr,va,te,rng,shuffle=False):
    Xtr,Xva,Xte=[],[],[]; ytr,yva,yte=[],[],[]
    d=pathlib.Path(d)
    for ci in range(1,C+1):
        M=ensure_length(read_matrix_any(find_file_for_class(d,ci)),L)
        N=M.shape[1]; idx=np.arange(N)
        if shuffle: rng.shuffle(idx)
        ntr,nva,nte=split_counts(N,tr,va,te)
        sel=[idx[:ntr], idx[ntr:ntr+nva], idx[ntr+nva:ntr+nva+nte]]
        def g(s):
            if len(s)==0: return np.zeros((0,1,L),np.float32)
            A=M[:,s].T[:,None,:]
            m=A.mean(2,keepdims=True); st=A.std(2,keepdims=True)+1e-8
            return ((A-m)/st).astype(np.float32)
        Xtr.append(g(sel[0])); ytr+=[ci-1]*len(sel[0])
        Xva.append(g(sel[1])); yva+=[ci-1]*len(sel[1])
        Xte.append(g(sel[2])); yte+=[ci-1]*len(sel[2])
    cat=lambda Ls: np.concatenate(Ls,0) if Ls else np.zeros((0,1,L),np.float32)
    return (cat(Xtr),np.array(ytr),cat(Xva),np.array(yva),cat(Xte),np.array(yte))

def build_fusion_test(fast,slow,C,L,tr,va,te,seed=2025):
    rng=np.random.default_rng(seed); state=rng.bit_generator.state
    Xf_tr,yf_tr,Xf_va,yf_va,Xf_te,yf_te = per_modality_dataset(fast,C,L,tr,va,te,rng,False)
    rng.bit_generator.state=state
    Xs_tr,ys_tr,Xs_va,ys_va,Xs_te,ys_te = per_modality_dataset(slow,C,L,tr,va,te,rng,False)
    assert np.array_equal(yf_te,ys_te)
    return torch.from_numpy(Xf_te), torch.from_numpy(Xs_te), torch.from_numpy(yf_te).long()

# ---------------- 模型（与训练一致命名） ----------------
class BranchCNN(nn.Module):
    def __init__(self, length:int):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv1d(1,32,5,padding=2), nn.ReLU(True), nn.MaxPool1d(2,2),
            nn.Conv1d(32,64,5,padding=2), nn.ReLU(True), nn.MaxPool1d(2,2),
            nn.Conv1d(64,128,5,padding=2), nn.ReLU(True), nn.MaxPool1d(2,2),
            nn.Dropout(0.5),
        )
        with torch.no_grad():
            self.flat_dim=self.net(torch.zeros(1,1,length)).numel()
        self.fc=nn.Sequential(nn.Flatten(), nn.Linear(self.flat_dim,512), nn.ReLU(True))
    def forward(self,x,collect=False):
        feats=[]; cur=x
        for layer in self.net:
            cur=layer(cur)
            if isinstance(layer, nn.MaxPool1d): feats.append(cur)
        feats.append(cur)  # afterDrop
        emb=self.fc(cur)   # [B,512]
        if collect: feats.append(emb); return emb,feats
        return emb

class FusionModel(nn.Module):
    def __init__(self,length:int,C:int):
        super().__init__()
        self.f_branch=BranchCNN(length); self.s_branch=BranchCNN(length)
        self.head=nn.Sequential(nn.Linear(1024,256), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(256,C))
    def forward(self,xf,xs,collect=False):
        hf,Ff=self.f_branch(xf,collect=True); hs,Fs=self.s_branch(xs,collect=True)
        h=self.head[1](self.head[0](torch.cat([hf,hs],1)))  # 256 after ReLU
        logits=self.head[3](self.head[2](h))
        if collect: return logits, {"fast":Ff, "slow":Fs, "fusion":h}
        return logits

# ---------------- CKA ----------------
def center(X): return X - X.mean(0, keepdim=True)
def linear_cka(X, Y):
    Xc, Yc = center(X), center(Y)
    K = Xc @ Xc.t(); L = Yc @ Yc.t()
    hsic = (K * L).mean()
    var1 = (K * K).mean().sqrt(); var2 = (L * L).mean().sqrt()
    if var1.item()==0 or var2.item()==0: return torch.tensor(0.0, device=X.device)
    return hsic / (var1 * var2)

def safe_load(path, device):
    try:    return torch.load(path, map_location=device, weights_only=True)  # torch>=2.4
    except TypeError:
        return torch.load(path, map_location=device)

# ---------------- 主程序 ----------------
def main():
    ap=argparse.ArgumentParser()
    # 数据&模型
    ap.add_argument("--fast-dir", default="fast"); ap.add_argument("--slow-dir", default="slow"); ap.add_argument("--outputs-dir", default="outputs")
    ap.add_argument("--num-classes", type=int, required=True); ap.add_argument("--length", type=int, default=3200)
    ap.add_argument("--dt", type=float, default=2e-4)
    ap.add_argument("--train", type=float, default=70); ap.add_argument("--val", type=float, default=25); ap.add_argument("--test", type=float, default=25)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--batch-size", type=int, default=64); ap.add_argument("--max-samples", type=int, default=1024)
    ap.add_argument("--bootstrap", type=int, default=40); ap.add_argument("--boot-frac", type=float, default=0.7)

    # —— 外观控制（小提琴图）——
    ap.add_argument("--fig-w", type=float, default= max(8.0, 20*0.35), help="violin figure width (inch)")
    ap.add_argument("--fig-h", type=float, default= 4.2, help="violin figure height (inch)")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--font-family", default="", help="e.g. SimHei, Arial, DejaVu Sans. Leave empty to keep default.")
    ap.add_argument("--title-size", type=float, default=12.5)
    ap.add_argument("--label-size", type=float, default=10.5)
    ap.add_argument("--tick-size", type=float, default=9.0)
    ap.add_argument("--x-rotation", type=float, default=45.0)

    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--grid-alpha", type=float, default=0.25)
    ap.add_argument("--ylims", type=str, default="", help="e.g. 0,1 to clamp y axis")

    ap.add_argument("--violin-face", default="#A0CED9")
    ap.add_argument("--violin-edge", default="#2c3e50")
    ap.add_argument("--violin-alpha", type=float, default=0.7)
    ap.add_argument("--violin-lw", type=float, default=1.2)
    ap.add_argument("--median-color", default="#3A6EA5")
    ap.add_argument("--median-lw", type=float, default=2.0)
    ap.add_argument("--line-color", default="#2c3e50", help="cbars/cmins/cmaxes color")
    ap.add_argument("--line-lw", type=float, default=1.2)

    ap.add_argument("--title", default="", help="custom title; leave empty to use default")

    # —— 外观控制（全局条形图）——
    ap.add_argument("--bar-fig-w", type=float, default=4.0)
    ap.add_argument("--bar-fig-h", type=float, default=3.2)
    ap.add_argument("--bar-face", default="#3A6EA5")
    ap.add_argument("--bar-edge", default="#1f2d3a")
    ap.add_argument("--bar-alpha", type=float, default=0.9)

    # 输出
    ap.add_argument("--save-violin", default="fusion_preference_violin.png")
    ap.add_argument("--save-bar", default="fusion_preference_global.png")
    ap.add_argument("--csv-out", default="", help="optional: write per-class bootstrap R to CSV")

    args=ap.parse_args()

    # 字体
    # 字体设置（如果传了 --font-family 就应用）
    if getattr(args, "font_family", ""):
        plt.rcParams["font.family"] = args.font_family
        # 进一步确保用到无衬线字体时也优先选择这个
        plt.rcParams["font.sans-serif"] = [args.font_family]

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据
    Xf_te, Xs_te, yte = build_fusion_test(args.fast_dir,args.slow_dir,args.num_classes,args.length,
                                          args.train,args.val,args.test,args.seed)
    N = min(args.max_samples, Xf_te.size(0))
    dl = DataLoader(TensorDataset(Xf_te[:N], Xs_te[:N], yte[:N]),
                    batch_size=args.batch_size, shuffle=False)

    # 模型
    model=FusionModel(args.length,args.num_classes).to(device)
    model.load_state_dict(safe_load(os.path.join(args.outputs_dir,"best_fusion.pt"), device))
    model.eval()

    # 收集嵌入
    Zf=[]; Zs=[]; Zu=[]; Y=[]
    with torch.no_grad():
        for xf,xs,y in dl:
            xf=xf.to(device); xs=xs.to(device)
            _, packs = model(xf,xs,collect=True)
            Zf.append(packs["fast"][-1].cpu())    # [B,512]
            Zs.append(packs["slow"][-1].cpu())    # [B,512]
            Zu.append(packs["fusion"].cpu())      # [B,256]
            Y.append(y)
    Zf=torch.cat(Zf,0); Zs=torch.cat(Zs,0); Zu=torch.cat(Zu,0); Y=torch.cat(Y,0)

    def cka(a,b): return linear_cka(a.to(device), b.to(device)).item()

    R_all = cka(Zu,Zf) / (cka(Zu,Zf) + cka(Zu,Zs))

    # 按类自举
    rng=np.random.default_rng(0)
    data=[]; medians=[]; labels = [f"O{i}" for i in range(1, args.num_classes + 1)]
    for k in range(args.num_classes):
        idx = torch.nonzero(Y==k).squeeze(1).cpu().numpy()
        if idx.size==0: data.append([np.nan]); medians.append(np.nan); continue
        B = min(args.bootstrap, max(1, idx.size))
        m = max(2, int(np.ceil(args.boot_frac*idx.size)))
        vals=[]
        for _ in range(B):
            sel = rng.choice(idx, size=m, replace=True)
            r = cka(Zu[sel],Zf[sel]) / (cka(Zu[sel],Zf[sel]) + cka(Zu[sel],Zs[sel]))
            vals.append(r)
        data.append(vals)
        medians.append(float(np.median(vals)))

    # 可选导出 CSV
    if args.csv_out:
        df = pd.DataFrame({"class": labels, "median_R": medians})
        for i,vals in enumerate(data): df[f"boot_{i}"] = pd.Series(vals)
        df.to_csv(os.path.join(args.outputs_dir, args.csv_out), index=False, encoding="utf-8-sig")

    # ---------- 小提琴图 ----------
    fig = plt.figure(figsize=(args.fig_w, args.fig_h), dpi=args.dpi)
    parts = plt.violinplot(data, showmedians=True)

    # 美化：填充体
    for b in parts["bodies"]:
        b.set_facecolor(args.violin_face)
        b.set_edgecolor(args.violin_edge)
        b.set_alpha(args.violin_alpha)
        b.set_linewidth(args.violin_lw)

    # 线条（cbars/cmins/cmaxes）
    for k in ("cbars","cmins","cmaxes"):
        if k in parts:
            parts[k].set_color(args.line_color)
            parts[k].set_linewidth(args.line_lw)

    # 中位线
    if "cmedians" in parts:
        parts["cmedians"].set_color(args.median_color)
        parts["cmedians"].set_linewidth(args.median_lw)

    plt.xticks(np.arange(1,args.num_classes+1), labels, rotation=args.x_rotation, ha="right", fontsize=args.tick_size)
    plt.yticks(fontsize=args.tick_size)

    if args.ylims:
        try:
            y0,y1 = [float(x) for x in args.ylims.split(",")]
            plt.ylim(y0,y1)
        except:
            pass

    ylabel = "Fusion preference R = CKA(Fu,Fast)/(CKA(Fu,Fast)+CKA(Fu,Slow))"
    plt.ylabel(ylabel, fontsize=args.label_size)

    ttl = args.title if args.title else f"Per-class fusion preference (global R={R_all:.3f})"
    plt.title(ttl, fontsize=args.title_size)

    if args.grid:
        plt.grid(True, axis="y", alpha=args.grid_alpha, linestyle="--", linewidth=0.8)

    out1 = os.path.join(args.outputs_dir, args.save_violin)
    plt.tight_layout(); plt.savefig(out1, dpi=args.dpi); plt.close(fig)

    # ---------- 全局条形 ----------
    fig2 = plt.figure(figsize=(args.bar_fig_w, args.bar_fig_h), dpi=args.dpi)
    plt.bar([0], [R_all], width=0.6, color=args.bar_face, edgecolor=args.bar_edge, alpha=args.bar_alpha)
    plt.ylim(0,1); plt.xticks([0], ["All"], fontsize=args.tick_size)
    plt.ylabel("R", fontsize=args.label_size)
    plt.title("Global fusion preference", fontsize=args.title_size)
    if args.grid:
        plt.grid(True, axis="y", alpha=args.grid_alpha, linestyle="--", linewidth=0.8)
    out2 = os.path.join(args.outputs_dir, args.save_bar)
    plt.tight_layout(); plt.savefig(out2, dpi=args.dpi); plt.close(fig2)

    print("Saved:", out1, out2)
    if args.csv_out:
        print("CSV  :", os.path.join(args.outputs_dir, args.csv_out))

if __name__ == "__main__":
    main()
