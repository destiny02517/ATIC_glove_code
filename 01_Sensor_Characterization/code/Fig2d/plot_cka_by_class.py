#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, pathlib, numpy as np, pandas as pd
import torch, torch.nn as nn, matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from matplotlib.colors import LinearSegmentedColormap

def read_matrix_any(p):
    suf=p.suffix.lower()
    if suf in [".xls",".xlsx"]: return np.asarray(pd.read_excel(p, sheet_name="Sheet1"))
    elif suf==".csv": return np.asarray(pd.read_csv(p, header=None))
    raise ValueError("bad file")

def ensure_length(A,L):
    l,n=A.shape
    if l==L: return A
    if l>L: s=(l-L)//2; return A[s:s+L,:]
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
        nt=int(np.floor((t if tis else 0)/r*N)); nv=int(np.floor((v if vis else 0)/r*N)); ne=N-nt-nv
    else: nt,nv,ne=int(t),int(v),int(e)
    nt=max(0,min(nt,N)); nv=max(0,min(nv,N-nt)); ne=max(0,min(ne,N-nt-nv)); return nt,nv,ne

def per_modality_dataset(d,C,L,tr,va,te,rng,shuffle=False):
    Xtr,Xva,Xte=[],[],[]; ytr,yva,yte=[],[],[]
    d=pathlib.Path(d)
    for ci in range(1,C+1):
        M=ensure_length(read_matrix_any(find_file_for_class(d,ci)),L); N=M.shape[1]; idx=np.arange(N)
        if shuffle: rng.shuffle(idx)
        ntr,nva,nte=split_counts(N,tr,va,te)
        sel=[idx[:ntr], idx[ntr:ntr+nva], idx[ntr+nva:ntr+nva+nte]]
        def g(s):
            if len(s)==0: return np.zeros((0,1,L),np.float32)
            A=M[:,s].T[:,None,:]; m=A.mean(2,keepdims=True); st=A.std(2,keepdims=True)+1e-8
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

class BranchCNN(nn.Module):
    def __init__(self, length:int):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv1d(1,32,5,padding=2), nn.ReLU(True), nn.MaxPool1d(2,2),
            nn.Conv1d(32,64,5,padding=2), nn.ReLU(True), nn.MaxPool1d(2,2),
            nn.Conv1d(64,128,5,padding=2), nn.ReLU(True), nn.MaxPool1d(2,2),
            nn.Dropout(0.5),
        )
        with torch.no_grad(): self.flat_dim=self.net(torch.zeros(1,1,length)).numel()
        self.fc=nn.Sequential(nn.Flatten(), nn.Linear(self.flat_dim,512), nn.ReLU(True))
    def forward(self,x,collect=False):
        feats=[]; cur=x
        for layer in self.net:
            cur=layer(cur)
            if isinstance(layer, nn.MaxPool1d): feats.append(cur)
        feats.append(cur)  # afterDrop
        emb=self.fc(cur)
        if collect:
            feats.append(emb); return emb,feats
        return emb

class FusionModel(nn.Module):
    def __init__(self,length:int,C:int):
        super().__init__()
        self.f_branch=BranchCNN(length); self.s_branch=BranchCNN(length)
        self.head=nn.Sequential(nn.Linear(1024,256), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(256,C))
    def forward(self,xf,xs,collect=False):
        hf,Ff=self.f_branch(xf,collect=True); hs,Fs=self.s_branch(xs,collect=True)
        h=self.head[1](self.head[0](torch.cat([hf,hs],1)))
        logits=self.head[3](self.head[2](h))
        if collect: return logits, {"fast":Ff, "slow":Fs, "fusion":h}
        return logits

def center(X): return X - X.mean(0, keepdim=True)
def linear_cka(X, Y):
    Xc, Yc = center(X), center(Y)
    K = Xc @ Xc.t(); L = Yc @ Yc.t()
    hsic = (K * L).mean()
    var1 = (K * K).mean().sqrt(); var2 = (L * L).mean().sqrt()
    if var1.item()==0 or var2.item()==0: return torch.tensor(0.0, device=X.device)
    return hsic / (var1 * var2)

def safe_load(path, device):
    try:    return torch.load(path, map_location=device, weights_only=True)
    except TypeError: return torch.load(path, map_location=device)

def make_two_color_cmap(start_hex, end_hex, name="custom"):
    return LinearSegmentedColormap.from_list(name, [start_hex, end_hex], N=256)

def main():
    ap=argparse.ArgumentParser()
    # 数据/模型
    ap.add_argument("--fast-dir", default="fast"); ap.add_argument("--slow-dir", default="slow"); ap.add_argument("--outputs-dir", default="outputs")
    ap.add_argument("--num-classes", type=int, required=True); ap.add_argument("--length", type=int, default=3200); ap.add_argument("--dt", type=float, default=2e-4)
    ap.add_argument("--train", type=float, default=70); ap.add_argument("--val", type=float, default=25); ap.add_argument("--test", type=float, default=25); ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--batch-size", type=int, default=64); ap.add_argument("--max-samples", type=int, default=256)
    ap.add_argument("--class-idx", type=int, required=True)
    ap.add_argument("--cmap-start", default="#A0CED9"); ap.add_argument("--cmap-end", default="#3A6EA5")
    ap.add_argument("--save", default="cka_class.png")
    # —— 新增：外观（字体/字号等） ——
    ap.add_argument("--font-family", default="", help="e.g. Calibri, Arial, SimHei（可留空）")
    ap.add_argument("--title-size", type=float, default=13.0)
    ap.add_argument("--tick-size", type=float, default=8.0)
    ap.add_argument("--label-size", type=float, default=10.0)   # 用于colorbar标签字号
    ap.add_argument("--cbar-tick-size", type=float, default=9.0)
    ap.add_argument("--x-rotation", type=float, default=45.0)
    ap.add_argument("--fig-w", type=float, default=10.0)
    ap.add_argument("--fig-h", type=float, default=8.0)
    ap.add_argument("--dpi", type=int, default=220)
    args=ap.parse_args()

    # 字体（如果提供了就应用）
    if getattr(args, "font_family", ""):
        plt.rcParams["font.family"] = args.font_family
        plt.rcParams["font.sans-serif"] = [args.font_family]

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xf_te, Xs_te, yte = build_fusion_test(args.fast_dir,args.slow_dir,args.num_classes,args.length,
                                          args.train,args.val,args.test,args.seed)

    idx = torch.nonzero(yte==args.class_idx).squeeze(1)
    if idx.numel()==0:
        raise RuntimeError(f"No samples for class {args.class_idx} in test split.")
    if idx.numel()>args.max_samples:
        idx = idx[:args.max_samples]

    dl = DataLoader(TensorDataset(Xf_te[idx], Xs_te[idx], yte[idx]),
                    batch_size=args.batch_size, shuffle=False)

    model=FusionModel(args.length,args.num_classes).to(device)
    model.load_state_dict(safe_load(os.path.join(args.outputs_dir,"best_fusion.pt"), device))
    model.eval()

    reps = {"fast.conv1":[], "fast.conv2":[], "fast.conv3":[], "fast.afterDrop":[], "fast.emb512":[],
            "slow.conv1":[], "slow.conv2":[], "slow.conv3":[], "slow.afterDrop":[], "slow.emb512":[],
            "fusion.emb256":[]}

    @torch.no_grad()
    def pool_flat(t): return t.mean(dim=2)

    with torch.no_grad():
        for xf,xs,_ in dl:
            xf=xf.to(device); xs=xs.to(device)
            _, packs = model(xf,xs,collect=True)
            Ff, Fs, Fu = packs["fast"], packs["slow"], packs["fusion"]
            for name,t in zip(["conv1","conv2","conv3","afterDrop"], Ff[:-1]):
                reps[f"fast.{name}"].append(pool_flat(t).cpu())
            reps["fast.emb512"].append(Ff[-1].cpu())
            for name,t in zip(["conv1","conv2","conv3","afterDrop"], Fs[:-1]):
                reps[f"slow.{name}"].append(pool_flat(t).cpu())
            reps["slow.emb512"].append(Fs[-1].cpu())
            reps["fusion.emb256"].append(Fu.cpu())

    for k in reps: reps[k]=torch.cat(reps[k],0)

    keys=list(reps.keys()); M=torch.zeros(len(keys),len(keys))
    for i,ki in enumerate(keys):
        for j,kj in enumerate(keys):
            M[i,j]=linear_cka(reps[ki].to(device), reps[kj].to(device)).cpu()

    cmap = make_two_color_cmap(args.cmap_start, args.cmap_end, "custom")
    plt.figure(figsize=(args.fig_w, args.fig_h), dpi=args.dpi)
    im = plt.imshow(M, cmap=cmap, vmin=0, vmax=1)
    cb = plt.colorbar(im)
    cb.set_label("Linear CKA", fontsize=args.label_size)
    cb.ax.tick_params(labelsize=args.cbar_tick_size)

    plt.xticks(range(len(keys)), keys, rotation=args.x_rotation, ha="right", fontsize=args.tick_size)
    plt.yticks(range(len(keys)), keys, fontsize=args.tick_size)
    plt.title(f"CKA by class={args.class_idx}", fontsize=args.title_size)

    out=os.path.join(args.outputs_dir, args.save)
    plt.tight_layout(); plt.savefig(out, dpi=args.dpi); plt.close()
    print("Saved:", out)

if __name__=="__main__":
    main()
