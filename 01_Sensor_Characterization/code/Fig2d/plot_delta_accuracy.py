#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def read_preds(path):
    y_true, y_pred = [], []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            if not line.strip(): continue
            t, p = line.strip().split()
            y_true.append(int(t)); y_pred.append(int(p))
    return np.array(y_true), np.array(y_pred)

def per_class_acc(y_true, y_pred, num_classes):
    acc = np.zeros(num_classes, dtype=float)
    for c in range(num_classes):
        idx = (y_true == c)
        acc[c] = accuracy_score(y_true[idx], y_pred[idx]) if idx.any() else np.nan
    return acc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", default="outputs")
    ap.add_argument("--num-classes", type=int, required=True)
    ap.add_argument("--labels", type=str, default=None, help="逗号分隔的标签名，长度=类别数。缺省用 1..N")
    ap.add_argument("--save", default="delta_accuracy_heatmap.png")
    args = ap.parse_args()

    paths = {m: os.path.join(args.outputs_dir, f"pred_{m}.txt") for m in ["fast","slow","fusion"]}
    for m,p in paths.items():
        if not os.path.isfile(p):
            raise FileNotFoundError(f"找不到 {p}")

    ytf, ypf = read_preds(paths["fast"])
    yts, yps = read_preds(paths["slow"])
    ytu, ypu = read_preds(paths["fusion"])
    # 断言标签一致（同一测试集）
    if not (np.array_equal(ytf, yts) and np.array_equal(ytf, ytu)):
        print("警告：三个文件的 y_true 不完全一致，将以 fusion 的 y_true 为基准对齐。")
    y_true = ytu

    C = args.num_classes
    acc_fast  = per_class_acc(y_true, ypf, C)
    acc_slow  = per_class_acc(y_true, yps, C)
    acc_fusion= per_class_acc(y_true, ypu, C)

    delta_fast  = acc_fusion - acc_fast   # Fusion − Fast
    delta_slow  = acc_fusion - acc_slow   # Fusion − Slow
    M = np.vstack([delta_fast, delta_slow])  # 2 x C

    labels = args.labels.split(",") if args.labels else [str(i) for i in range(1, C+1)]
    row_names = ["Fusion − Fast", "Fusion − Slow"]

    plt.figure(figsize=(max(8, C*0.4), 3.6))
    plt.imshow(M, aspect="auto", cmap="RdBu", vmin=-1.0, vmax=1.0)
    plt.yticks([0,1], row_names, fontsize=11)
    plt.xticks(range(C), labels, rotation=45, ha="right", fontsize=9)
    plt.colorbar(label="Δ accuracy")
    plt.title("Per-class accuracy gain of Fusion", fontsize=13)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i,j]):
                plt.text(j, i, f"{M[i,j]*100:.1f}%", ha="center", va="center",
                         color="black" if abs(M[i,j])<0.4 else "white", fontsize=8)
    out = os.path.join(args.outputs_dir, args.save)
    plt.tight_layout(); plt.savefig(out, dpi=200); plt.close()
    # 打印总体准确率
    import sklearn.metrics as skm
    print(f"Overall acc  Fast   : {skm.accuracy_score(y_true, ypf):.4f}")
    print(f"Overall acc  Slow   : {skm.accuracy_score(y_true, yps):.4f}")
    print(f"Overall acc  Fusion : {skm.accuracy_score(y_true, ypu):.4f}")
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
