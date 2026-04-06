# -*- coding: utf-8 -*-
"""
WKNN fingerprint localization + ONNX export

功能：
1. 读取 sss.csv（兼容你上面那种串口调试格式）
2. 从 CSV 构建指纹库
3. 用 WKNN（Weighted KNN）实现指纹定位
4. 将 WKNN 推理过程封装成 PyTorch 模型并导出 ONNX

输入（ONNX）:
  x : float32, shape = (1, F)

输出（ONNX）:
  y : float32, shape = (1, 2)
      [pred_x, pred_y]

默认特征：
  [v1, v2, v3, v4, sx, sy, vavg]

其中：
  sx   = v2 - v1
  sy   = v4 - v3
  vavg = mean(v1,v2,v3,v4)

说明：
- 训练本质上不是神经网络训练，而是“建立指纹数据库”
- ONNX 内部保存参考库 fingerprints + positions
- 推理时输入一个指纹特征，输出预测坐标
"""

import argparse
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

import torch
import torch.nn as nn


CSV_DEFAULT = r"sss.csv"
OUT_ONNX_DEFAULT = r"wknn_fingerprint.onnx"


# -----------------------------
# 1) 解析串口调试 CSV
# -----------------------------
def parse_serial_debug(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    # 如果第一行有 PROC 之类的表头，去掉
    if lines and ("PROC" in lines[0] or "proc" in lines[0]):
        lines = lines[1:]

    # 格式1：逗号分隔
    p_head_comma = re.compile(
        r"Loc\(x y\):,([-\d.]+),([-\d.]+),ADC\(V\):,([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+)"
    )

    # 格式2：空格分隔
    p_head_space = re.compile(
        r"Loc\(x y\):\s*([-\d.]+)\s+([-\d.]+)\s+ADC\(V\):\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"
    )

    # 格式3：续行，只给 4 个 ADC 值（继承上一行坐标）
    p_tail_space = re.compile(
        r"^\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*$"
    )

    rows = []
    cur_x, cur_y = None, None

    for s in lines:
        m = p_head_comma.search(s)
        if m:
            cur_x, cur_y = float(m.group(1)), float(m.group(2))
            v1, v2, v3, v4 = map(float, m.group(3, 4, 5, 6))
            rows.append([cur_x, cur_y, v1, v2, v3, v4])
            continue

        m = p_head_space.search(s)
        if m:
            cur_x, cur_y = float(m.group(1)), float(m.group(2))
            v1, v2, v3, v4 = map(float, m.group(3, 4, 5, 6))
            rows.append([cur_x, cur_y, v1, v2, v3, v4])
            continue

        m2 = p_tail_space.match(s)
        if m2 and cur_x is not None:
            v1, v2, v3, v4 = map(float, m2.group(1, 2, 3, 4))
            rows.append([cur_x, cur_y, v1, v2, v3, v4])

    if not rows:
        raise RuntimeError("Parsed 0 rows. CSV format not matched.")

    df = pd.DataFrame(rows, columns=["x", "y", "v1", "v2", "v3", "v4"])
    df["vavg"] = df[["v1", "v2", "v3", "v4"]].mean(axis=1)
    return df


# -----------------------------
# 2) 指纹特征构造
# -----------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # 假设：
    # v1=left, v2=right, v3=down, v4=up
    out["sx"] = out["v2"] - out["v1"]
    out["sy"] = out["v4"] - out["v3"]
    out["vavg"] = out[["v1", "v2", "v3", "v4"]].mean(axis=1)

    return out


def aggregate_fingerprint_database(df: pd.DataFrame) -> pd.DataFrame:
    """
    同一 (x,y) 可能采了多次，这里对同一坐标做均值聚合，得到参考指纹库。
    """
    feat_cols = ["v1", "v2", "v3", "v4", "sx", "sy", "vavg"]
    db = df.groupby(["x", "y"], as_index=False)[feat_cols].mean()
    return db


# -----------------------------
# 3) 归一化
# -----------------------------
@dataclass
class NormStats:
    mean: np.ndarray
    std: np.ndarray


def compute_norm_stats(X: np.ndarray) -> NormStats:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return NormStats(mean=mean.astype(np.float32), std=std.astype(np.float32))


def apply_norm(X: np.ndarray, stats: NormStats) -> np.ndarray:
    return ((X - stats.mean) / stats.std).astype(np.float32)


# -----------------------------
# 4) WKNN 模型（可导出 ONNX）
# -----------------------------
class WKNNFingerprintModel(nn.Module):
    """
    将参考指纹库存成模型常量：
      fingerprints: (N, F)
      positions:    (N, 2)

    输入：
      x: (B, F)

    输出：
      y: (B, 2)
    """
    def __init__(self, fingerprints: np.ndarray, positions: np.ndarray, k: int = 4, p: float = 2.0):
        super().__init__()

        if fingerprints.ndim != 2:
            raise ValueError("fingerprints must be 2D")
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("positions must be (N,2)")
        if fingerprints.shape[0] != positions.shape[0]:
            raise ValueError("fingerprints and positions size mismatch")

        self.k = int(k)
        self.p = float(p)

        self.register_buffer(
            "fingerprints",
            torch.tensor(fingerprints, dtype=torch.float32)
        )
        self.register_buffer(
            "positions",
            torch.tensor(positions, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, F)
        """
        # 计算欧氏距离
        # diff: (B, N, F)
        diff = x.unsqueeze(1) - self.fingerprints.unsqueeze(0)
        dist = torch.sqrt(torch.sum(diff * diff, dim=-1) + 1e-12)  # (B, N)

        # 找最近的 k 个参考点
        k_eff = min(self.k, self.fingerprints.shape[0])
        d_k, idx_k = torch.topk(dist, k_eff, dim=1, largest=False, sorted=True)  # (B, k)

        # 取对应位置
        pos_k = self.positions[idx_k]  # (B, k, 2)

        # WKNN 权重：w_i = 1 / d_i^p
        w_k = 1.0 / torch.pow(d_k + 1e-6, self.p)  # (B, k)
        w_sum = torch.sum(w_k, dim=1, keepdim=True) + 1e-12
        w_norm = w_k / w_sum  # (B, k)

        # 加权求坐标
        pred = torch.sum(pos_k * w_norm.unsqueeze(-1), dim=1)  # (B, 2)
        return pred


# -----------------------------
# 5) 评估
# -----------------------------
def evaluate_wknn(model: WKNNFingerprintModel, X: np.ndarray, Y: np.ndarray):
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X.astype(np.float32))).cpu().numpy()

    err = np.sqrt(np.sum((pred - Y) ** 2, axis=1))
    mae_x = np.mean(np.abs(pred[:, 0] - Y[:, 0]))
    mae_y = np.mean(np.abs(pred[:, 1] - Y[:, 1]))
    rmse = np.sqrt(np.mean(err ** 2))
    mean_err = np.mean(err)

    return {
        "mae_x": float(mae_x),
        "mae_y": float(mae_y),
        "rmse": float(rmse),
        "mean_err": float(mean_err),
    }


# -----------------------------
# 6) ONNX 导出
# -----------------------------
def export_onnx(model: nn.Module, feat_dim: int, out_onnx: str):
    import onnx

    model.eval()
    dummy = torch.zeros((1, feat_dim), dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        out_onnx,
        input_names=["x"],
        output_names=["y"],
        opset_version=13,
        do_constant_folding=True,
        dynamo=False,
    )

    # 与你原来的导出风格一致，强制 IR 版本低一点，方便兼容
    m = onnx.load(out_onnx)
    m.ir_version = 9
    for ops in m.opset_import:
        if ops.domain in ("", "ai.onnx") and ops.version > 13:
            ops.version = 13
    onnx.save(m, out_onnx)


# -----------------------------
# 7) 整体流程
# -----------------------------
def build_wknn_and_export(
    csv_path: str,
    out_onnx: str,
    k: int = 4,
    p: float = 2.0,
    test_ratio: float = 0.2,
    seed: int = 42,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 读取数据
    df_raw = parse_serial_debug(csv_path)
    print("[INFO] parsed rows:", len(df_raw))

    df_feat = build_features(df_raw)
    print("[INFO] total samples:", len(df_feat))

    # 参考指纹库：同一坐标取均值
    db = aggregate_fingerprint_database(df_feat)
    print("[INFO] reference points:", len(db))

    feat_cols = ["v1", "v2", "v3", "v4", "sx", "sy", "vavg"]

    # 构建“样本集”
    X_all = df_feat[feat_cols].to_numpy(dtype=np.float32)
    Y_all = df_feat[["x", "y"]].to_numpy(dtype=np.float32)

    valid = np.isfinite(X_all).all(axis=1) & np.isfinite(Y_all).all(axis=1)
    X_all = X_all[valid]
    Y_all = Y_all[valid]

    if len(X_all) < 10:
        raise RuntimeError(f"too few valid samples: {len(X_all)}")

    # 划分 train/test
    idx = np.arange(len(X_all))
    np.random.shuffle(idx)
    n_test = max(1, int(len(idx) * test_ratio))
    te = idx[:n_test]
    tr = idx[n_test:]

    if len(tr) < 2:
        raise RuntimeError("training set too small after split")

    # 归一化统计量只用训练集
    stats = compute_norm_stats(X_all[tr])
    Xn_all = apply_norm(X_all, stats)

    # 参考库也要归一化
    db_X = db[feat_cols].to_numpy(dtype=np.float32)
    db_Y = db[["x", "y"]].to_numpy(dtype=np.float32)
    db_Xn = apply_norm(db_X, stats)

    # 建立模型
    model = WKNNFingerprintModel(
        fingerprints=db_Xn,
        positions=db_Y,
        k=k,
        p=p,
    )

    # 评估
    metrics_tr = evaluate_wknn(model, Xn_all[tr], Y_all[tr])
    metrics_te = evaluate_wknn(model, Xn_all[te], Y_all[te])

    print("\n[TRAIN]")
    for kk, vv in metrics_tr.items():
        print(f"{kk}: {vv:.6f}")

    print("\n[TEST]")
    for kk, vv in metrics_te.items():
        print(f"{kk}: {vv:.6f}")

    # 保存归一化参数
    np.savez(
        out_onnx + ".norm.npz",
        mean=stats.mean.astype(np.float32),
        std=stats.std.astype(np.float32),
        feature_names=np.array(feat_cols),
    )

    # 保存参考库，便于 MCU / 上位机排查
    np.savez(
        out_onnx + ".db.npz",
        fingerprints=db_Xn.astype(np.float32),
        positions=db_Y.astype(np.float32),
        raw_fingerprints=db_X.astype(np.float32),
        feature_names=np.array(feat_cols),
        k=np.array([k], dtype=np.int32),
        p=np.array([p], dtype=np.float32),
    )

    # 导出 ONNX
    export_onnx(model, feat_dim=len(feat_cols), out_onnx=out_onnx)

    print(f"\n[OK] Exported ONNX: {out_onnx}")
    print(f"[OK] Saved norm:    {out_onnx}.norm.npz")
    print(f"[OK] Saved db:      {out_onnx}.db.npz")
    print("ONNX input : x shape=(1,7)")
    print("ONNX output: y shape=(1,2) => [pred_x, pred_y]")


# -----------------------------
# 8) onnxruntime 推理示例
# -----------------------------
def demo_onnx_inference(onnx_path: str, norm_path: str, sample_feat: np.ndarray):
    import onnxruntime as ort

    d = np.load(norm_path, allow_pickle=True)
    mean = d["mean"].astype(np.float32)
    std = d["std"].astype(np.float32)

    x = ((sample_feat.astype(np.float32) - mean) / std).reshape(1, -1).astype(np.float32)

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    y = sess.run(["y"], {"x": x})[0]
    return y


# -----------------------------
# 9) main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=CSV_DEFAULT, help="sss.csv path")
    ap.add_argument("--out", type=str, default=OUT_ONNX_DEFAULT, help="output onnx file")
    ap.add_argument("--k", type=int, default=4, help="WKNN k")
    ap.add_argument("--p", type=float, default=2.0, help="weight exponent, w=1/d^p")
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(
            f"CSV not found: {args.csv}\n"
            "example:\n"
            "python wknn_fingerprint_onnx.py --csv sss.csv --out wknn_fingerprint.onnx --k 4 --p 2.0"
        )

    build_wknn_and_export(
        csv_path=args.csv,
        out_onnx=args.out,
        k=args.k,
        p=args.p,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
