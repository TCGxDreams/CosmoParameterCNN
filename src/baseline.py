"""
Buoc 2 — Baseline: du doan (theta1, theta2) tu vector log P(k).
Hai model: Ridge (tuyen tinh) va Random Forest. Day la "|m| cua bai nay":
moi ket qua CNN sau nay phai so voi con so o day.

Cach dung:
    python baseline.py --data mock_data.npz
"""

import argparse
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from spectrum import pk_features
from dataset import make_or_load_split, metrics


def run_baseline(data_path, n_bins=20, split_path="split.npz", out_preds="baseline_preds.npz", maps_per_sim=1):
    d = np.load(data_path)
    maps, params = d["maps"], d["params"]
    tr, va, te = make_or_load_split(len(maps), split_path, maps_per_sim=maps_per_sim)

    print("Tinh power spectrum cho toan bo ban do...")
    X = pk_features(maps, n_bins=n_bins)
    y = params

    results = {}
    for name, model in [
        ("Ridge", Ridge(alpha=1.0)),
        ("RandomForest", MultiOutputRegressor(
            RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1))),
    ]:
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        rmse, r2 = metrics(y[te], pred)
        results[name] = (rmse, r2, pred)
        print(f"\n[{name}] (tap test)")
        for j, pname in enumerate(d["param_names"]):
            print(f"  {pname:20s} RMSE = {rmse[j]:.4f}  R2 = {r2[j]:.4f}")

    # Luu prediction cua baseline tot nhat de ve chung voi CNN sau nay
    best = max(results, key=lambda k: results[k][1].mean())
    np.savez(out_preds, pred=results[best][2], y=y[te],
             name=best, rmse=results[best][0], r2=results[best][1])
    print(f"\nBaseline chinh thuc: {best} (da luu {out_preds})")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="mock_data.npz")
    p.add_argument("--n_bins", type=int, default=20)
    p.add_argument("--split_path", type=str, default="split.npz")
    p.add_argument("--out_preds", type=str, default="baseline_preds.npz")
    p.add_argument("--maps_per_sim", type=int, default=1)
    args = p.parse_args()
    run_baseline(args.data, n_bins=args.n_bins, split_path=args.split_path, out_preds=args.out_preds, maps_per_sim=args.maps_per_sim)
