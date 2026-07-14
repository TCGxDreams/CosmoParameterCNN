"""
Split train/val/test CO DINH, luu ra file — dung chung cho MOI mo hinh
(baseline lan CNN) tu Buoc 2 den het du an, dung nhu lo trinh yeu cau.
"""

import numpy as np


def make_or_load_split(n, path="split.npz", frac=(0.7, 0.15, 0.15),
                       seed=42, maps_per_sim=1):
    """Chia train/val/test theo MO PHONG, khong theo ban do.

    Voi CMD: n=15000, maps_per_sim=15 (mo phong i chiem index 15i..15i+14).
    Voi mock: maps_per_sim=1 (moi ban do mot 'mo phong' rieng — hanh vi cu).
    """
    try:
        d = np.load(path)
        assert len(d["train"]) + len(d["val"]) + len(d["test"]) == n
        # Chan viec tai split cu sai kieu: file phai ghi maps_per_sim khop
        assert int(d["maps_per_sim"]) == maps_per_sim
        return d["train"], d["val"], d["test"]
    except (FileNotFoundError, AssertionError, KeyError):
        assert n % maps_per_sim == 0, \
            f"n={n} khong chia het cho maps_per_sim={maps_per_sim}"
        n_sims = n // maps_per_sim
        rng = np.random.default_rng(seed)
        sim_idx = rng.permutation(n_sims)          # hoan vi MO PHONG
        n_tr = int(frac[0] * n_sims)
        n_va = int(frac[1] * n_sims)
        groups = {"train": sim_idx[:n_tr],
                  "val":   sim_idx[n_tr:n_tr + n_va],
                  "test":  sim_idx[n_tr + n_va:]}
        # Trai tu chi so mo phong ra chi so ban do: sim s -> [s*m, s*m+m)
        out = {k: np.concatenate(
                   [np.arange(s * maps_per_sim, (s + 1) * maps_per_sim)
                    for s in v]) for k, v in groups.items()}
        np.savez(path, train=out["train"], val=out["val"], test=out["test"],
                 maps_per_sim=maps_per_sim)
        print(f"Da tao split theo mo phong ({n_sims} sims x {maps_per_sim} "
              f"lat) va luu vao {path}")
        return out["train"], out["val"], out["test"]


def metrics(y_true, y_pred):
    """RMSE va R^2 cho tung tham so."""
    rmse = np.sqrt(((y_true - y_pred) ** 2).mean(axis=0))
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / ss_tot
    return rmse, r2
