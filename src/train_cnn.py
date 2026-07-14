"""
Buoc 3-4 — CNN hoi quy: ban do 2D -> (theta1, theta2), loss MSE.
Kien truc giong bai Ising (conv + GAP + FC) nhung dau ra hoi quy 2 gia tri.
Ho tro chay nhieu seed de bao cao mean +/- std nhu quy trinh cu.

Augmentation: xoay 90 do / lat ngau nhien — hop le vi truong dang huong.

Cach dung:
    python train_cnn.py --data mock_data.npz --epochs 15 --seeds 0 1 2
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from dataset import make_or_load_split, metrics


class MapDataset(Dataset):
    def __init__(self, maps, params, augment=False):
        self.maps = maps
        self.params = params
        self.augment = augment

    def __len__(self):
        return len(self.maps)

    def __getitem__(self, i):
        x = self.maps[i]
        if self.augment:
            k = np.random.randint(4)
            x = np.rot90(x, k)
            if np.random.rand() < 0.5:
                x = np.flipud(x)
            x = np.ascontiguousarray(x)
        return torch.from_numpy(x)[None], torch.from_numpy(self.params[i])


class CosmoCNN(nn.Module):
    """Conv x3 + GAP + FC. GAP -> khong phu thuoc kich thuoc ban do."""

    def __init__(self, n_filters=32, n_out=2):
        super().__init__()
        f = n_filters
        self.features = nn.Sequential(
            nn.Conv2d(1, f, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(f, 2*f, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(2*f, 2*f, 3, padding=1), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(2*f, 64), nn.ReLU(), nn.Linear(64, n_out))

    def forward(self, x):
        return self.fc(self.pool(self.features(x)).flatten(1))


def normalize_fit(maps_train):
    """Chuan hoa bang thong ke TAP TRAIN (khong dung log vi truong
    Gaussian hoa co the co delta < -1)."""
    return maps_train.mean(), maps_train.std()


def check_device_consistency(model, sample_batch, device):
    """Kiem tra forward pass tren CPU va GPU/MPS cho cung mot sample batch.
    Max difference phai < 1e-4.
    """
    model.eval()
    # CPU forward pass
    model.to("cpu")
    with torch.no_grad():
        out_cpu = model(sample_batch)
    
    # Device forward pass
    model.to(device)
    with torch.no_grad():
        out_dev = model(sample_batch.to(device)).cpu()
    
    diff = torch.max(torch.abs(out_cpu - out_dev)).item()
    print(f"\n[Device Check] Max absolute difference CPU vs {device}: {diff:.2e}")
    assert diff < 1e-4, f"Output mismatch between CPU and {device}! Diff: {diff:.2e}"
    print("[Device Check] Consistency check PASSED (diff < 1e-4)\n")


def train_one_seed(maps, params, tr, va, te, seed, epochs, batch_size, lr,
                   mu, sd, y_mu, y_sd, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xn = ((maps - mu) / sd).astype(np.float32)
    yn = ((params - y_mu) / y_sd).astype(np.float32)

    dl_tr = DataLoader(MapDataset(Xn[tr], yn[tr], augment=True),
                       batch_size=batch_size, shuffle=True)
    dl_va = DataLoader(MapDataset(Xn[va], yn[va]), batch_size=256)
    dl_te = DataLoader(MapDataset(Xn[te], yn[te]), batch_size=256)

    model = CosmoCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val, best_state = np.inf, None

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl_losses = []
            for xb, yb in dl_va:
                xb, yb = xb.to(device), yb.to(device)
                vl_losses.append(loss_fn(model(xb), yb).item())
            vl = np.mean(vl_losses)
        if vl < best_val:
            best_val, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  seed {seed} epoch {ep:2d}  val_mse = {vl:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in dl_te:
            xb = xb.to(device)
            preds.append(model(xb).cpu().numpy())
    pred = np.concatenate(preds) * y_sd + y_mu  # ve don vi goc
    rmse, r2 = metrics(params[te], pred)
    return model, pred, rmse, r2


def main(data_path, epochs, batch_size, lr, seeds, model_prefix="cnn", split_path="split.npz", maps_per_sim=1):
    d = np.load(data_path)
    maps, params = d["maps"], d["params"]
    tr, va, te = make_or_load_split(len(maps), split_path, maps_per_sim=maps_per_sim)

    mu, sd = normalize_fit(maps[tr])
    y_mu, y_sd = params[tr].mean(0), params[tr].std(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    # Sanity check CPU vs GPU consistency
    if device.type != "cpu":
        test_model = CosmoCNN()
        test_batch = torch.randn(64, 1, maps.shape[1], maps.shape[2])
        check_device_consistency(test_model, test_batch, device)

    all_rmse, all_r2, all_pred = [], [], []
    for s in seeds:
        model, pred, rmse, r2 = train_one_seed(
            maps, params, tr, va, te, s, epochs, batch_size, lr,
            mu, sd, y_mu, y_sd, device)
        all_rmse.append(rmse); all_r2.append(r2); all_pred.append(pred)
        # Move model back to CPU before saving to avoid device-related load issues
        model.cpu()
        torch.save({"state_dict": model.state_dict(),
                    "norm": (mu, sd), "y_norm": (y_mu, y_sd)},
                   f"{model_prefix}_seed{s}.pt")

    all_rmse, all_r2 = np.array(all_rmse), np.array(all_r2)
    print("\n===== CNN, tap test, mean +/- std qua cac seed =====")
    for j, pname in enumerate(d["param_names"]):
        print(f"{pname:20s} RMSE = {all_rmse[:,j].mean():.4f} +/- {all_rmse[:,j].std():.4f}"
              f"   R2 = {all_r2[:,j].mean():.4f} +/- {all_r2[:,j].std():.4f}")

    np.savez(f"{model_prefix}_preds.npz", pred=np.mean(all_pred, axis=0), y=params[te],
             rmse_mean=all_rmse.mean(0), rmse_std=all_rmse.std(0),
             r2_mean=all_r2.mean(0), r2_std=all_r2.std(0))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="mock_data.npz")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--model_prefix", type=str, default="cnn")
    p.add_argument("--split_path", type=str, default="split.npz")
    p.add_argument("--maps_per_sim", type=int, default=1)
    args = p.parse_args()
    main(args.data, args.epochs, args.batch_size, args.lr, args.seeds, args.model_prefix, args.split_path, args.maps_per_sim)
