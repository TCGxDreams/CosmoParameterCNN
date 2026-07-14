"""
Buoc 5-6 — Dien giai (trai tim tieu luan) + hinh scatter true-vs-pred
Sua doi theo feedback:
1. Chay interpret tren ca 3 model (seed0, seed1, seed2) de lay mean +/- std cho Gaussian hoa,
   trung binh hoa prediction cua 3 model de ve scatter plot (ensemble) va noise robustness.
2. Them buoc sanity check: kiem tra P(k) cua ban Gaussian hoa trung khit ban goc (< 1e-5) va ve sanity_pk.png.
3. Tach rieng gradient cua Omega_m (index 0) va sigma_8 (index 1) cho thi nghiem saliency.
4. Ho tro chay GPU/MPS de thuc hien nhanh tren CMD.
"""

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spectrum import gaussianize, power_spectrum_2d
from dataset import make_or_load_split, metrics
from train_cnn import CosmoCNN


def load_model(path):
    # Load model to CPU by default, can be moved to device later
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = CosmoCNN()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["norm"], ckpt["y_norm"]


def predict(model, maps, norm, y_norm, device, batch=256):
    mu, sd = norm
    y_mu, y_sd = y_norm
    Xn = ((maps - mu) / sd).astype(np.float32)
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xn), batch):
            xb = torch.from_numpy(Xn[i:i+batch]).unsqueeze(1).to(device)  # (B,1,L,L)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds) * y_sd + y_mu


LABELS = {
    "vi": {
        "orig_pk": "Bản gốc (Phi Gaussian thật)",
        "gauss_pk": "Bản Gaussian hóa (Cùng P(k))",
        "xlabel_pk": "k (chu kỳ/pixel)",
        "title_pk": "Kiểm chứng: Bảo toàn phổ công suất P(k)",
        "ensemble_lbl": "CNN Ensemble (3 hạt giống)",
        "true_lbl": "thật",
        "pred_lbl": "Dự đoán",
        "orig_title": "Phi Gaussian thật",
        "gauss_title": "Gaussian hóa (Cùng P(k))",
        "orig_cnn": "CNN Gốc",
        "gauss_cnn": "CNN Gaussian hóa",
        "r2_lbl": "Hiệu năng $R^2$",
        "title_r2": "So sánh hiệu năng $R^2$ với xáo pha Fourier",
        "sal_om": "Độ nhạy $\Omega_m$",
        "sal_s8": "Độ nhạy $\sigma_8$",
        "xlabel_sal": "Phân vị mật độ pixel (%)",
        "ylabel_sal": "Saliency trung bình (chuẩn hóa)",
        "title_sal": "Độ nhạy (Saliency) theo phân vị mật độ pixel",
        "mean_lbl": "Trung bình",
        "xlabel_noise": "Biên độ nhiễu (theo độ lệch chuẩn dữ liệu)",
        "title_noise": "Độ bền trước nhiễu (Trung bình qua 3 hạt giống)"
    },
    "en": {
        "orig_pk": "Original (N-body)",
        "gauss_pk": "Phase-randomized (same P(k))",
        "xlabel_pk": "k (cycles/pixel)",
        "title_pk": "Sanity Check: Power Spectrum P(k) Conservation",
        "ensemble_lbl": "CNN Ensemble (3 seeds)",
        "true_lbl": "True",
        "pred_lbl": "Predicted",
        "orig_title": "Original (N-body)",
        "gauss_title": "Gaussianized (same P(k))",
        "orig_cnn": "Original CNN",
        "gauss_cnn": "Gaussianized CNN",
        "r2_lbl": "$R^2$ Score",
        "title_r2": "Performance Comparison ($R^2$ Score) with Fourier Phase Randomization",
        "sal_om": "$\Omega_m$ Saliency",
        "sal_s8": "$\sigma_8$ Saliency",
        "xlabel_sal": "Pixel Density Percentile (%)",
        "ylabel_sal": "Mean Saliency (normalized)",
        "title_sal": "Saliency by Pixel Density Percentile",
        "mean_lbl": "Mean",
        "xlabel_noise": "Noise Amplitude (in units of field std)",
        "title_noise": "Noise Robustness (Mean over 3 seeds)"
    }
}


def check_pk_conservation(maps_orig, maps_gauss, n_bins=20, outdir="figs", lang="en"):
    """Sanity check: kiem tra P(k) cua ban Gaussian hoa va ban goc lech < 1e-5."""
    diffs = []
    pks_orig = []
    pks_gauss = []
    
    # Lay k tu mau dau tien de lam truc hoanh
    k, _ = power_spectrum_2d(maps_orig[0], n_bins)
    
    for i in range(len(maps_orig)):
        _, pk_orig = power_spectrum_2d(maps_orig[i], n_bins)
        _, pk_gauss = power_spectrum_2d(maps_gauss[i], n_bins)
        pks_orig.append(pk_orig)
        pks_gauss.append(pk_gauss)
        # Relative difference
        diff = np.abs(pk_orig - pk_gauss) / (pk_orig + 1e-30)
        diffs.append(diff)
        
    mean_pk_orig = np.mean(pks_orig, axis=0)
    mean_pk_gauss = np.mean(pks_gauss, axis=0)
    max_diff = np.max(diffs)
    mean_diff = np.mean(diffs)
    
    print("\n===== Sanity check: Bao toan P(k) =====")
    print(f"  Max relative difference across all modes: {max_diff:.2e}")
    print(f"  Mean relative difference across all modes: {mean_diff:.2e}")
    
    # Assert strict conservation as requested (< 1e-5)
    assert max_diff < 1e-5, f"P(k) conservation check failed! Max diff: {max_diff:.2e}"
    print("  Sanity check PASSED: P(k) is conserved within 1e-5 (relative error)")
    
    # Save the verification plot
    plt.figure(figsize=(6, 4.5))
    plt.loglog(k, mean_pk_orig, "b-", lw=1.5, label=LABELS[lang]["orig_pk"])
    plt.loglog(k, mean_pk_gauss, "r--", lw=1.5, label=LABELS[lang]["gauss_pk"])
    plt.xlabel(LABELS[lang]["xlabel_pk"])
    plt.ylabel("P(k)")
    plt.legend()
    plt.title(LABELS[lang]["title_pk"])
    plt.tight_layout()
    plt.savefig(f"{outdir}/sanity_pk.png", dpi=150)
    plt.close()


def fig_scatter(y, pred_ensemble, pred_base, names, outpath, lang="en"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for j, ax in enumerate(axes):
        ax.scatter(y[:, j], pred_base[:, j], s=6, alpha=0.35,
                   label="Baseline P(k)", color="tab:orange")
        ax.scatter(y[:, j], pred_ensemble[:, j], s=6, alpha=0.35,
                   label=LABELS[lang]["ensemble_lbl"], color="tab:blue")
        lims = [y[:, j].min(), y[:, j].max()]
        ax.plot(lims, lims, "k--", lw=1)
        name_en = r"$\Omega_m$" if names[j] == "Omega_m" else r"$\sigma_8$"
        if lang == "vi":
            ax.set_xlabel(f"{name_en} {LABELS[lang]['true_lbl']}")
        else:
            ax.set_xlabel(f"{LABELS[lang]['true_lbl']} {name_en}")
        ax.set_ylabel(LABELS[lang]["pred_lbl"])
        ax.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def exp_gaussianization(models, norm, y_norm, maps, params, te, device, seed=0):
    rng = np.random.default_rng(seed)
    # 1. Tao ban do Gaussianized (pha ngau nhien, cung P(k))
    maps_g = np.array([gaussianize(m, rng) for m in maps[te]], dtype=np.float32)
    
    all_orig_r2, all_orig_rmse = [], []
    all_gauss_r2, all_gauss_rmse = [], []
    all_orig_preds, all_gauss_preds = [], []
    
    # 2. Tinh toan tren tung model/seed
    for idx, model in enumerate(models):
        model.to(device)
        pred_orig = predict(model, maps[te], norm, y_norm, device)
        pred_gauss = predict(model, maps_g, norm, y_norm, device)
        
        rmse_o, r2_o = metrics(params[te], pred_orig)
        rmse_g, r2_g = metrics(params[te], pred_gauss)
        
        all_orig_r2.append(r2_o)
        all_orig_rmse.append(rmse_o)
        all_gauss_r2.append(r2_g)
        all_gauss_rmse.append(rmse_g)
        
        all_orig_preds.append(pred_orig)
        all_gauss_preds.append(pred_gauss)
        
    all_orig_r2, all_orig_rmse = np.array(all_orig_r2), np.array(all_orig_rmse)
    all_gauss_r2, all_gauss_rmse = np.array(all_gauss_r2), np.array(all_gauss_rmse)
    
    # Ensemble predictions (mean of 3 seeds)
    ensemble_orig_pred = np.mean(all_orig_preds, axis=0)
    ensemble_gauss_pred = np.mean(all_gauss_preds, axis=0)
    
    rmse_ens_o, r2_ens_o = metrics(params[te], ensemble_orig_pred)
    rmse_ens_g, r2_ens_g = metrics(params[te], ensemble_gauss_pred)
    
    results = {
        "orig_r2_mean": all_orig_r2.mean(axis=0), "orig_r2_std": all_orig_r2.std(axis=0),
        "gauss_r2_mean": all_gauss_r2.mean(axis=0), "gauss_r2_std": all_gauss_r2.std(axis=0),
        "orig_rmse_mean": all_orig_rmse.mean(axis=0), "orig_rmse_std": all_orig_rmse.std(axis=0),
        "gauss_rmse_mean": all_gauss_rmse.mean(axis=0), "gauss_rmse_std": all_gauss_rmse.std(axis=0),
        "ensemble_orig_pred": ensemble_orig_pred, "ensemble_gauss_pred": ensemble_gauss_pred,
        "ensemble_orig_r2": r2_ens_o, "ensemble_gauss_r2": r2_ens_g
    }
    return results, maps_g


def exp_saliency_by_density(models, norm, maps, te, device, n_samples=100, seed=0):
    """Saliency trung binh theo phan vi mat do cua pixel cho tung tham so (Omega_m va sigma_8 rieng)."""
    mu, sd = norm
    rng = np.random.default_rng(seed)
    idx = rng.choice(te, size=min(n_samples, len(te)), replace=False)
    percentile_bins = np.linspace(0, 100, 11)
    
    sal_by_bin_slope = np.zeros(10)
    sal_by_bin_sigma = np.zeros(10)
    counts = np.zeros(10)
    
    for i in idx:
        dens = maps[i]
        edges = np.percentile(dens, percentile_bins)
        
        model_sals_slope = []
        model_sals_sigma = []
        
        for model in models:
            model.to(device)
            x = torch.from_numpy(((dens - mu) / sd).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            x.requires_grad_(True)
            
            pred = model(x)
            
            # Saliency cho slope (Om-analog, index 0)
            out_slope = pred[0, 0]
            out_slope.backward(retain_graph=True)
            sal_slope = x.grad[0, 0].abs().cpu().numpy()
            model_sals_slope.append(sal_slope)
            
            # Reset gradients
            x.grad.zero_()
            
            # Saliency cho sigma (s8-analog, index 1)
            out_sigma = pred[0, 1]
            out_sigma.backward()
            sal_sigma = x.grad[0, 0].abs().cpu().numpy()
            model_sals_sigma.append(sal_sigma)
            
        mean_sal_slope = np.mean(model_sals_slope, axis=0)
        mean_sal_sigma = np.mean(model_sals_sigma, axis=0)
        
        for b in range(10):
            m = (dens >= edges[b]) & (dens <= edges[b + 1])
            if m.any():
                sal_by_bin_slope[b] += mean_sal_slope[m].mean()
                sal_by_bin_sigma[b] += mean_sal_sigma[m].mean()
                counts[b] += 1
                
    return sal_by_bin_slope / np.maximum(counts, 1), sal_by_bin_sigma / np.maximum(counts, 1)


def exp_noise_robustness(models, norm, y_norm, maps, params, te, device,
                         noise_levels=(0.0, 0.05, 0.1, 0.2, 0.4), seed=0):
    rng = np.random.default_rng(seed)
    sd_data = maps.std()
    
    # 1. Sinh san bo map nhieu cho tung muc nhieu (dung chung cho moi model)
    noisy_maps_all = []
    for nl in noise_levels:
        if nl == 0.0:
            noisy_maps_all.append(maps[te])
        else:
            noise = rng.standard_normal(maps[te].shape).astype(np.float32) * nl * sd_data
            noisy_maps_all.append(maps[te] + noise)
            
    # 2. Danh gia tung model tren cung bo map nhieu nay
    r2_curves = []  # shape: (n_seeds, n_levels, 2)
    
    for model in models:
        model.to(device)
        seed_r2s = []
        for idx, nl in enumerate(noise_levels):
            noisy = noisy_maps_all[idx]
            pred = predict(model, noisy, norm, y_norm, device)
            _, r2 = metrics(params[te], pred)
            seed_r2s.append(r2)
        r2_curves.append(seed_r2s)
        
    return np.array(noise_levels), np.mean(r2_curves, axis=0), np.std(r2_curves, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="mock_data.npz")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--outdir", type=str, default="figs")
    p.add_argument("--model_prefix", type=str, default="cnn")
    p.add_argument("--baseline_preds", type=str, default="baseline_preds.npz")
    p.add_argument("--split_path", type=str, default="split.npz")
    p.add_argument("--maps_per_sim", type=int, default=1)
    p.add_argument("--lang", type=str, default="en", choices=["en", "vi"])
    args = p.parse_args()

    import os
    os.makedirs(args.outdir, exist_ok=True)

    d = np.load(args.data)
    maps, params, names = d["maps"], d["params"], d["param_names"]
    tr, va, te = make_or_load_split(len(maps), args.split_path, maps_per_sim=args.maps_per_sim)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device for inference:", device)

    # 1. Load all seed models
    models = []
    norm, y_norm = None, None
    for s in args.seeds:
        model_path = f"{args.model_prefix}_seed{s}.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Khong tim thay checkpoint model: {model_path}. Hay chay train_cnn.py truoc.")
        model, n_norm, yn_norm = load_model(model_path)
        models.append(model)
        norm, y_norm = n_norm, yn_norm  # Norms are identical because split is identical

    # 2. Baseline load
    base = np.load(args.baseline_preds)

    # 3. Running Gaussianization Experiment (Multi-seed)
    gauss_res, maps_g = exp_gaussianization(models, norm, y_norm, maps, params, te, device)
    
    print("\n===== (1) Thi nghiem Gaussian hoa (mean +/- std tap test qua 3 seeds) =====")
    for j, n in enumerate(names):
        print(f"{n:20s}  R2 goc = {gauss_res['orig_r2_mean'][j]:.4f} +/- {gauss_res['orig_r2_std'][j]:.4f}"
              f" | R2 sau Gaussian hoa = {gauss_res['gauss_r2_mean'][j]:.4f} +/- {gauss_res['gauss_r2_std'][j]:.4f}"
              f" | baseline P(k) = {base['r2'][j]:.4f}")
        
    print("\n[R2 of Ensemble Predictions (Averaged over 3 seeds)]")
    for j, n in enumerate(names):
        print(f"{n:20s}  Ensemble R2 goc = {gauss_res['ensemble_orig_r2'][j]:.4f}"
              f" | Ensemble R2 sau Gaussian hoa = {gauss_res['ensemble_gauss_r2'][j]:.4f}")

    # 4. Sanity check: P(k) conservation
    check_pk_conservation(maps[te][:100], maps_g[:100], outdir=args.outdir, lang=args.lang)

    # 5. Scatter Plot (Ensemble vs Baseline)
    fig_scatter(params[te], gauss_res["ensemble_orig_pred"], base["pred"], names,
                f"{args.outdir}/scatter_true_pred.png", lang=args.lang)
    print(f"\nHinh scatter prediction ensemble duoc ghi vao {args.outdir}/scatter_true_pred.png")

    # Figure: Goc vs Gaussianized Map Example
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    vmax = np.percentile(maps[te][0], 99)
    axes[0].imshow(maps[te][0], vmax=vmax)
    axes[0].set_title(LABELS[args.lang]["orig_title"])
    axes[1].imshow(maps_g[0], vmax=vmax)
    axes[1].set_title(LABELS[args.lang]["gauss_title"])
    for ax in axes: ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{args.outdir}/gaussianized_example.png", dpi=150)
    plt.close()

    # Figure: R2 comparison bar chart (with error bars for CNN)
    x = np.arange(len(names))
    w = 0.25
    plt.figure(figsize=(7, 4.5))
    plt.bar(x - w, gauss_res["orig_r2_mean"], w, yerr=gauss_res["orig_r2_std"], capsize=5, label=LABELS[args.lang]["orig_cnn"])
    plt.bar(x, gauss_res["gauss_r2_mean"], w, yerr=gauss_res["gauss_r2_std"], capsize=5, label=LABELS[args.lang]["gauss_cnn"])
    plt.bar(x + w, base["r2"], w, label="Baseline P(k)")
    tick_names = [r"$\Omega_m$" if n == "Omega_m" else r"$\sigma_8$" for n in names]
    plt.xticks(x, tick_names)
    plt.ylabel(LABELS[args.lang]["r2_lbl"])
    plt.legend()
    plt.title(LABELS[args.lang]["title_r2"])
    plt.tight_layout()
    plt.savefig(f"{args.outdir}/gaussianization_r2.png", dpi=150)
    plt.close()

    # 6. Saliency Experiment (Separated parameters)
    sal_slope, sal_sigma = exp_saliency_by_density(models, norm, maps, te, device)
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(np.arange(5, 100, 10), sal_slope / sal_slope.max(), "o-", color="tab:blue", label=LABELS[args.lang]["sal_om"])
    plt.plot(np.arange(5, 100, 10), sal_sigma / sal_sigma.max(), "s-", color="tab:red", label=LABELS[args.lang]["sal_s8"])
    plt.xlabel(LABELS[args.lang]["xlabel_sal"])
    plt.ylabel(LABELS[args.lang]["ylabel_sal"])
    plt.legend()
    plt.title(LABELS[args.lang]["title_sal"])
    plt.tight_layout()
    plt.savefig(f"{args.outdir}/saliency_by_density.png", dpi=150)
    plt.close()
    
    print("\n===== (2) Saliency theo phan vi mat do (Tach rieng tham so) =====")
    print("thap (void) -> cao (dinh):")
    print("  slope (Om-analog):", np.round(sal_slope / sal_slope.max(), 3))
    print("  sigma (s8-analog):", np.round(sal_sigma / sal_sigma.max(), 3))

    # 7. Noise Robustness Experiment
    nls, r2_mean, r2_std = exp_noise_robustness(models, norm, y_norm, maps, params, te, device)
    plt.figure(figsize=(6.5, 4.5))
    for j, n in enumerate(names):
        color = "tab:blue" if j == 0 else "tab:red"
        marker = "o" if j == 0 else "s"
        label_name = r"$\Omega_m$" if n == "Omega_m" else r"$\sigma_8$"
        plt.plot(nls, r2_mean[:, j], marker + "-", color=color, label=f"{LABELS[args.lang]['mean_lbl']} {label_name}")
        plt.fill_between(nls, r2_mean[:, j] - r2_std[:, j], r2_mean[:, j] + r2_std[:, j],
                         color=color, alpha=0.15)
    plt.xlabel(LABELS[args.lang]["xlabel_noise"])
    plt.ylabel(LABELS[args.lang]["r2_lbl"])
    plt.legend()
    plt.title(LABELS[args.lang]["title_noise"])
    plt.tight_layout()
    plt.savefig(f"{args.outdir}/noise_robustness.png", dpi=150)
    plt.close()
    
    print("\n===== (3) Do ben nhieu ===== (R2 mean theo muc nhieu)")
    print(np.round(r2_mean, 3))
    print(f"\nHinh anh da duoc luu vao {args.outdir}/")


if __name__ == "__main__":
    main()
