"""
Power spectrum 2D: FFT -> trung binh theo vanh |k|.
Dung chung cho: baseline (Buoc 2) va thi nghiem Gaussian hoa (Buoc 5).
"""

import numpy as np


# Cache for bin edges to avoid recomputing for every map
_bin_cache = {}

def get_bins(N, n_bins):
    key = (N, n_bins)
    if key in _bin_cache:
        return _bin_cache[key]
    
    kx = np.fft.fftfreq(N)
    KX, KY = np.meshgrid(kx, kx, indexing="ij")
    K = np.sqrt(KX**2 + KY**2)
    k_min = K[K > 0].min()
    k_edges = np.geomspace(k_min, 0.5, n_bins + 1)
    
    valid_bins = []
    i = 0
    while i < n_bins:
        start_idx = i
        end_idx = i + 1
        mask = (K >= k_edges[start_idx]) & (K < k_edges[end_idx])
        while mask.sum() < 4 and end_idx < n_bins:
            end_idx += 1
            mask = (K >= k_edges[start_idx]) & (K < k_edges[end_idx])
        if mask.sum() < 4:
            if valid_bins:
                valid_bins[-1] = (valid_bins[-1][0], k_edges[end_idx])
            else:
                valid_bins.append((k_edges[start_idx], k_edges[end_idx]))
        else:
            valid_bins.append((k_edges[start_idx], k_edges[end_idx]))
        i = end_idx
        
    print(f"Log-spaced binning initialized: L={N}, requested n_bins={n_bins}, actual valid_bins={len(valid_bins)}")
    for idx, (lo, hi) in enumerate(valid_bins):
        mask = (K >= lo) & (K < hi)
        print(f"  Bin {idx:2d} [{lo:.4f}, {hi:.4f}): {mask.sum()} modes")
        
    _bin_cache[key] = valid_bins
    return valid_bins


def power_spectrum_2d(field, n_bins=20):
    """Tinh P(k) cua ban do 2D vuong bang log-spacing va gop bin dong.

    Tra ve:
        k_centers: (n_bins_actual,) tam cac vanh k (don vi: chu ky/pixel)
        Pk:        (n_bins_actual,) power trung binh trong tung vanh
    """
    N = field.shape[0]
    F = np.fft.fft2(field)
    P2d = np.abs(F) ** 2 / N**2

    kx = np.fft.fftfreq(N)
    KX, KY = np.meshgrid(kx, kx, indexing="ij")
    K = np.sqrt(KX**2 + KY**2)

    valid_bins = get_bins(N, n_bins)
    
    Pk = []
    k_centers = []
    for lo, hi in valid_bins:
        mask = (K >= lo) & (K < hi)
        Pk.append(P2d[mask].mean() if mask.any() else 0.0)
        k_centers.append(np.sqrt(lo * hi))
        
    return np.array(k_centers), np.array(Pk)


def pk_features(fields, n_bins=20, log=True):
    """Bien moi ban do thanh vector dac trung log P(k) cho baseline."""
    feats = np.array([power_spectrum_2d(f, n_bins)[1] for f in fields])
    if log:
        feats = np.log10(feats + 1e-12)
    return feats


def gaussianize(field, rng):
    """Thi nghiem Gaussian hoa (Buoc 5, thi nghiem 'an diem' nhat):
    giu nguyen bien do Fourier (=> giu nguyen P(k)) nhung thay pha bang
    pha ngau nhien -> pha huy toan bo thong tin phi Gaussian.

    Meo ky thuat: lay pha tu FFT cua nhieu trang thuc de tu dong thoa man
    doi xung Hermite (dam bao ket qua la truong thuc, khong am phan ao).
    """
    F = np.fft.fft2(field)
    amp = np.abs(F)
    W = np.fft.fft2(rng.standard_normal(field.shape))
    phases = W / (np.abs(W) + 1e-30)
    new = np.fft.ifft2(amp * phases).real
    # Giu nguyen mean cua truong goc (mode k=0 bi doi pha)
    new = new - new.mean() + field.mean()
    return new
