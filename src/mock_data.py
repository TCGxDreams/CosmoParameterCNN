"""
MOCK DATA — truong mat do lognormal 2D, DUNG DE KIEM CHUNG PIPELINE.
Khi co du lieu Quijote/CMD that, chi can thay file .npz nay bang du lieu that
(cung dinh dang: maps (N,L,L), params (N,2)) — moi script khac giu nguyen.

Vat ly cua mock:
- Truong Gaussian g voi pho P(k) ~ k^(-n): tham so n dong vai "Omega_m"
  (dieu khien hinh dang/do doc pho — nhieu cau truc lon hay nho).
- Bien do sigma cua g: dong vai "sigma_8" (bien do thang giang).
- Bien doi lognormal delta = exp(g - sigma^2/2) - 1: tao phi Gaussian
  (duoi nang, bat doi xung) giong truong mat do vu tru thuc — day chinh la
  thong tin ma P(k) KHONG chua, de thi nghiem Gaussian hoa co cai de "bat".

Tham so lay mau kieu Latin Hypercube (giong Quijote LH set).
"""

import argparse
import numpy as np


def latin_hypercube(n, ranges, rng):
    """LH sampling don gian: chia moi chieu thanh n khoang, hoan vi."""
    d = len(ranges)
    samples = np.zeros((n, d))
    for j, (lo, hi) in enumerate(ranges):
        edges = np.linspace(lo, hi, n + 1)
        pts = edges[:-1] + rng.random(n) * (edges[1:] - edges[:-1])
        samples[:, j] = rng.permutation(pts)
    return samples


def gaussian_field(L, slope, rng):
    """Truong Gaussian 2D voi P(k) ~ k^(-slope), chuan hoa unit variance."""
    kx = np.fft.fftfreq(L)
    KX, KY = np.meshgrid(kx, kx, indexing="ij")
    K = np.sqrt(KX**2 + KY**2)
    K[0, 0] = np.inf  # bo mode k=0
    amp = K ** (-slope / 2.0)
    noise = np.fft.fft2(rng.standard_normal((L, L)))
    g = np.fft.ifft2(amp * noise).real
    return (g - g.mean()) / g.std()


def make_mock_dataset(n_maps=1500, L=64, seed=0,
                      slope_range=(1.0, 2.2), sigma_range=(0.4, 1.2)):
    rng = np.random.default_rng(seed)
    params = latin_hypercube(n_maps, [slope_range, sigma_range], rng)
    maps = np.zeros((n_maps, L, L), dtype=np.float32)
    for i, (n_slope, sigma) in enumerate(params):
        g = gaussian_field(L, n_slope, rng) * sigma
        delta = np.exp(g - sigma**2 / 2.0) - 1.0  # lognormal, mean ~ 0
        maps[i] = delta
    return maps, params.astype(np.float32)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_maps", type=int, default=1500)
    p.add_argument("--L", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="mock_data.npz")
    args = p.parse_args()

    maps, params = make_mock_dataset(args.n_maps, args.L, args.seed)
    np.savez_compressed(args.out, maps=maps, params=params,
                        param_names=np.array(["slope (Om-analog)",
                                              "sigma (s8-analog)"]))
    print(f"Da sinh {len(maps)} ban do {args.L}x{args.L} -> {args.out}")
    print(f"delta: min={maps.min():.2f} max={maps.max():.2f} "
          f"(duoi nang -> phi Gaussian ro)")
