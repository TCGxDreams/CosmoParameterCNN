import numpy as np
import argparse
import os
import time
from spectrum import gaussianize

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--maps_npy", type=str, required=True, help="Duong dan den Maps_Mtot_Nbody_LH_z=0.00.npy")
    p.add_argument("--params_txt", type=str, required=True, help="Duong dan den params_Nbody.txt")
    p.add_argument("--out_dir", type=str, default="data", help="Thu muc ghi dau ra (.npz)")
    p.add_argument("--seed", type=int, default=42, help="Seed ngau nhien cho Gaussianization")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()
    print("Dang load maps...")
    # Shape: (15000, 256, 256)
    maps = np.load(args.maps_npy).astype(np.float32)
    print(f"Loaded maps shape: {maps.shape} in {time.time()-t0:.1f}s")

    t0 = time.time()
    print("Dang load parameters...")
    # Shape: (1000, 6)
    params_lh = np.loadtxt(args.params_txt).astype(np.float32)
    print(f"Loaded params shape: {params_lh.shape}")

    # Tuyen tinh hoa parameters: rep 15 lan vi moi cosmology co 15 realizations
    params = np.repeat(params_lh, 15, axis=0)
    # Lay 2 cot dau tien: Omega_m va sigma_8
    params_target = params[:, :2]
    print(f"Linearized target parameters shape: {params_target.shape}")

    # Chuan hoa maps bang log10 de co truong mat do lognormal sach
    print("Dang tinh log10(maps + 1e-10)...")
    maps_log = np.log10(maps + 1e-10)

    # Kiem chung thu tu: 15 lat cung mot mo phong phai "giong nhau" hon
    # nhieu so voi lat cua mo phong khac (cung cosmology + cung realization 3D)
    assert maps.shape[0] == 15 * params_lh.shape[0]
    m = maps_log.mean(axis=(1, 2)).reshape(-1, 15)   # (1000, 15)
    within = m.std(axis=1).mean()                     # std trong tung khoi 15
    between = m.mean(axis=1).std()                    # std giua cac khoi
    print(f"Ordering check: within-block std = {within:.4f}, "
          f"between-block std = {between:.4f}")
    assert within < between, "Thu tu ban do KHONG khop gia dinh 15 lat/mo phong!"

    out_orig = os.path.join(args.out_dir, "cmd_data.npz")
    np.savez(out_orig, maps=maps_log, params=params_target,
             param_names=np.array(["Omega_m", "sigma_8"]))
    print(f"Saved original CMD dataset to {out_orig}")

    # Tao them ban Gaussianized doi chung luon de khoi mat cong chay rieng
    t_gauss = time.time()
    print("Dang Gaussian hoa toan bo dataset CMD (khoang 15,000 maps)...")
    rng = np.random.default_rng(args.seed)
    
    maps_gauss = np.zeros_like(maps_log)
    for i in range(len(maps_log)):
        if i > 0 and i % 3000 == 0:
            print(f"  Da xong {i}/{len(maps_log)} maps ({time.time()-t_gauss:.1f}s)...")
        maps_gauss[i] = gaussianize(maps_log[i], rng)

    out_gauss = os.path.join(args.out_dir, "cmd_data_gauss.npz")
    np.savez(out_gauss, maps=maps_gauss, params=params_target,
             param_names=np.array(["Omega_m", "sigma_8"]))
    print(f"Saved Gaussianized CMD dataset to {out_gauss} in {time.time()-t_gauss:.1f}s")
    print("Reorganization of CMD dataset complete!")

if __name__ == "__main__":
    main()
