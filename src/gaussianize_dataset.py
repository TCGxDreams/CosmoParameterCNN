import numpy as np
import argparse
from spectrum import gaussianize

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_data", type=str, default="unused/mock_data.npz")
    p.add_argument("--out_data", type=str, default="unused/mock_data_gauss.npz")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    d = np.load(args.in_data)
    maps = d["maps"]
    params = d["params"]
    names = d["param_names"]

    rng = np.random.default_rng(args.seed)
    print(f"Gaussianizing {len(maps)} maps...")
    maps_g = np.array([gaussianize(m, rng) for m in maps], dtype=np.float32)

    np.savez_compressed(args.out_data, maps=maps_g, params=params, param_names=names)
    print(f"Saved Gaussianized dataset to {args.out_data}")

if __name__ == "__main__":
    main()
