# CosmoParameterCNN

Cosmological Parameter Estimation from Density Fields via Convolutional Neural Networks and Interpreting Non-Gaussian Information.

This repository contains the official implementation, datasets guide, and LaTeX manuscripts for estimating cosmological parameters ($\Omega_m$, $\sigma_8$) directly from 2D total matter density maps ($M_{\text{tot}}$) using deep Convolutional Neural Networks (CNNs). We implement a Fourier phase-randomization control framework to isolate and interpret non-Gaussian structural information (cosmic filaments, halos, and voids) in the nonlinear regime.

---

## Repository Structure

```
├── run_pipeline.sh                              # Complete end-to-end execution script
├── LICENSE                                      # MIT License
├── README.md                                    # This guide
├── src/
│   ├── prepare_cmd.py                           # Preprocessing, log transform, and Gaussianization
│   ├── dataset.py                               # Leakage-free simulation-level data split
│   ├── baseline.py                              # Ridge & Random Forest on binned power spectra P(k)
│   ├── train_cnn.py                             # Deep CosmoCNN regression network with GAP
│   └── interpret.py                             # Saliency maps, noise robustness, and OOD domain shift checks
└── figs/                                        # Diagnostic and paper figures
```

---

## Key Experimental Results

The models are trained and validated on the **CAMELS Multifield Dataset (CMD)**IllustrisTNG N-body total matter density maps (15,000 maps of size $256 \times 256$, corresponding to 1,000 different cosmologies with 15 realizations each). Data splits are performed strictly at the **simulation level** to prevent data leakage of large-scale structure footprints.

Performance comparison ($R^2$ score) across models:

| Model | Train Domain | Test Domain | $R^2$ ($\Omega_m$) | $R^2$ ($\sigma_8$) |
| :--- | :---: | :---: | :---: | :---: |
| **Ridge Regression (Original)** | Real (Non-Gaussian) | Real (Non-Gaussian) | `0.6867` | `0.3408` |
| **Ridge Regression (Gauss)** | Gaussianized | Gaussianized | `0.6867` | `0.3408` |
| **CNN (In-Distribution)** | Real (Non-Gaussian) | Real (Non-Gaussian) | **`0.9811 ± 0.0024`** | **`0.9350 ± 0.0008`** |
| **CNN (Gaussianized)** | Gaussianized | Gaussianized | `0.7794 ± 0.0071` | `0.8732 ± 0.0034` |

### Key Physical Insights:
- **Clean Non-Gaussian Phase Contribution ($\Delta R^2_{\text{NG}}$):** Non-linear structures contribute a clean **`20.17%`** for $\Omega_m$ and **`6.18%`** for $\sigma_8$, calculated by subtracting the performance of the Gaussianized CNN from the Original CNN under an in-distribution framework.
- **OOD Domain Shift Bias:** Naive cross-evaluation (testing the Original CNN directly on Gaussianized maps) suffers from statistical domain shift, artificially inflating the performance drop ($R^2$ drops to $-10.0909$ for $\Omega_m$ and $-0.3038$ for $\sigma_8$). This domain shift bias accounts for **`98.18%`** ($\Omega_m$) and **`95.01%`** ($\sigma_8$) of the naive drop.

---

## Installation & Prerequisites

Ensure you have Python 3.8+ installed along with PyTorch, scikit-learn, and matplotlib:

```bash
pip install numpy torch scikit-learn matplotlib
```

---

## Quick Start & Usage Guide

### 1. Download CMD Raw Data
Download the $256 \times 256$ total matter density maps ($M_{\text{tot}}$) and cosmological parameter labels from the official CAMELS server:

```bash
mkdir -p cmd_raw
cd cmd_raw
# Download N-body Mtot 2D maps (~3.66 GB)
curl -O https://users.flatironinstitute.org/~fvillaescusa/priv/DEPnzxoWlaTQ6CjrXqsm0vYi8L7Jy/CMD/2D_maps/data/Nbody/Maps_Mtot_Nbody_IllustrisTNG_LH_z=0.00.npy
# Download parameter labels (1000 cosmologies)
curl -O https://users.flatironinstitute.org/~fvillaescusa/priv/DEPnzxoWlaTQ6CjrXqsm0vYi8L7Jy/CMD/2D_maps/data/Nbody/params_LH_Nbody_IllustrisTNG.txt
cd ..
```

### 2. Run the End-to-End Pipeline
We provide a single bash script that handles data preparation, baseline evaluation, CNN training, and interpretability visualization:

```bash
bash run_pipeline.sh
```

The script runs the following steps sequentially:
1. **Data Preprocessing & Verification:** Computes $\log_{10}(\text{maps} + 10^{-10})$, verifies simulation slice orders, and generates the Fourier phase-randomized Gaussianized control dataset.
2. **Ridge Regression Baseline:** Evaluates linear baselines on dynamic log-spaced power spectrum $P(k)$ bins.
3. **CosmoCNN Training:** Trains 3 independent seeds of the Original CNN on real maps.
4. **Gaussianized CosmoCNN Training:** Trains 3 independent seeds of the control CNN on Gaussianized maps.
5. **Interpretability & Figures Generation:** Generates true-vs-predicted scatter plots, saliency maps grouped by density percentiles, and noise robustness analysis plots.

## Citation

If you use this code or results in your research, please cite:

```text
Nguyễn Vũ Trọng Nhân (2026) ‘Cosmological Parameter Estimation from Density Fields via Convolutional Neural Networks and Interpreting Non-Gaussian Information’. Zenodo, 14 July. Available at: https://doi.org/10.5281/zenodo.21358304.
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
