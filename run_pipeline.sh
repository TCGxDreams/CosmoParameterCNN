#!/bin/bash
set -e

echo "=== [1/7] Chạy chuẩn bị dữ liệu (prepare_cmd.py) và Kiểm chứng thứ tự ==="
python3 src/prepare_cmd.py \
    --maps_npy ./cmd_raw/Maps_Mtot_Nbody_IllustrisTNG_LH_z=0.00.npy \
    --params_txt ./cmd_raw/params_LH_Nbody_IllustrisTNG.txt

echo "=== [2/7] Chạy Baseline Ridge & Random Forest trên split mô phỏng ==="
python3 src/baseline.py \
    --data data/cmd_data.npz \
    --split_path cmd_split.npz \
    --out_preds cmd_baseline_preds.npz \
    --maps_per_sim 15

echo "=== [3/7] Huấn luyện CNN Gốc (Log-normal) - 3 Seeds x 25 Epochs ==="
python3 src/train_cnn.py \
    --data data/cmd_data.npz \
    --epochs 25 \
    --seeds 0 1 2 \
    --model_prefix cmd_cnn \
    --split_path cmd_split.npz \
    --maps_per_sim 15

echo "=== [4/7] Huấn luyện CNN Đối chứng (Gaussianized) - 3 Seeds x 25 Epochs ==="
python3 src/train_cnn.py \
    --data data/cmd_data_gauss.npz \
    --epochs 25 \
    --seeds 0 1 2 \
    --model_prefix cmd_cnn_gauss \
    --split_path cmd_split.npz \
    --maps_per_sim 15

echo "=== [5/7] Phân tích và Vẽ hình cho CNN Gốc (Bản tiếng Anh & tiếng Việt) ==="
python3 src/interpret.py \
    --data data/cmd_data.npz \
    --seeds 0 1 2 \
    --model_prefix cmd_cnn \
    --baseline_preds cmd_baseline_preds.npz \
    --split_path cmd_split.npz \
    --maps_per_sim 15 \
    --outdir figs_cmd \
    --lang en

python3 src/interpret.py \
    --data data/cmd_data.npz \
    --seeds 0 1 2 \
    --model_prefix cmd_cnn \
    --baseline_preds cmd_baseline_preds.npz \
    --split_path cmd_split.npz \
    --maps_per_sim 15 \
    --outdir figs_cmd_vi \
    --lang vi

echo "=== [6/7] Phân tích và Vẽ hình cho CNN Đối chứng (Bản tiếng Anh & tiếng Việt) ==="
python3 src/interpret.py \
    --data data/cmd_data_gauss.npz \
    --seeds 0 1 2 \
    --model_prefix cmd_cnn_gauss \
    --baseline_preds cmd_baseline_preds.npz \
    --split_path cmd_split.npz \
    --maps_per_sim 15 \
    --outdir figs_cmd_gauss \
    --lang en

python3 src/interpret.py \
    --data data/cmd_data_gauss.npz \
    --seeds 0 1 2 \
    --model_prefix cmd_cnn_gauss \
    --baseline_preds cmd_baseline_preds.npz \
    --split_path cmd_split.npz \
    --maps_per_sim 15 \
    --outdir figs_cmd_gauss_vi \
    --lang vi

echo "=== [7/7] Đồng bộ hóa các biểu đồ mới cho cả hai bản thảo ==="
# Đồng bộ bản tiếng Anh (figs/)
cp figs_cmd/gaussianized_example.png figs/gaussianized_example.png
cp figs_cmd/sanity_pk.png figs/sanity_pk.png
cp figs_cmd/scatter_true_pred.png figs/scatter_true_pred.png
cp figs_cmd/gaussianization_r2.png figs/gaussianization_r2.png
cp figs_cmd/saliency_by_density.png figs/saliency_by_density.png
cp figs_cmd_gauss/saliency_by_density.png figs/saliency_by_density_gauss.png
cp figs_cmd/noise_robustness.png figs/noise_robustness.png

# Đồng bộ bản tiếng Việt (figs_vi/)
mkdir -p figs_vi
cp figs_cmd_vi/gaussianized_example.png figs_vi/gaussianized_example.png
cp figs_cmd_vi/sanity_pk.png figs_vi/sanity_pk.png
cp figs_cmd_vi/scatter_true_pred.png figs_vi/scatter_true_pred.png
cp figs_cmd_vi/gaussianization_r2.png figs_vi/gaussianization_r2.png
cp figs_cmd_vi/saliency_by_density.png figs_vi/saliency_by_density.png
cp figs_cmd_gauss_vi/saliency_by_density.png figs_vi/saliency_by_density_gauss.png
cp figs_cmd_vi/noise_robustness.png figs_vi/noise_robustness.png

echo "=== TẤT CẢ ĐÃ HOÀN THÀNH THÀNH CÔNG! ==="
