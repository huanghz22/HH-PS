python main_test_swinir.py \
    --opt options/swinir/test_swinir_sr_lightweight_hhps_x4cr01.json \
    --task lightweight_sr \
    --folder_lq datasets/benchmark/Set5/LR_bicubic/X4 \
    --folder_gt datasets/benchmark/Set5/HR \
    --dataset Set5

python main_test_swinir.py \
    --opt options/swinir/test_swinir_sr_lightweight_hhps_x4cr01.json \
    --task lightweight_sr \
    --folder_lq datasets/benchmark/Set14/LR_bicubic/X4 \
    --folder_gt datasets/benchmark/Set14/HR \
    --dataset Set5

python main_test_swinir.py \
    --opt options/swinir/test_swinir_sr_lightweight_hhps_x4cr01.json \
    --task lightweight_sr \
    --folder_lq datasets/benchmark/B100/LR_bicubic/X4 \
    --folder_gt datasets/benchmark/B100/HR \
    --dataset Set5

python main_test_swinir.py \
    --opt options/swinir/test_swinir_sr_lightweight_hhps_x4cr01.json \
    --task lightweight_sr \
    --folder_lq datasets/benchmark/Urban100/LR_bicubic/X4 \
    --folder_gt datasets/benchmark/Urban100/HR \
    --dataset Set5

python main_test_swinir.py \
    --opt options/swinir/test_swinir_sr_lightweight_hhps_x4cr01.json \
    --task lightweight_sr \
    --folder_lq datasets/benchmark/Manga109/LR_bicubic/X4_ \
    --folder_gt datasets/benchmark/Manga109/HR \
    --dataset Set5


    