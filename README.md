# Memory-efficient High-ratio Model Compression for Image Super-resolution via Hybrid Hashing

## Installation
1) Create and activate the Conda environment:
```bash
conda env create -f hhps_env.yml
conda activate hhps
```

2) Install basicsr in develop mode:
```bash
cd EDSR
python setup.py develop
```

## Pretrained Models
The pretrained models are provided in:
- EDSR: `EDSR/pretrained/EDSRPS_Lx4_cr001_avg_sr05.pth`
- SwinIR: `SWINIR/pretrained/swinir_sr_lightweight_hhps_x4cr01.pth`

## Benchmark
Download benchmark: [Benchmark](https://drive.google.com/file/d/1yMbItvFKVaCT93yPWmlP3883XtJ-wSee/view?usp=sharing) (from [Iterative-Soft-Shrinkage-SR](https://github.com/Jiamian-Wang/Iterative-Soft-Shrinkage-SR) — thanks!).

After downloading, update the benchmark path according to your download location:
- For EDSR: Modify `dataroot_gt` and `dataroot_lq` in `EDSR/options/test/EDSRPS/test_EDSR_Lx4_cr001_sr05.yml`
- For SwinIR: Modify `--folder_lq` and `--folder_gt` in `SWINIR/test_SWINIR_HHPS.sh`

## Testing（X4 SR）
- For EDSR (CR:100×):
```bash
cd EDSR
sh test_EDSRL_HHPS.sh
```

- For SwinIR (CR:10×):
```bash
cd SWINIR
sh test_SWINIR_HHPS.sh
```

More models and training code will be released after acceptance.

## Acknowledgements
This project builds upon: [BasicSR](https://github.com/XPixelGroup/BasicSR), [RzLinear](https://github.com/apd10/RzLinear), [KAIR](https://github.com/cszn/KAIR), Thanks to the authors of these works.
