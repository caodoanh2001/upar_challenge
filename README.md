## Channel-Aware Cross-Fused Transformer-style Networks (C$^2$T-Net)

Doanh C. Bui, Thinh V. Le and Hung Ba Ngo

### 1. Prepare the dataset from UPAR challenge

Clone the dataset repository from UPAR organizer

```
git clone https://github.com/speckean/upar_challenge
```

Download datasets for phase 1:

```
python download_datasets.py
```

Downlaod datasets for phase 2 (using `gdown`):

```
gdown --id 1eJKKvWenl6aQE76D0j0asf3YVthVvqpq
```

Datasets for phase 2 are encrypted, hence, should use decryption key to unzip:

```
7z x phase2.zip
```

Key: `UVk4yayzy38zEMKH`

Phase 1 dataset structure:

```
phase1
|__annotations
|__Market1501
|__PA100k
|__annotations
```

Phase 2 dataset structure:

```
phase2
|__annotations
|__MEVID
|__submission_templates_test
```

Modify the `DATASET.PHASE1_ROOT_PATH` and `DATASET.PHASE2_ROOT_PATH` in config file `configs/upar.yaml`

### 2. Inference for testing dataset in phase 2:

Download our best checkpoint [here](https://) (`best_model.pth`). Place it under `checkpoints` folder.

Run the below file for inference:

```
CUDA_VISIBLE_DEVICES=0 python infer_upar_test_phase.py
```

The results are written in `predictions.csv` file. This file is valid for submission in the codalearn portal.


### 3. Training model

We are updating.