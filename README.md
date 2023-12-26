## Channel-Aware Cross-Fused Transformer-style Networks (C2T-Net)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/c2t-net-channel-aware-cross-fused-transformer/pedestrian-attribute-recognition-on-pa-100k)](https://paperswithcode.com/sota/pedestrian-attribute-recognition-on-pa-100k?p=c2t-net-channel-aware-cross-fused-transformer)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/c2t-net-channel-aware-cross-fused-transformer/pedestrian-attribute-recognition-on-peta)](https://paperswithcode.com/sota/pedestrian-attribute-recognition-on-peta?p=c2t-net-channel-aware-cross-fused-transformer)

<img src="https://caodoanh2001.github.io/assets/img/upar-certificate.jpg" data-canonical-src="https://caodoanh2001.github.io/assets/img/upar-certificate.jpg" width="750" height="500" />

Doanh C. Bui, Thinh V. Le and Hung Ba Ngo

### 1. Prepare the dataset from UPAR challenge

Follow the instruction of data download of organizer [website](https://github.com/speckean/upar_challenge).

The data should be arranged as below tree directory:

```
data
├── phase1
│   ├── annotations
│   ├── Market1501
│   ├── market_1501.zip
│   ├── PA100k
│   └── PETA
├── phase2
│   ├── annotations
│   ├── MEVID
│   └── submission_templates_test
```

### 2. Prepare docker image

Download docker image [here](https://drive.google.com/file/d/1sht0y_6lhzP1IAwUb_CRNtuZom6JZnkx/view).

Run the below command to load the docker image:

```
sudo docker load < upar_hdt.tar
```

Go into the `data` folder, run below command to create a container

```
sudo docker run -d --shm-size 8G --gpus="all" -it --name upar_hdt --mount type=bind,source="$(pwd)",target=/home/data upar_hdt:v0
```

Run the container

```
sudo docker exec -ti upar_hdt /bin/bash
```

Then, follow the step 3 for reproducing the results, and step 4 for training.

### 3. Inference for testing dataset in phase 2:

Download our best checkpoint [here](https://drive.google.com/file/d/1YCHeRhEPcyb6fD9byi3flNFSQDsY2qA0/view?usp=drive_link) (`best_model.pth`). Place it under `checkpoints` folder (we already put it in the docker image).

Run the below file for inference:

```
CUDA_VISIBLE_DEVICES=0 python infer_upar_test_phase.py
```

The results are written in `predictions.csv` file. This file is valid for submission in the codalearn portal.


### 3. Training model

Run the below command for training:

```
CUDA_VISIBLE_DEVICES=0 bash run.sh
```

The checkpoints and logs would be saved at `exp_results/upar/`

If this repository proves beneficial for your projects, we kindly request acknowledgment through proper citation:
```
@InProceedings{Bui_2024_WACV,
    author    = {Bui, Doanh C. and Le, Thinh V. and Ngo, Ba Hung},
    title     = {C2T-Net: Channel-Aware Cross-Fused Transformer-Style Networks for Pedestrian Attribute Recognition},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops},
    month     = {January},
    year      = {2024},
    pages     = {351-358}
}
```
