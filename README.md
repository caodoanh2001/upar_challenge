# upar

## Download dataset

Train & dev-test sets:
```
git clone https://github.com/speckean/upar_challenge
```

```
python download_datasets.py
```

Test set & dev-test annotations:

```
gdown --id 1eJKKvWenl6aQE76D0j0asf3YVthVvqpq
```
```
7z -x phase2.zip
password: UVk4yayzy38zEMKH
```

## Download pretrained models (ImageNet)

Put under `pretrained` folder

```
cd pretrained
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth
```

## Edit directory of datasets

Edit paths in `train.py` (`csv_file` and `root_path`)

```[python3]
train_set = PedesAttrUPAR(cfg=cfg, csv_file='/data3/doanhbc/upar/annotations/phase1/train/train.csv', transform=train_tsfm,
                              root_path='/data3/doanhbc/upar/', target_transform=cfg.DATASET.TARGETTRANSFORM)

valid_set = PedesAttrUPAR(cfg=cfg, csv_file='/data3/doanhbc/upar/annotations/phase1/train/valid_data.csv', transform=valid_tsfm,
                              root_path='/data3/doanhbc/upar/', target_transform=cfg.DATASET.TARGETTRANSFORM)
```

## Training

```
bash run.sh
```

Weights would be saved under the folder `exp_results/swin_b*/*.pth`

## Inference

Before inference, change the ckpt path at line:

```
(76) model = get_reload_weight(model_dir, model, pth='/home/compu/doanhbc/upar_challenge/SOLIDER-PersonAttributeRecognition/exp_result/PA100k/swin_b.sm08/img_model/best_ckpt_max_2023-10-15_10:55:57.pth')
```

and paths of datasets at:

```
(50) valid_set = PedesAttrUPARInferTestPhase(cfg=cfg, csv_file='/data3/doanhbc/upar/phase2/submission_templates_test/task1/predictions.csv', transform=valid_tsfm,
                              root_path='/data3/doanhbc/upar/phase2/', target_transform=cfg.DATASET.TARGETTRANSFORM)
```

in file `infer_upar_test_phase.py`

```
python infer_upar_test_phase.py
```
