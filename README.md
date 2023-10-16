# upar

## Download dataset

```
git clone https://github.com/speckean/upar_challenge
```

```
python download_datasets.py
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