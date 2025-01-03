# Brain Tumor Segmentation

## Models
### Transformer
- Swin-UNETR
  - `swin_unetr.py`
  - [GitHub](https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/swin_unetr.py) & [paper](https://arxiv.org/abs/2201.01266)
- UNETR
  - `unetr.py`
  - [GitHub](https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/unetr.py) & [paper](https://arxiv.org/abs/2103.10504)
- UNETR++
  - `unetr_pp/`
  - [GitHub](https://github.com/Amshaker/unetr_plus_plus) & [paper](https://arxiv.org/abs/2212.04497)

- nnFormer
  - `nnformer/`
  - [GitHub](https://github.com/282857341/nnFormer) & [paper](https://arxiv.org/abs/2109.03201)

### CNN
- 3D-UNet
  - `unet3d.py`
  - [GitHub](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/nnUNet) & [paper](https://arxiv.org/abs/2110.03352)

## Data
### BraTS_2023
[Kaggle BraTS_2023 Part 1](https://www.kaggle.com/datasets/aiocta/brats2023-part-1) &
[Kaggle BraTS_2023 Part 2](https://www.kaggle.com/datasets/aiocta/brats2023-part-2zip)

1. Download and unzip both files.
2. Create directory `BraTS_2023` and move all files into this directory.

#### Statistics
- `Total #`: 1251
- `Train #`: 1001 (~80%)
- `Val #`: 250 (~20%)
- `Total Size`: 147.44GB

#### Labels
- `label 1` -> necrotic and non-enhancing tumor core
- `label 2` -> peritumoral edema
- `label 4` -> GD-enhancing tumor

#### Classes
- `TC` -> Tumor Core
  - `Label 1` & `Label 4`
- `WT` -> Whole Tumor
  - `Label 1` & `Label 2` & `Label 4`
- `ET` -> Enhancing Tumor
  - `Label 4`

## Train Models
### Train Configuration
To modify train configuration, check `config.py` file.

### Train
To train 3D-UNet.
```
./train_unet3d.py
```
To train UNETR.
```
./train_unetr.py
```
To train UNETR++.
```
./train_unetr_pp.py
```
Fine-tune Swin-UNETR.
```
./train_swin_unetr.py
```

## Models
### Saved models
Saved models will be in `models` directory.

## Evaluation
The evaluation is performed both quantitatively and qualitatively.
- Dice Similarity Coefficient (Dice Score)
- Visual Segmentation Results

### Evaluation Configuration
To modify test configuration, check `config.py` file.

### Evaluate
To evaluate models.
```
./main.py -m <model name> -d <directory> -f <filepath>
```
- `<model name>` - model name to evaluate
  - unet3d
  - unetr
  - unetr_pp
  - swin
  - nnformer
- `<directory>` - 3D-MRI test set directory
- `<filepath>` - inference result filepath

## Project Structure
```
├── brats (project directory)
│   ├── models                              <- stores best models
│   │   ├── XXXXX_best.ckpt                 <- best model
│   │   ├──       :
│   │   └── XXXXX_best.ckpt
│   ├── nnformer                            <- nnFormer
│   ├── unet_pp                             <- UNETR++
│   ├── .gitignore                          <- .gitignore
│   ├── avgmeter                            <- average meter
│   ├── config.py                           <- configuration file
│   ├── data.py                             <- data preprocess & dataloader
│   ├── dataset.py                          <- BraTS Dataset
│   ├── inference.py                        <- inference pipeline
│   ├── logger.py                           <- log
│   ├── lr_scheduler.py                     <- learning rate schedulers
│   ├── README.md                           <- read me
│   ├── seg.py                              <- predict segmentation results
│   ├── swin_unetr.py                       <- Swin-UNETR
│   ├── test.py                             <- calculate dice score of a model
│   ├── train.py                            <- train epoch & training pipeline
│   ├── train_nnformer.py                   <- nnFormer training script
│   ├── train_swin_unetr.py                 <- Swin-UNETR training script
│   ├── train_unet3d.py                     <- 3D-UNET training script
│   ├── train_unetr.py                      <- UNETR training script
│   ├── train_unetr_pp.py                   <- UNETR++ training script
│   ├── unet3d.py                           <- 3D-UNet
│   ├── unetr.py                            <- UNETR
│   └── val.py                              <- val epoch & val (test) pipeline
│
├── BraTS_2023 (dataset directory, 1251 subdirectories)
│   ├── BraTS-GLI-XXXXX-XXX                 <- 1 brain tumor segmentation subject
│   │   ├── BraTS-GLI-XXXXX-XXX-seg.nii     <- brain tumor segmentation ground truth
│   │   ├── BraTS-GLI-XXXXX-XXX-t1c.nii     <- post-contrast T1-weighted 3D-MRI scans
│   │   ├── BraTS-GLI-XXXXX-XXX-t1n.nii     <- native T1 3D-MRI scans
│   │   ├── BraTS-GLI-XXXXX-XXX-t2f.nii     <- T2 Fluid Attenuated Inversion Recovery (T2-FLAIR)
│   │   └── BraTS-GLI-XXXXX-XXX-t2w.nii     <- T2-weighted 3D-MRI scans
│   ├──         :
│   └── BraTS-GLI-XXXXX-XXX
```
