# Brain Tumor Segmentation

## Models
#### Transformer
##### Swin-UNETR
swin_unetr.py

##### UNETR
unetr.py

##### UNETR++
unetrpp.py

##### nnFormer
nnformer/

#### CNN
##### 3D-UNet
unet3d.py

## Evaluation
The model will predict 3 classes
- Enhancing Tumor (ET)
- Tumor Core (TC)
- Whole Tumor (WT)

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
│   ├── logger.py                           <- log
│   ├── lr_scheduler.py                     <- learning rate schedulers
│   ├── README.md                           <- read me
│   ├── swin_unetr.py                       <- Swin-UNETR
│   ├── train.py                            <- train epoch & training pipeline
│   ├── train_nnformer.py                   <- nnFormer training script
│   ├── train_swin_unetr.py                 <- Swin-UNETR training script
│   ├── train_unet3d.py                     <- 3D-UNET training script
│   ├── train_unetr.py                      <- UNETR training script
│   ├── train_unetr_pp.py                   <- UNETR++ training script
│   ├── unet3d.py                           <- 3D-UNet
│   ├── unetr.py                            <- UNETR
│   └── val.py                              <- validation epoch
│
├── BraTS_2023 (dataset directory, 1251 folders)
│   ├── BraTS-GLI-XXXXX-XXX                 <- 1 brain tumor segmentation subject
│   │   ├── BraTS-GLI-XXXXX-XXX-seg.nii     <- brain tumor segmentation ground truth
│   │   ├── BraTS-GLI-XXXXX-XXX-t1c.nii     <- post-contrast T1-weighted 3D-MRI scans
│   │   ├── BraTS-GLI-XXXXX-XXX-t1n.nii     <- native T1 3D-MRI scans
│   │   ├── BraTS-GLI-XXXXX-XXX-t2f.nii     <- T2 Fluid Attenuated Inversion Recovery (T2-FLAIR)
│   │   └── BraTS-GLI-XXXXX-XXX-t2w.nii     <- T2-weighted 3D-MRI scans
│   ├──         :
│   └── BraTS-GLI-XXXXX-XXX
```