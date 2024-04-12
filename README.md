# Brain Tumor Segmentation

## Models
#### Transformer
##### Swin-UNETR
swin_unetr.py

[GitHub](https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/swin_unetr.py)

[paper](https://arxiv.org/abs/2201.01266)

##### UNETR
unetr.py

[GitHub](https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/unetr.py)

[paper](https://arxiv.org/abs/2103.10504)

##### UNETR++
unetr_pp/

[GitHub](https://github.com/Amshaker/unetr_plus_plus)

[paper](https://arxiv.org/abs/2212.04497)

##### nnFormer
nnformer/

[GitHub](https://github.com/282857341/nnFormer)

[paper](https://arxiv.org/abs/2109.03201)

#### CNN
##### 3D-UNet
unet3d.py

[GitHub](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/nnUNet)

[paper](https://arxiv.org/abs/2110.03352)


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