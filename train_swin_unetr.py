#!/usr/bin/env python3


import config
from data import *
from lr_scheduler import LinearWarmupCosineAnnealingLR
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.utils.enums import MetricReduction
from torch import optim as optim
from train import train_model
from swin_unetr import SwinUNETR


def main() -> None:
    # split data into train & val in dict format
    train_data, val_data = train_val_split(
        data_dir=config.DATA_DIR,
        seed=config.SEED,
        val_pct=config.VAL_PCT,
    )

    # create train & val dataloaders
    train_loader, val_loader = loader(
        train_data=train_data,
        val_data=val_data,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        roi=config.ROI,
    )

    # define model
    model = SwinUNETR(
        img_size=config.ROI,
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
        depths=config.DEPTHS,
        num_heads=config.NUM_HEADS,
        feature_size=config.FEATURE_SIZE,
        norm_name=config.NORM_NAME,
        drop_rate=config.DROP_RATE,
        attn_drop_rate=config.ATTN_DROP_RATE,
        dropout_path_rate=config.DROPOUT_PATH_RATE,
        normalize=config.NORMALIZE,
        use_checkpoint=config.USE_CHECKPOINT,
        spatial_dims=config.SPATIAL_DIMS,
        downsample=config.DOWNSAMPLE,
        use_v2=config.USE_V2,
    ).to(device=config.DEVICE)

    # define optimizer
    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY,
    )

    # define lr scheduler
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_epochs=config.WARMUP_EPOCHS,
        max_epochs=config.N_EPOCHS,
    )

    # loss fn (train)
    loss_fn = DiceLoss(
        include_background=True,
        to_onehot_y=False,
        sigmoid=True,
        softmax=False,
    )

    # postproc
    postproc_fn = AsDiscrete(argmax=False, threshold=0.5)

    # acc fn (val)
    acc_fn = DiceMetric(
        include_background=True,
        reduction=MetricReduction.MEAN_BATCH,
        get_not_nans=True,
    )

    train_model(
        model=model,
        ckpt_filepath=config.SWIN_UNETR_BEST_MODEL,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        postproc_fn=postproc_fn,
        acc_fn=acc_fn,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    return


if __name__ == "__main__":
    main()