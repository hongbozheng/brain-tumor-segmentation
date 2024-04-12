#!/usr/bin/env python3


from config import SEED, DEVICE, get_config
from data import *
from lr_scheduler import LinearWarmupCosineAnnealingLR
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from torch import optim as optim
from train import train_model
from swin_unetr import SwinUNETR


def main() -> None:
    config = get_config(args=None)

    # split data into train & val in dict format
    train_data, val_data = train_val_split(
        data_dir=config.DATA.DATA_DIR,
        seed=SEED,
        val_pct=config.DATA.VAL_PCT,
    )

    # create train & val dataloaders
    train_loader, val_loader = loader(
        train_data=train_data,
        val_data=val_data,
        batch_size=config.TRAIN.BATCH_SIZE,
        num_workers=config.TRAIN.NUM_WORKERS,
        pin_memory=config.TRAIN.PIN_MEMORY,
        roi=config.MODEL.SWIN.ROI,
    )

    # define model
    model = SwinUNETR(
        img_size=config.MODEL.SWIN.ROI,
        in_channels=config.MODEL.SWIN.IN_CHANNELS,
        out_channels=config.MODEL.SWIN.OUT_CHANNELS,
        depths=config.MODEL.SWIN.DEPTHS,
        num_heads=config.MODEL.SWIN.NUM_HEADS,
        feature_size=config.MODEL.SWIN.FEATURE_SIZE,
        norm_name=config.MODEL.SWIN.NORM_NAME,
        drop_rate=config.MODEL.SWIN.DROP_RATE,
        attn_drop_rate=config.MODEL.SWIN.ATTN_DROP_RATE,
        dropout_path_rate=config.MODEL.SWIN.DROPOUT_PATH_RATE,
        normalize=config.MODEL.SWIN.NORMALIZE,
        use_checkpoint=config.MODEL.SWIN.USE_CHECKPOINT,
        spatial_dims=config.MODEL.SWIN.SPATIAL_DIMS,
        downsample=config.MODEL.SWIN.DOWNSAMPLE,
        use_v2=config.MODEL.SWIN.USE_V2,
    ).to(device=DEVICE)

    # define optimizer
    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=config.MODEL.SWIN.LR,
        weight_decay=config.MODEL.SWIN.WEIGHT_DECAY,
    )

    # define lr scheduler
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_epochs=config.TRAIN.WARMUP_EPOCHS,
        max_epochs=config.TRAIN.N_EPOCHS,
        warmup_start_lr=config.TRAIN.WARMUP_START_LR,
        eta_min=config.TRAIN.ETA_MIN,
    )

    # loss fn (train)
    loss_fn = DiceLoss(
        include_background=True,
        to_onehot_y=False,
        sigmoid=True,
        softmax=False,
    )

    # acc fn (val)
    acc_fn = DiceMetric(
        include_background=True,
        reduction=MetricReduction.MEAN_BATCH,
        get_not_nans=True,
    )

    train_model(
        model=model,
        device=DEVICE,
        ckpt_filepath=config.SAVE.SWIN_BEST_MODEL,
        optimizer=optimizer,
        scheduler=scheduler,
        n_epochs=config.TRAIN.N_EPOCHS,
        loss_fn=loss_fn,
        roi=config.MODEL.SWIN.ROI,
        sw_batch_size=config.VAL.SW_BATCH_SIZE,
        overlap=config.VAL.OVERLAP,
        acc_fn=acc_fn,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    return


if __name__ == "__main__":
    main()