#!/usr/bin/env python3


from config import SEED, DEVICE, get_config
from data import *
from lr_scheduler import LinearWarmupCosineAnnealingLR
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from torch import optim as optim
from train import train_model
from unetr import UNETR


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
    model = UNETR(
        in_channels=config.MODEL.UNETR.IN_CHANNELS,
        out_channels=config.MODEL.UNETR.OUT_CHANNELS,
        img_size=config.MODEL.UNETR.ROI,
        feature_size=config.MODEL.UNETR.FEATURE_SIZE,
        hidden_size=config.MODEL.UNETR.HIDDEN_SIZE,
        mlp_dim=config.MODEL.UNETR.MLP_DIM,
        num_heads=config.MODEL.UNETR.NUM_HEADS,
        pos_embed=config.MODEL.UNETR.POS_EMBED,
        norm_name=config.MODEL.UNETR.NORM_NAME,
        conv_block=config.MODEL.UNETR.CONV_BLOCK,
        res_block=config.MODEL.UNETR.RES_BLOCK,
        dropout_rate=config.MODEL.UNETR.DROPOUT_RATE,
        spatial_dims=config.MODEL.UNETR.SPATIAL_DIMS,
        qkv_bias=config.MODEL.UNETR.QKV_BIAS,
        save_attn=config.MODEL.UNETR.SAVE_ATTN,
    ).to(device=DEVICE)

    # define optimizer
    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=config.MODEL.UNETR.LR,
        weight_decay=config.MODEL.UNETR.WEIGHT_DECAY,
    )

    # define lr scheduler
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_epochs=config.TRAIN.WARMUP_EPOCHS,
        max_epochs=config.TRAIN.N_EPOCHS,
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
        ckpt_filepath=config.SAVE.UNETR_BEST_MODEL,
        optimizer=optimizer,
        scheduler=scheduler,
        n_epochs=config.TRAIN.N_EPOCHS,
        loss_fn=loss_fn,
        acc_fn=acc_fn,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    return


if __name__ == "__main__":
    main()