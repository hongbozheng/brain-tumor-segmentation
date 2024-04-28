#!/usr/bin/env python3


from config import SEED, DEVICE, get_config
from data import train_val_split, train_transform, val_transform
from dataset import BraTS
from lr_scheduler import LinearWarmupCosineAnnealingLR
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from torch import optim as optim
from torch.utils.data import DataLoader
from train import train_model
from nnformer.nnFormer import nnFormer


def main() -> None:
    config = get_config(args=None)

    # split data into train & val in dict format
    train_data, val_data = train_val_split(
        data_dir=config.DATA.DIR,
        seed=SEED,
        val_pct=config.DATA.VAL_PCT,
    )

    # dataset
    train_dataset = BraTS(data=train_data, transform=train_transform)
    val_dataset = BraTS(data=val_data, transform=val_transform)

    # dataloader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=config.LOADER.NUM_WORKERS_TRAIN,
        pin_memory=config.LOADER.PIN_MEMORY,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.VAL.BATCH_SIZE,
        shuffle=False,
        num_workers=config.LOADER.NUM_WORKERS_VAL,
        pin_memory=config.LOADER.PIN_MEMORY,
    )

    # define model
    model = nnFormer(
        crop_size=config.MODEL.NNFORMER.CROP_SIZE,
        embedding_dim=config.MODEL.NNFORMER.EMBEDDING_DIM,
        input_channels=config.MODEL.NNFORMER.INPUT_CHANNELS,
        num_classes=config.MODEL.NNFORMER.NUM_CLASSES,
        # conv_op=config.MODEL.NNFORMER.CONV_OP,
        depths=config.MODEL.NNFORMER.DEPTHS,
        num_heads=config.MODEL.NNFORMER.NUM_HEADS,
        patch_size=config.MODEL.NNFORMER.PATCH_SIZE,
        window_size=config.MODEL.NNFORMER.WINDOW_SIZE,
        deep_supervision=config.MODEL.NNFORMER.DEEP_SUPERVISION,
    )

    # define optimizer
    optimizer = optim.SGD(
        params=model.parameters(),
        lr=config.MODEL.NNFORMER.LR,
        momentum=config.MODEL.NNFORMER.MOMENTUM,
        weight_decay=config.MODEL.NNFORMER.WEIGHT_DECAY,
        nesterov=config.MODEL.NNFORMER.NESTEROV,
    )

    # define lr scheduler
    '''
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_epochs=config.TRAIN.WARMUP_EPOCHS,
        max_epochs=config.TRAIN.N_EPOCHS,
        warmup_start_lr=config.TRAIN.WARMUP_START_LR,
        eta_min=config.TRAIN.ETA_MIN,
    )
    '''

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode=config.TRAIN.MODE,
        factor=config.TRAIN.FACTOR,
        patience=config.TRAIN.PATIENCE,
        threshold=config.TRAIN.THRESHOLD,
        cooldown=config.TRAIN.COOLDOWN,
        min_lr=config.TRAIN.ETA_MIN,
        eps=config.TRAIN.EPS,
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
        ckpt_filepath=config.BEST_MODEL.NNFORMER,
        optimizer=optimizer,
        scheduler=scheduler,
        n_epochs=config.TRAIN.N_EPOCHS,
        loss_fn=loss_fn,
        roi=config.MODEL.NNFORMER.CROP_SIZE,
        sw_batch_size=config.VAL.SW_BATCH_SIZE,
        overlap=config.VAL.OVERLAP,
        acc_fn=acc_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        stats_filepath="nnformer_" + config.TRAIN.STATS_FILEPATH,
    )

    return


if __name__ == "__main__":
    main()
