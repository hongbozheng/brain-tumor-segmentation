#!/usr/bin/env python3


import argparse
import os
import torch
from config import DEVICE, MODEL_NAMES, get_config
import logger
from data import image_label, val_transform
from dataset import BraTS
from nnformer.nnFormer import nnFormer
from swin_unetr import SwinUNETR
from torch.utils.data import DataLoader
from unetr import UNETR
from unetr_pp.unetr_pp import UNETR_PP
from unet3d import UNet3D
from inference import inference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        '-m',
        type=str,
        required=True,
        help="model names: 'swin', 'unetr', 'unetr_pp', 'nnformer', "
             "'unet3d'"
    )
    parser.add_argument(
        "--directory",
        '-d',
        type=str,
        required=True,
        help="directory containing 3D-MRI"
    )
    parser.add_argument(
        "--filepath",
        '-f',
        type=str,
        required=True,
        help="inference results filepath"
    )

    args = parser.parse_args()
    model_name = args.model
    data_dir = args.directory
    filepath = args.filepath

    if model_name not in MODEL_NAMES:
        logger.log_error("Invalid model name.")
        logger.log_error(f"Please choose from {MODEL_NAMES}")
        exit(1)

    config = get_config(args=None)

    if model_name == "swin":
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
        )
        ckpt_filepath = config.BEST_MODEL.SWIN
        roi = config.MODEL.SWIN.ROI
    elif model_name == "unetr":
        model = UNETR(
            in_channels=config.MODEL.UNETR.IN_CHANNELS,
            out_channels=config.MODEL.UNETR.OUT_CHANNELS,
            img_size=config.MODEL.UNETR.ROI,
            feature_size=config.MODEL.UNETR.FEATURE_SIZE,
            hidden_size=config.MODEL.UNETR.HIDDEN_SIZE,
            mlp_dim=config.MODEL.UNETR.MLP_DIM,
            num_heads=config.MODEL.UNETR.NUM_HEADS,
            proj_type=config.MODEL.UNETR.PROJ_TYPE,
            norm_name=config.MODEL.UNETR.NORM_NAME,
            conv_block=config.MODEL.UNETR.CONV_BLOCK,
            res_block=config.MODEL.UNETR.RES_BLOCK,
            dropout_rate=config.MODEL.UNETR.DROPOUT_RATE,
            spatial_dims=config.MODEL.UNETR.SPATIAL_DIMS,
            qkv_bias=config.MODEL.UNETR.QKV_BIAS,
            save_attn=config.MODEL.UNETR.SAVE_ATTN,
        )
        ckpt_filepath = config.BEST_MODEL.UNETR
        roi = config.MODEL.UNETR.ROI
    elif model_name == "unetr_pp":
        model = UNETR_PP(
            in_channels=config.MODEL.UNETR_PP.IN_CHANNELS,
            out_channels=config.MODEL.UNETR_PP.OUT_CHANNELS,
            feature_size=config.MODEL.UNETR_PP.FEATURE_SIZE,
            hidden_size=config.MODEL.UNETR_PP.HIDDEN_SIZE,
            num_heads=config.MODEL.UNETR_PP.NUM_HEADS,
            pos_embed=config.MODEL.UNETR_PP.POS_EMBED,
            norm_name=config.MODEL.UNETR_PP.NORM_NAME,
            dropout_rate=config.MODEL.UNETR_PP.DROPOUT_RATE,
            depths=config.MODEL.UNETR_PP.DEPTHS,
            dims=config.MODEL.UNETR_PP.DIMS,
            # conv_op=config.CONV_OP,
            do_ds=config.MODEL.UNETR_PP.DO_DS,
        )
        ckpt_filepath = config.BEST_MODEL.UNETR_PP
        roi = config.MODEL.UNETR_PP.ROI
    elif model_name == "nnformer":
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
        ckpt_filepath = config.BEST_MODEL.NNFORMER
        roi = config.MODEL.NNFORMER.CROP_SIZE
    elif model_name == "unet3d":
        model = UNet3D(
            in_channels=config.MODEL.UNET3D.IN_CHANNELS,
            n_class=config.MODEL.UNET3D.N_CLASS,
            kernels=config.MODEL.UNET3D.KERNELS,
            strides=config.MODEL.UNET3D.STRIDES,
            norm=config.MODEL.UNET3D.NORM,
            dim=config.MODEL.UNET3D.DIM,
            deep_supervision=config.MODEL.UNET3D.DEEP_SUPERVISION,
        )
        ckpt_filepath = config.BEST_MODEL.UNET3D
        roi = config.MODEL.UNET3D.ROI

    ids = os.listdir(path=data_dir)
    ids.sort()
    paths = [os.path.join(data_dir, id) for id in ids]
    data = image_label(paths=paths)

    # dataset
    dataset = BraTS(data=data, transform=val_transform)

    # dataloader
    loader = DataLoader(
        dataset=dataset,
        batch_size=config.VAL.BATCH_SIZE,
        shuffle=False,
        num_workers=config.LOADER.NUM_WORKERS,
        pin_memory=config.LOADER.PIN_MEMORY,
    )

    preds = inference(
        model=model,
        device=DEVICE,
        ckpt_filepath=ckpt_filepath,
        loader=loader,
        roi=roi,
        sw_batch_size=config.VAL.SW_BATCH_SIZE,
        overlap=config.VAL.OVERLAP,
    )

    torch.save(obj=preds, f=filepath)

    return


if __name__ == "__main__":
    main()