import logger
import numpy as np
import os
import torch
import torch.nn as nn
from avg_meter import AverageMeter
from monai.inferers import sliding_window_inference
from torch.utils.data import DataLoader
from tqdm import tqdm


def val_epoch(
        model: nn.Module,
        val_loader: DataLoader,
        device,
        roi: tuple[int, int, int],
        sw_batch_size: int,
        overlap: float,
        acc_fn,
) -> np.ndarray:
    model.eval()

    loader_tqdm = tqdm(iterable=val_loader, position=1)
    loader_tqdm.set_description(
        desc=f"[Batch 0]",
        refresh=True
    )

    acc_meter = AverageMeter()

    with torch.no_grad():
        for idx, batch in enumerate(iterable=loader_tqdm):
            data = batch["image"].to(device=device)
            target = batch["label"].to(device=device)
            logits = sliding_window_inference(
                inputs=data,
                roi_size=roi,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=overlap,
            )
            preds = (torch.sigmoid(input=logits) >= 0.5).to(torch.float32)
            acc_fn.reset()
            acc_fn(y_pred=preds, y=target)
            accs, not_nans = acc_fn.aggregate()
            acc_meter.update(
                val=accs.detach().cpu().numpy(),
                n=not_nans.detach().cpu().numpy()
            )
            dice_tc = acc_meter.avg[0]
            dice_wt = acc_meter.avg[1]
            dice_et = acc_meter.avg[2]
            avg_dice = np.mean(a=acc_meter.avg)

            loader_tqdm.set_description(
                desc=f"[Batch {idx+1}]: dice_tc {dice_tc:.4f} dice_wt {dice_wt:.4f} "
                     f"dice_et {dice_et:.4f} avg_dice {avg_dice:.4f}",
                refresh=True
            )

    return acc_meter.avg


def val_model(
        model: nn.Module,
        device,
        ckpt_filepath: str,
        val_loader: DataLoader,
        roi: tuple[int, int, int],
        sw_batch_size: int,
        overlap: float,
        acc_fn,
) -> np.ndarray:
    model.to(device=device)

    if os.path.exists(path=ckpt_filepath):
        ckpt = torch.load(f=ckpt_filepath, map_location=device)
        model.load_state_dict(state_dict=ckpt["model_state"])
        filename = os.path.basename(ckpt_filepath)
        logger.log_info(f"Loaded '{filename}'")
    else:
        logger.log_error(f"Cannot find ckpt file at '{ckpt_filepath}'.")
        exit(1)

    dice_score = val_epoch(
        model=model,
        val_loader=val_loader,
        device=device,
        roi=roi,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        acc_fn=acc_fn,
    )

    return dice_score
