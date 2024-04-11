import config
import numpy as np
import torch
import torch.nn as nn
from avg_meter import AverageMeter
from data import *
from monai.inferers import sliding_window_inference
from tqdm import tqdm


def val_epoch(
        model: nn.Module,
        val_loader: DataLoader,
        roi: tuple[int, int, int],
        sw_batch_size: int,
        overlap: float,
        acc_fn,
) -> float:
    model.eval()

    loader_tqdm = tqdm(iterable=val_loader, position=1)
    loader_tqdm.set_description(
        desc=f"[Batch 0]",
        refresh=True
    )

    acc_meter = AverageMeter()

    with torch.no_grad():
        for idx, batch in enumerate(iterable=loader_tqdm):
            data = batch["image"].to(device=config.DEVICE)
            target = batch["label"].to(device=config.DEVICE)
            logits_batch = sliding_window_inference(
                inputs=data,
                roi_size=roi,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=overlap,
            )
            # Note: if batch size > 1, need decollect_batch
            preds = [
                (torch.sigmoid(input=logits) >= 0.5).to(torch.float32)
                for logits in logits_batch
            ]
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