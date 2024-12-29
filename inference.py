import logger
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from torch.utils.data import DataLoader
from tqdm import tqdm


def inference(
        model: nn.Module,
        device,
        ckpt_filepath: str,
        loader: DataLoader,
        roi: tuple[int, int, int],
        sw_batch_size: int,
        overlap: float,
) -> torch.Tensor:
    model.to(device=device)

    if os.path.exists(path=ckpt_filepath):
        ckpt = torch.load(f=ckpt_filepath, map_location=device)
        model.load_state_dict(state_dict=ckpt["model_state"])
        filename = os.path.basename(ckpt_filepath)
        logger.log_info(f"Loaded '{filename}'")
    else:
        logger.log_error(f"Cannot find ckpt file at '{ckpt_filepath}'.")
        exit(1)

    model.eval()

    loader_tqdm = tqdm(iterable=loader, position=1)
    loader_tqdm.set_description(desc=f"[Batch 0]", refresh=True)

    segs = []

    with torch.no_grad():
        for idx, batch in enumerate(iterable=loader_tqdm):
            data = batch["image"].to(device=device)
            logits = sliding_window_inference(
                inputs=data,
                roi_size=roi,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=overlap,
            )
            pred = (torch.sigmoid(input=logits) >= 0.5)
            pred = pred[0].detach().cpu().to(torch.int8)

            seg = torch.zeros(size=pred.shape[1:], dtype=torch.int8)
            seg[pred[1] == 1] = 2
            seg[pred[0] == 1] = 1
            seg[pred[2] == 1] = 3
            segs.append(seg)

            loader_tqdm.set_description(
                desc=f"[Batch {idx+1}]",
                refresh=True
            )

    x = max(seg.shape[0] for seg in segs)
    y = max(seg.shape[1] for seg in segs)
    z = max(seg.shape[2] for seg in segs)

    segs_pad = []
    for seg in segs:
        x_pad = x - seg.shape[0]
        y_pad = y - seg.shape[1]
        z_pad = z - seg.shape[2]

        xl = x_pad // 2
        xr = x_pad - xl
        yt = y_pad // 2
        yb = y_pad - yt
        zf = z_pad // 2
        zb = z_pad -zf

        seg = F.pad(input=seg, pad=(zf, zb, yt, yb, xl, xr))
        segs_pad.append(seg)

    segs_pad = torch.stack(tensors=segs_pad, dim=0)

    return segs_pad
