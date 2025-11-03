#!/usr/bin/env python3

from pathlib import Path

import torch
from torch.utils.data import DataLoader
try:
    from torchmetrics.detection import MeanAveragePrecision
except ImportError:
    from torchmetrics.detection import MAP
    MeanAveragePrecision = MAP
from tqdm import tqdm
from utils.misc import get_pytorch_device
from datasets.nfs import NFSResize, NFS
from models.vitdet import ViTDet
from utils.config import initialize_run
from utils.evaluate import run_evaluations
from utils.misc import dict_to_device, squeeze_dict
from utils.unprocess_np import _significant_tokens_stats,fast_mask_visulize,visualize_detection,get_significant_tokens
from utils.image import as_float32, pad_to_size
def evaluate_vitdet_metrics(device, model, data, config):
    model.counting()
    model.clear_counts()
    n_frames = 0
    outputs = []
    labels = []
    n_items = config.get("n_items", len(data))
    debug_signal=True
    for j, nfs_item in tqdm(zip(range(n_items), data), total=n_items, ncols=0):
        nfs_item = DataLoader(nfs_item, batch_size=1)
        n_frames += len(nfs_item)
        model.reset()
        # if j>=1:
        #     print("debug break")
        #     break
        for frame, annotations in tqdm(nfs_item, ncols=0):
            with torch.inference_mode():
                results= model(frame.to(device))
                # if debug_signal:
                #     debug_signal=False
                #     visualize_detection(pad_to_size(frame[0], (672,672)), results[0],mask=None,patch_size=(16,16),img_size=(672,672),alpha=0.7)
                outputs.extend(results)
            labels.append(squeeze_dict(dict_to_device(annotations, device), dim=0))

    # MeanAveragePrecision is extremely slow. It seems fastest to call
    # update() and compute() just once, after all predictions are done.
    mean_ap = MeanAveragePrecision()
    # Filter out outputs and labels where labels are empty
    filtered_outputs = []
    filtered_labels = []
    for output, label in zip(outputs, labels):
        if label and any(v.numel() > 0 if isinstance(v, torch.Tensor) else v for v in label.values()):  # Check if label has data
            filtered_outputs.append(output)
            filtered_labels.append(label)
    outputs = filtered_outputs
    labels = filtered_labels
    mean_ap.update(outputs, labels)
    metrics = mean_ap.compute()
    k,tau=get_significant_tokens(None,None,None,None,output_kt=True)
    metrics['k']=k
    metrics['tau']=tau
    print('metrics:', metrics)
    metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}
    pruning_rate=_significant_tokens_stats.plot_statistics()
    metrics['pruning_rate']=pruning_rate
    counts = model.total_counts() / n_frames
    model.clear_counts()
    return {"metrics": metrics, "counts": counts}


def main():
    config = initialize_run(config_location=Path("configs", "evaluate", "vitdet_nfs"))
    long_edge = max(config["model"]["input_shape"][-2:])
    data = NFS(
        Path("../datasets", "nfs"),
        split=config.get("split", "coco_annotations_eval"),
        combined_transform=NFSResize(
            short_edge_length=640 * long_edge // 1024, max_size=long_edge
        ),
    )
    run_evaluations(config, ViTDet, data, evaluate_vitdet_metrics)


if __name__ == "__main__":
    main()
