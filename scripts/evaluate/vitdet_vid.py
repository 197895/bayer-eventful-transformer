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
from datasets.vid import VIDResize, VID
from models.vitdet import ViTDet
from utils.config import initialize_run
from utils.evaluate import run_evaluations
from utils.misc import dict_to_device, squeeze_dict
from utils.unprocess_np import _significant_tokens_stats

def evaluate_vitdet_metrics(device, model, data, config):
    model.counting()
    model.clear_counts()
    n_frames = 0
    outputs = []
    labels = []
    n_items = config.get("n_items", len(data))
    for j, vid_item in tqdm(zip(range(n_items), data), total=n_items, ncols=0):
        # if j<=-1:
        #     print("debug skip")
        #     continue
        if j>=4:
            print("debug break")
            break
        vid_item = DataLoader(vid_item, batch_size=1)
        n_frames += len(vid_item)
        model.reset()
        for frame, annotations in tqdm(vid_item, ncols=0):
            with torch.inference_mode():
                outputs.extend(model(frame.to(device)))
            labels.append(squeeze_dict(dict_to_device(annotations, device), dim=0))

    # MeanAveragePrecision is extremely slow. It seems fastest to call
    # update() and compute() just once, after all predictions are done.
    mean_ap = MeanAveragePrecision()
    mean_ap.update(outputs, labels)
    metrics = mean_ap.compute()
    print('metrics:', metrics)
    _significant_tokens_stats.plot_statistics()
    counts = model.total_counts() / n_frames
    model.clear_counts()
    return {"metrics": metrics, "counts": counts}


def main():
    config = initialize_run(config_location=Path("configs", "evaluate", "vitdet_vid"))
    long_edge = max(config["model"]["input_shape"][-2:])
    data = VID(
        Path("../datasets", "vid"),
        split=config["split"],
        tar_path=Path("../datasets", "vid", "data.tar"),
        combined_transform=VIDResize(
            short_edge_length=640 * long_edge // 1024, max_size=long_edge
        ),
    )
    run_evaluations(config, ViTDet, data, evaluate_vitdet_metrics)


if __name__ == "__main__":
    main()
