import json
from collections import defaultdict
from pathlib import Path
from sys import stderr

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from tqdm import tqdm

from utils.image import rescale
from utils.misc import seeded_shuffle

try:
    import orjson
    def load_json(file_path):
        with open(file_path, 'rb') as f:
            return orjson.loads(f.read())
except ImportError:
    try:
        import ujson
        load_json = lambda fp: ujson.load(open(fp))
    except ImportError:
        load_json = lambda fp: json.load(open(fp))


class NFS(Dataset):
    """
    A loader for the NFS (Need for Speed) dataset in COCO format.
    """

    def __init__(
        self,
        location,
        split="coco_annotations_eval",
        shuffle=True,
        shuffle_seed=42,
        frame_transform=None,
        annotation_transform=None,
        combined_transform=None,
    ):
        """
        Initializes the loader.

        :param location: Directory containing the dataset (e.g., data/nfs)
        :param split: Either "coco_annotations_eval" or "coco_annotations_train"
        :param shuffle: Whether to shuffle videos.
        :param shuffle_seed: The seed to use if shuffling.
        :param frame_transform: A callable to be applied to each frame
        as it is loaded. Passed to NFSItem constructor.
        :param annotation_transform: A callable to be applied to each
        bounding-box annotation as it is loaded. Passed to NFSItem
        constructor.
        :param combined_transform: A callable to be applied to each
        (frame, annotation) tuple as it is loaded. Passed to NFSItem
        constructor.
        """
        self.location = Path(location)
        self.frame_transform = frame_transform
        self.annotation_transform = annotation_transform
        self.combined_transform = combined_transform

        # Load annotations from JSON
        print(f"Loading NFS annotations from {split}.json...", file=stderr, flush=True)
        annotations_path = self.location / f"{split}.json"
        json_data = load_json(annotations_path)

        # Load information about each video in the dataset.
        self.video_info = self._get_videos_info(json_data)

        # Optionally shuffle the videos (by default they are sorted).
        if shuffle:
            seeded_shuffle(self.video_info, shuffle_seed)
        
        print(f"Loaded {len(self.video_info)} videos from {split}.", file=stderr, flush=True)

    def __getitem__(self, index):
        """
        Loads and returns an item from the dataset.

        :param index: The index of the item to load
        :return: An NFSItem object.
        """
        video_info = self.video_info[index]
        # JSON中的file_name已经是相对于location的完整路径
        frame_paths = [str(self.location / frame["file_name"]) for frame in video_info["frames"]]
        annotations = [frame["annotations"] for frame in video_info["frames"]]
        nfs_item = NFSItem(
            frame_paths,
            annotations,
            self.frame_transform,
            self.annotation_transform,
            self.combined_transform,
        )
        return nfs_item

    def __len__(self):
        """
        Returns the number of items in the dataset.
        """
        return len(self.video_info)

    @staticmethod
    def _get_videos_info(json_data):
        """
        Organizes frames by video from COCO format JSON.
        Uses optimized data structures for fast processing.

        :param json_data: Loaded JSON data from COCO annotations file
        :return: List of video info dictionaries
        """
        print("Building frame dictionary...", file=stderr, flush=True)
        
        # Place frames in a dictionary with their ID as the key.
        frame_dict = {}
        images = json_data["images"]
        for item in tqdm(images, desc="Processing images", ncols=80):
            frame_dict[item["id"]] = {
                "video": item.get("video", "unknown"),
                "frame_id": item.get("frame_id", item["id"]),
                "file_name": item["file_name"],
                "boxes": [],
                "labels": [],
            }

        print("Assigning annotations...", file=stderr, flush=True)
        
        # Create image_id to frame mapping for O(1) lookup
        annotations = json_data.get("annotations", [])
        for item in tqdm(annotations, desc="Processing annotations", ncols=80):
            image_id = item["image_id"]
            if image_id in frame_dict:
                frame = frame_dict[image_id]
                # Convert from xywh to xyxy format
                x, y, w, h = item["bbox"]
                frame["boxes"].append([x, y, x + w, y + h])
                # Use category_id (assuming 1-based, convert to 0-based)
                frame["labels"].append(item.get("category_id", 0) - 1)

        print("Organizing frames by video...", file=stderr, flush=True)
        
        # Convert annotations to tensors and organize frames by video.
        video_dict = defaultdict(list)
        for frame in tqdm(frame_dict.values(), desc="Organizing videos", ncols=80):
            # Convert to tensors
            frame["annotations"] = {
                "boxes": torch.tensor(frame.pop("boxes"), dtype=torch.float32),
                "labels": torch.tensor(frame.pop("labels"), dtype=torch.int64),
            }
            video = frame.pop("video")
            video_dict[video].append(frame)

        print("Sorting frames within videos...", file=stderr, flush=True)
        
        # Build videos_info list with sorted frames
        videos_info = []
        for video_id in tqdm(sorted(video_dict.keys()), desc="Finalizing videos", ncols=80):
            video = video_dict[video_id]
            video.sort(key=lambda v: v["frame_id"])
            videos_info.append({"video_id": video_id, "frames": video})

        return videos_info


class NFSItem(Dataset):
    """
    A Dataset subclass for iterating over a single NFS item (video).
    Necessary due to the very long length of some videos (loading into 
    a single tensor would exhaust memory).
    """

    def __init__(
        self,
        frame_paths,
        annotations,
        frame_transform,
        annotation_transform,
        combined_transform,
    ):
        """
        Initializes the item.

        :param frame_paths: A list of frame paths for this item
        :param annotations: A list of annotations for this item
        :param frame_transform: A callable to be applied to each frame
        as it is loaded.
        :param annotation_transform: A callable to be applied to each
        bounding-box annotation as it is loaded.
        :param combined_transform: A callable to be applied to each
        (frame, annotation) tuple as it is loaded.
        """
        self.frame_paths = frame_paths
        self.annotations = annotations
        self.frame_transform = frame_transform
        self.annotation_transform = annotation_transform
        self.combined_transform = combined_transform

    def __getitem__(self, index):
        """
        Loads and returns a frame and the corresponding labels.

        :param index: The frame index
        :return: A (frame, annotations) tuple
        """
        frame = read_image(self.frame_paths[index])
        if self.frame_transform is not None:
            frame = self.frame_transform(frame)
        
        annotations = self.annotations[index]
        
        # Ensure annotations have correct shape even if empty
        if annotations["boxes"].numel() == 0:
            # Create [[]] shaped tensors for consistency
            annotations["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
        if annotations["labels"].numel() == 0:
            annotations["labels"] = torch.zeros((0,), dtype=torch.int64)
        
        if self.annotation_transform is not None:
            annotations = self.annotation_transform(annotations)
        if self.combined_transform is not None:
            return self.combined_transform((frame, annotations))
        else:
            return frame, annotations

    def __len__(self):
        """
        Returns the number of frames in the item.
        """
        return len(self.frame_paths)


class NFSResize(torch.nn.Module):
    """
    A PyTorch module for simultaneously resizing frames and annotations.
    Should be passed to NFS.__init__ as a combined_transform.
    """

    def __init__(self, short_edge_length, max_size):
        """
        :param short_edge_length: The size to which the short edge
        should be resized
        :param max_size: The maximum size of the long edge (this
        overrides short_edge_length if there is a conflict)
        """
        super().__init__()
        self.short_edge_length = short_edge_length
        self.max_size = max_size

    def forward(self, x):
        frame, annotations = x
        short_edge = min(frame.shape[-2:])
        long_edge = max(frame.shape[-2:])
        scale = min(self.short_edge_length / short_edge, self.max_size / long_edge)
        frame = rescale(frame, scale)
        annotations = {
            "boxes": annotations["boxes"] * scale,
            "labels": annotations["labels"],
        }
        return frame, annotations