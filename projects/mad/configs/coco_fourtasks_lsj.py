from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator
from detectron2.data.detection_utils import create_keypoint_hflip_indices

from projects.mad.data.dataset_mappers import DatasetMapper_KeyPoint
from projects.mad.data.evaluation import COCOEvaluatorKeypoint, COCOCapEvaluator


dataloader = OmegaConf.create()

image_size = 1024

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="coco_2017_train_fourtasks"),
    mapper=L(DatasetMapper_KeyPoint)(
        augmentations=[
            L(T.RandomFlip)(horizontal=True),  # flip first
            L(T.ResizeScale)(
                min_scale=0.3, max_scale=2.0, target_height=image_size, target_width=image_size
            ),
            L(T.FixedSizeCrop)(crop_size=(image_size, image_size), pad=False),
        ],
        is_train=True,
        image_format="RGB",
        use_instance_mask=True,
        use_keypoint=True,
        keypoint_hflip_indices=create_keypoint_hflip_indices("keypoints_coco_2017_train"),
        recompute_boxes=True,
    ),
    total_batch_size=16,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="coco_2017_val", filter_empty=False),
    mapper=L(DatasetMapper_KeyPoint)(
        augmentations=[
            L(T.ResizeShortestEdge)(short_edge_length=image_size, max_size=image_size),
        ],
        is_train=False,
        image_format="RGB",
    ),
    num_workers=4,
)

dataloader.evaluator = [
    L(COCOEvaluator)(
        dataset_name="coco_2017_val",
        tasks=("bbox", "segm"),
    ),
    L(COCOEvaluatorKeypoint)(
        dataset_name="keypoints_coco_2017_val",
        tasks=("keypoints",),
    ),
    L(COCOCapEvaluator)(
        dataset_name="captions_coco_2017_val"
    )
]
