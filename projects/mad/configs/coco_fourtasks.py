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

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="coco_2017_train_fourtasks"),
    mapper=L(DatasetMapper_KeyPoint)(
        augmentations=[
            L(T.RandomFlip)(),
            L(T.RandomApply)(
                tfm_or_aug=L(T.AugmentationList)(
                    augs=[
                        L(T.ResizeShortestEdge)(
                            short_edge_length=(400, 500, 600),
                            sample_style="choice",
                        ),
                        L(T.RandomCrop)(
                            crop_type="absolute_range",
                            crop_size=(384, 600),
                        ),
                    ]),
                prob=0.5,
            ),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
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
            L(T.ResizeShortestEdge)(
                short_edge_length=800,
                max_size=1333,
            ),
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
