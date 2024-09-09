from typing import MutableSequence
from detectron2.config import LazyCall as L

from projects.mad.configs.mad_r50_300ep_32bs import optimizer, lr_multiplier, train
from projects.mad.configs.coco_fourtasks import dataloader

from detrex.layers.position_embedding import PositionEmbeddingSine

from projects.detr.modeling.transformer import DetrTransformerEncoder
from .transformer import (
    DetrTransformer,
    DetrTransformerDecoder,
)
from detrex.layers import MLP

from .mad import MAD
from .tasks.object_detection import DetectionProcessor
from .tasks.instance_segmentation import BitMaskProcessor
from .tasks.keypoint import KeypointProcessor
from .tasks.caption import CaptionProcessor
from .utils import KNNClassifier, List2ModuleDict

from torchtext.models import T5Transform

from detectron2.modeling.backbone import SwinTransformer


num_classes = 80
num_bins = 500
pred_token_shift = 20
num_vocab_word = 11421
text_pad_idx = 0
text_eos_idx = 0


# A shared vocab among tasks and its structure
vocab_dict = {
    # Tasks id
    "det_token": [0, 1, 2, 3, 4, 5],
    "seg_token": 6,
    "keypoint_token": 7,
    "caption_token": [8, 9, 10, 11, 12, 13],
    # For specific tokens
    "mask_token": 19,

    # for object detection
    "class_range": [pred_token_shift, pred_token_shift + num_classes],
    "fake_class_token": pred_token_shift + num_classes,
    "coord_range": [pred_token_shift + num_classes + 1,
                    pred_token_shift + num_classes + num_bins + 2],
    # for segmentation
    "foreground_token": pred_token_shift + num_classes + num_bins + 2,
    "background_token": pred_token_shift + num_classes + num_bins + 3,
    # for keypoint
    "visible_token": pred_token_shift + num_classes + num_bins + 4,
    "invisible_token": pred_token_shift + num_classes + num_bins + 5,
    # for caption, note that the pad token and eos token are already contained in text range,
    # so the overall vocab_length won't change.
    "text_range": [pred_token_shift + num_classes + num_bins + 6,
                   pred_token_shift + num_classes + num_bins + num_vocab_word + 6],
    "text_pad_token": pred_token_shift + num_classes + num_bins + 6 + text_pad_idx,
    "text_eos_token": pred_token_shift + num_classes + num_bins + 6 + text_eos_idx,

    # first ten special tokens, then class tokens, final coord tokens
    "vocab_length": pred_token_shift + num_classes + num_bins + num_vocab_word + 6,
}


model = L(MAD)(
    backbone=L(SwinTransformer)(
        pretrain_img_size=384,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        window_size=12,
        out_indices=(3,),
    ),
    in_features=["p3"],
    in_channels=1024,
    position_embedding=L(PositionEmbeddingSine)(
        num_pos_feats=128,
        temperature=10000,
        normalize=True,
    ),
    transformer=L(DetrTransformer)(
        encoder=L(DetrTransformerEncoder)(
            embed_dim=256,
            num_heads=8,
            attn_dropout=0.1,
            feedforward_dim=2048,
            ffn_dropout=0.1,
            num_layers=6,
            post_norm=False,
        ),
        decoder=L(DetrTransformerDecoder)(
            embed_dim=256,
            num_heads=8,
            attn_dropout=0.1,
            feedforward_dim=2048,
            ffn_dropout=0.1,
            num_layers=6,
            return_intermediate=True,
            post_norm=True,
        ),
    ),
    embed_dim=256,
    seq_pos_length=506,
    task_dict={
        # because we average the loss for the aux decoder output,
        # so here adopt a weight of num_decoder
        "det": 1.5 * 6,
        "seg": 2.7 * 6,
        "keypoint": 0.5 * 6,
        "caption": 0.3 * 6,
    },
    task_processor=L(List2ModuleDict)(
        module_name=["det", "seg", "keypoint", "caption"],
        module_list=[
            L(DetectionProcessor)(
                vocab_dict=vocab_dict,
                task_seq=vocab_dict["det_token"],
                train_mask_ratio=(1.0, 0.7,),
                test_mask_ratio=(1.0,),
                use_vocab_mask=True,
                num_bins=num_bins,
                num_queries=100,
                use_skip_vocab=True,
            ),
            L(BitMaskProcessor)(
                vocab_dict=vocab_dict,
                task_seq=vocab_dict["seg_token"],
                train_mask_ratio=(1.0, 0.7),
                test_mask_ratio=(1.0,),
                use_rand_infer_mask=False,
                use_vocab_mask=True,
                num_bins=num_bins,
                mask_size=(16, 16),
                use_skip_vocab=True,
                mask_threshold=0.5,
            ),
            L(KeypointProcessor)(
                vocab_dict=vocab_dict,
                task_seq=vocab_dict["keypoint_token"],
                train_mask_ratio=(1.0, 0.7),
                test_mask_ratio=(1.0,),
                use_vocab_mask=True,
                num_bins=num_bins,
                num_keypoints=17,
                use_skip_vocab=True,
                use_rand_coord=True,
            ),
            L(CaptionProcessor)(
                vocab_dict=vocab_dict,
                task_seq=vocab_dict["caption_token"],
                train_mask_ratio=(1.0, 0.7),
                test_mask_ratio=(1.0, 0.8, 0.6, 0.4),
                use_vocab_mask=True,
                tokenizer=L(T5Transform)(
                    sp_model_path='./project/mad/model/t5_tokenizer_coco_caption.model',
                    max_seq_len="${..max_sentence_length}",
                    eos_idx=text_eos_idx,
                    padding_idx=text_pad_idx,
                ),
                max_sentence_length=20,
                num_target_per_img=5,
                use_noised_train=True,
            ),
        ],
    ),
    vocab_embed=L(KNNClassifier)(
        num_embeddings=vocab_dict["vocab_length"],
        embed_dim="${..embed_dim}",
        predictor=L(MLP)(
            input_dim="${..embed_dim}",
            hidden_dim="${..embed_dim}",
            output_dim="${..embed_dim}",
            num_layers=2,
        ),
        amp_infer=True,
    ),
)

model.device = train.device

train.output_dir = "./outputs/mad_swinbase_300ep"
if isinstance(dataloader.evaluator, MutableSequence):
    for i in range(len(dataloader.evaluator)):
        dataloader.evaluator[i].output_dir = train.output_dir
else:
    dataloader.evaluator.output_dir = train.output_dir

train.amp.enabled = True
train.ddp.fp16_compression = True
train.init_checkpoint = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22kto1k.pth?matching_heuristics=True"

dataloader.train.total_batch_size = 32