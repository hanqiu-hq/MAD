from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.structures import ImageList, Instances
from detrex.utils import get_world_size, is_dist_avail_and_initialized


class MAD(nn.Module):
    """

    Args:
        backbone (nn.Module): Backbone module for feature extraction.
        in_features (List[str]): Selected backbone output features for transformer module.
        in_channels (int): Dimension of the last feature in `in_features`.
        position_embedding (nn.Module): Position encoding layer for generating position embeddings.
        transformer (nn.Module): Transformer module used for further processing features
            and input queries.
        embed_dim (int): Hidden dimension for transformer module.
        seq_pos_length (int): maximum length of prediction sequences.
        task_dict (dict): dictionary contains task names and their loss weights.
        task_processor (Module List): list of modules to process each task.
        vocab_embed (nn.Module): prediction module
        infer_list: list for task names to be evaluated.
        pixel_mean (List[float]): Pixel mean value for image normalization.
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        device (str): Training device. Default: "cuda".
    """

    def __init__(
        self,
        backbone: nn.Module,
        in_features: List[str],
        in_channels: int,
        position_embedding: nn.Module,
        transformer: nn.Module,
        embed_dim: int,
        seq_pos_length: int,
        task_dict: dict,
        task_processor: nn.ModuleDict,
        vocab_embed: nn.Module,
        infer_list: list = None,
        pixel_mean: List[float] = [123.675, 116.280, 103.530],
        pixel_std: List[float] = [58.395, 57.120, 57.375],
        device: str = "cuda",
    ):
        super().__init__()
        # define backbone and position embedding module
        self.backbone = backbone
        self.in_features = in_features
        self.position_embedding = position_embedding

        # project the backbone output feature into the required dim for transformer block
        self.input_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

        # define learnable object queries and transformer module
        self.transformer = transformer

        # initialize position embedding for input sequence for all tasks
        self.seq_pos_embed = nn.Parameter(torch.rand(seq_pos_length, embed_dim))
        nn.init.normal_(self.seq_pos_embed)

        self.vocab_embed = vocab_embed
        self.task_dict = task_dict
        if infer_list is not None:
            self.infer_list = infer_list
        else:
            self.infer_list = list(task_dict.keys())
        self.task_processor = task_processor

        self.target_keys = {
            "det": "gt_instances",
            "seg": "gt_instances",
            "keypoint": "gt_instances",
            "caption": "gt_caption_tokens",
        }

        assert all([key in self.task_processor for key in self.task_dict.keys()]) and \
               all([key in self.task_dict for key in self.task_processor.keys()]), \
            "loss dict and processor dict must match with each other."

        # normalizer for input raw images
        self.device = torch.device(device)
        pixel_mean = torch.Tensor(pixel_mean).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(pixel_std).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs (List[dict]): A list of instance dict, and each dict must consists of:
                - dict["image"] (torch.Tensor): The unnormalized image tensor.
                - dict["height"] (int): The original image height.
                - dict["width"] (int): The original image width.
                - dict["instance"] (detectron2.structures.Instances):
                    Image meta informations and ground truth boxes and labels during training.
                    Please refer to
                    https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.Instances
                    for the basic usage of Instances.
                - dict["captions"] (list): list of the captioning sentence candidates.

        Returns:
            During training: Returns a dict with the following elements:
                dict["loss_xxx(task)"]: contains loss for each involved task.
            During Inference: Returns a list of predictions for batched images.
                    The list comprises dict with following elements.
                dict["instances"]: predictions for object detection, instance segmentation, and keypoint detection,
                    which are organized with instance class.
                dict["caption"]: predicted captioning sentence.
        """
        images, img_masks, origin_img_sizes = self.preprocess_image(batched_inputs)

        features = self.backbone(images.tensor)[self.in_features[-1]]
        features = self.input_proj(features)
        img_masks = F.interpolate(img_masks[None], size=features.shape[-2:]).bool()[0]
        pos_embed = self.position_embedding(img_masks)

        # first encode the image feature
        features = self.transformer.encode(features, pos_embed, img_masks)

        if self.training:
            targets = self.preprocess_target(batched_inputs)
            loss_dict = dict()
            for task, task_weight in self.task_dict.items():
                loss_tuple = self.forward_training(task, features, img_masks, pos_embed, targets)
                loss_dict["loss_" + task] = task_weight * self.compute_loss_for_seq(*loss_tuple)
            return loss_dict
        else:
            pred_dict = {}
            for task in self.infer_list:
                pred_dict = self.forward_inference(
                    task, features, img_masks, pos_embed, pred_dict, origin_img_sizes)
            results = self.inference(pred_dict, origin_img_sizes)
            return results

    def forward_training(self, task, features, img_masks, pos_embed, target_dict):
        """
        Args:
            task: task name in "det", "seg", "keypoint", and "caption"
            features: image feature map with shape [H * W, batch size, dim]
            img_masks: image padding mask with shape [batch size, H * W]. '1' for padding pixels and
                        '0' for unpadded pixels.
            pos_embed: positional encoding for image feature with shape [H * W, batch size, dim]
            target_dict: dict contains targets

        Returns:
            loss dict: value for "loss_det", "loss_seg", "loss_keypoint", "loss_caption".
        """
        targets = target_dict[self.target_keys[task]]
        input_seq, target_seq, seq_mask = self.task_processor[task].process_training_input(
            features.shape[0], targets)
        seq_pos_embed = self.seq_pos_embed[:input_seq.shape[2]]
        decoded_embeds = self.transformer.decode(
            features, img_masks, pos_embed, self.vocab_embed.embeddings(input_seq), seq_pos_embed)
        seq_logits = self.vocab_embed(decoded_embeds, self.task_processor[task].task_vacab_endidx)
        loss_tuple = self.task_processor[task].process_training_triplet(
            seq_logits, target_seq, seq_mask, targets)
        return loss_tuple

    def forward_inference(self, task, features, img_masks, pos_embed, pred_dict, img_sizes):
        """
        Args:
            task: task name in "det", "seg", "keypoint", and "caption"
            features: image feature map with shape [H * W, batch size, dim]
            img_masks: image padding mask with shape [batch size, H * W]. '1' for padding pixels and
                        '0' for unpadded pixels.
            pos_embed: positional encoding for image feature with shape [H * W, batch size, dim]
            target_dict: dict contains targets for inferring with ground truth.
            pred_dict:
            img_sizes:

        Returns:
            pred_dict:
        """
        bs = features.shape[0]
        last_pred_logits = seq_mask = seq_logits = None

        for i in range(self.task_processor[task].num_infer_stage):
            masked_input_seq, seq_mask, last_pred_logits = \
                self.task_processor[task].process_inference_input(
                    bs, i, pred_dict, last_pred_logits, seq_mask, seq_logits)
            seq_pos_embed = self.seq_pos_embed[:masked_input_seq.shape[2]]
            decoded_embeds = self.transformer.decode(
                features, img_masks, pos_embed,
                self.vocab_embed.embeddings(masked_input_seq), seq_pos_embed)
            seq_logits = self.vocab_embed(decoded_embeds[[-1]], self.task_processor[task].task_vacab_endidx).squeeze(0)

        _, _, last_pred_logits = self.task_processor[task].process_inference_input(
            bs, -1, pred_dict, last_pred_logits, seq_mask, seq_logits)

        pred_dict = self.task_processor[task].process_inference(pred_dict, last_pred_logits, img_sizes)
        return pred_dict

    def compute_loss_for_seq(self, pred_logits, target_seq, target_weight):
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        loss_div = target_weight.sum().float().clone()
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(loss_div)
        loss_div = torch.clamp(loss_div / get_world_size(), min=1).item()

        loss = F.cross_entropy(pred_logits, target_seq, reduction='none')
        loss = (loss * target_weight).sum() / loss_div
        return loss

    def inference(self, pred_dict, origin_image_size):
        """Inference function for MAD

        Args:
            pred_dict (dict contains instance attribution predicted from each task)
            origin_image_size: original sizes of input images

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        batched_results = []
        valid_instance_keys = ["pred_boxes", "scores", "pred_classes", "pred_masks", "pred_keypoints"]
        for batch_idx, image_size in enumerate(origin_image_size):
            pred_instances = Instances(image_size)
            for k, v in pred_dict.items():
                if k in valid_instance_keys:
                    pred_instances.set(k, v[batch_idx])

            result_dict = dict()
            if pred_instances.has("pred_boxes"):
                pred_instances = pred_instances[pred_instances.pred_boxes.nonempty()]
                result_dict["instances"] = pred_instances

            if "pred_captions" in pred_dict:
                result_dict["captions"] = pred_dict["pred_captions"][batch_idx]

            batched_results.append(result_dict)

        return batched_results

    def preprocess_image(self, batched_inputs):
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )

        origin_image_size = []
        for input_per_image, image_size in zip(batched_inputs, images.image_sizes):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            origin_image_size.append((height, width))

        batch_size, _, H, W = images.tensor.shape
        img_masks = images.tensor.new_ones(batch_size, H, W)
        for img_id in range(batch_size):
            img_h, img_w = images.image_sizes[img_id]
            img_masks[img_id, :img_h, :img_w] = 0

        return images, img_masks, origin_image_size

    def preprocess_target(self, batched_inputs):
        targets = dict()
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            for img_i, targets_per_img in enumerate(gt_instances):
                h, w = targets_per_img.image_size
                image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
                gt_boxes = targets_per_img.gt_boxes.tensor / image_size_xyxy
                gt_instances[img_i].norm_gt_boxes = gt_boxes
            targets["gt_instances"] = gt_instances

        if "captions" in batched_inputs[0] and "caption" in self.task_processor:
            gt_captions = [x["captions"] for x in batched_inputs]
            gt_caption_tokens = []
            max_seq_len = self.task_processor["caption"].tokenizer.max_seq_len
            pad_idx = self.task_processor["caption"].tokenizer.padding_idx
            for img_i, caption_per_img in enumerate(gt_captions):
                lower_caption_per_img = [cap.lower() for cap in caption_per_img]
                token_ids = self.task_processor["caption"].tokenizer(lower_caption_per_img)
                seq_length = token_ids.shape[1]
                if seq_length < max_seq_len:
                    token_ids = F.pad(token_ids, (0, max_seq_len - seq_length), value=pad_idx)
                else:
                    token_ids = token_ids[:, :max_seq_len]
                gt_caption_tokens.append(token_ids.long().to(self.device))
            targets["gt_caption_tokens"] = gt_caption_tokens
        return targets