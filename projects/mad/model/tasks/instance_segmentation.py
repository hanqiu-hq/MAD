import torch
from .raw_task import TaskProcessor
from detectron2.structures import ROIMasks


class BitMaskProcessor(TaskProcessor):
    def __init__(
            self,
            *,
            num_bins: int = 500,
            mask_size: tuple = (16, 16),
            use_skip_vocab: bool = False,
            mask_threshold: float = 0.5,
            **kwargs,
    ):
        kwargs = self.rewrite_param_default(**kwargs)
        vocab_dict = kwargs["vocab_dict"]
        prompt_length = kwargs["prompt_length"]

        kwargs["max_seq_length"] = int(mask_size[0] * mask_size[1]) + prompt_length

        # build vocab mask to ignore the loss computation on specific tokens for efficiency
        self.task_vacab_endidx = vocab_dict["text_range"][0] if use_skip_vocab else vocab_dict["vocab_length"]

        # mask to make sure that the prediction is in the proper format
        infer_mask = torch.zeros(self.task_vacab_endidx)
        infer_mask[vocab_dict["foreground_token"]] = 1
        infer_mask[vocab_dict["background_token"]] = 1
        kwargs["infer_mask"] = infer_mask

        # mask to separate the loss computation for different tasks
        if kwargs.pop("use_vocab_mask", False):
            vocab_mask = torch.ones(self.task_vacab_endidx)
            vocab_mask[vocab_dict["foreground_token"]] = 0
            vocab_mask[vocab_dict["background_token"]] = 0
            vocab_mask = vocab_mask.bool()
        else:
            if "pred_token_shift" in vocab_dict:
                # avoid compute loss for prompt tokens
                vocab_mask = torch.ones(self.task_vacab_endidx)
                vocab_mask[vocab_dict["pred_token_shift"]:] = 0
                vocab_mask = vocab_mask.bool()
            else:
                vocab_mask = None
        kwargs["vocab_mask"] = vocab_mask

        super(BitMaskProcessor, self).__init__(**kwargs)

        self.num_bins = num_bins
        self.mask_size = mask_size
        self.mask_threshold = mask_threshold

    def rewrite_param_default(self, **kwargs):
        kwargs["num_seq_train"] = kwargs.pop("num_seq_train", 10)
        kwargs["num_seq_test"] = kwargs.pop("num_seq_test", 100)
        kwargs["prompt_length"] = kwargs.pop("prompt_length", 6)
        return kwargs

    def process_training_input(self, batch_size, targets=None):
        assert targets is not None, "for instance segmentation, the input must contain box"
        mask_seq_size = self.mask_size[0] * self.mask_size[1]

        batched_prompt_seq = []
        batched_target_seq = []
        batched_valid_mask = []
        for targets_per_img in targets:
            gt_classes = targets_per_img.gt_classes
            gt_boxes = targets_per_img.norm_gt_boxes

            if gt_classes.shape[0] == 0:
                prompt_seq = torch.zeros([1, self.num_seq_train, self.prompt_length]).to(self.device)
                target_seq = torch.full(
                    (self.num_seq_train, mask_seq_size),
                    self.vocab_dict["background_token"],
                    device=self.device, dtype=torch.long)
                valid_mask = torch.zeros(self.num_seq_train, mask_seq_size, device=self.device)
            else:
                prompt_seq, indices = self.build_batched_prompt_seq(
                    1, gt_boxes.unsqueeze(0), gt_classes.unsqueeze(0))
                indices = indices.squeeze(0)
                gt_mask = targets_per_img.gt_masks.crop_and_resize(
                    targets_per_img.gt_boxes.tensor, self.mask_size[0]).to(self.device)
                target_seq = (gt_mask * self.vocab_dict["foreground_token"] +
                              ~gt_mask * self.vocab_dict["background_token"]).long()
                target_seq = target_seq[indices].flatten(1, 2)
                valid_mask = indices.gt(-1).float().unsqueeze(1).repeat(1, mask_seq_size)

            batched_prompt_seq.append(prompt_seq)
            batched_target_seq.append(target_seq)
            batched_valid_mask.append(valid_mask)

        batched_prompt_seq = torch.cat(batched_prompt_seq, dim=0).long()
        batched_target_seq = torch.stack(batched_target_seq, dim=0).long()
        batched_valid_mask = torch.stack(batched_valid_mask, dim=0)

        all_input_seq = []
        all_target_seq = []
        all_seq_mask = []

        # for masked training
        for mr in self.train_mask_ratio:
            noise = torch.rand(batched_target_seq.shape).to(self.device)
            input_seq, seq_mask = self.apply_masking(batched_target_seq.clone(), noise, mr)
            seq_mask = seq_mask * batched_valid_mask
            input_seq = torch.cat([batched_prompt_seq, input_seq], dim=2).long()

            all_input_seq.append(input_seq)
            all_target_seq.append(batched_target_seq)
            all_seq_mask.append(seq_mask)

        all_input_seq = torch.cat(all_input_seq, dim=1)
        all_target_seq = torch.cat(all_target_seq, dim=1)
        all_seq_mask = torch.cat(all_seq_mask, dim=1)

        return all_input_seq, all_target_seq, all_seq_mask

    def process_inference(self, result_dict, pred_logits, batched_img_sizes):
        """
        :param pred_logits: [bs, num_sequence, sequence_length, vocab_size]
        :return:
        """
        assert "pred_boxes" in result_dict, "model should first detect boxes"

        bs = pred_logits.shape[0]
        if self.vocab_mask is not None:
            pred_logits = pred_logits.masked_fill(self.vocab_mask, float('-inf'))
        pred_prob = pred_logits.softmax(dim=3)
        pred_bitmasks = pred_prob[:, :, :, self.vocab_dict["foreground_token"]]
        pred_bitmasks = pred_bitmasks.reshape(bs, self.num_seq_test, self.mask_size[0], self.mask_size[1])

        pred_box_list = result_dict["pred_boxes"]
        pred_masks_list = []
        for bs_i, (masks_per_img, box_per_img, img_size) in \
                enumerate(zip(pred_bitmasks, pred_box_list, batched_img_sizes)):
            roi_masks = ROIMasks(masks_per_img)
            masks_per_img = roi_masks.to_bitmasks(
                box_per_img, img_size[0], img_size[1], self.mask_threshold).tensor
            pred_masks_list.append(masks_per_img)

        result_dict["pred_masks"] = pred_masks_list

        return result_dict

    def build_batched_prompt_seq(self, batch_size, boxes, classes):
        """
        :param
            batch_size: unused parameter introduced to match the original function
            boxes: [bs, num_boxes, 4] in (0 - 1)
            classes: [bs, num_boxes,]
        :return:
            out_prompts: [bs, num_instance_per_img, 5]
            instance_weight: [bs, num_instance_per_img] for loss computation
            indices: [bs, num_instance_per_img] indices of target for loss if the targets are randomly shuffled
        """
        bs, num_boxes, _ = boxes.shape
        assert batch_size == bs
        # first concat box and class token index
        classes = classes.unsqueeze(2)
        box_tokens = (boxes * self.num_bins).floor().clamp(min=0, max=self.num_bins)
        task_seq = self.task_seq.reshape(1, 1, -1).repeat(bs, num_boxes, 1)
        prompt_seq = torch.cat([
            task_seq,
            box_tokens + self.vocab_dict["coord_range"][0],
            classes + self.vocab_dict["class_range"][0]
        ], dim=2)

        if not self.training:
            return prompt_seq

        # pad or select instance based on max_instance_per_img_train/test
        assert bs == 1, "the prompt is processed one by one during training"
        num_sequence = self.num_seq_train
        if num_boxes < num_sequence:
            padded_prompt_seq = torch.zeros(
                (bs, num_sequence - num_boxes, self.prompt_length), device=self.device)
            prompt_seq = torch.cat([prompt_seq, padded_prompt_seq], dim=1)
            indices = torch.arange(num_sequence).long().to(self.device)
            indices[num_boxes:] = -1
            indices = indices.unsqueeze(0).repeat(bs, 1)
        else:
            box_scores = torch.rand(bs, num_boxes).to(self.device)
            indices = box_scores.argsort(dim=1)[:, :num_sequence]
            prompt_seq = torch.gather(
                prompt_seq, dim=1, index=indices.unsqueeze(2).repeat(1, 1, self.prompt_length))

        return prompt_seq, indices
