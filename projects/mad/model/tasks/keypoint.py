import torch
from .raw_task import TaskProcessor


class KeypointProcessor(TaskProcessor):
    def __init__(
            self,
            *,
            num_bins: int = 500,
            num_keypoints: int = 17,
            use_skip_vocab: bool = False,
            use_rand_coord: bool = False,
            **kwargs,
    ):
        kwargs = self.rewrite_param_default(**kwargs)
        vocab_dict = kwargs["vocab_dict"]
        prompt_length = kwargs["prompt_length"]

        kwargs["max_seq_length"] = num_keypoints * 3 + prompt_length

        # build vocab mask to ignore the loss computation of specific tokens
        self.task_vacab_endidx = vocab_dict["text_range"][0] if use_skip_vocab else vocab_dict["vocab_length"]

        # mask to make sure that the prediction is in the proper format
        infer_mask = torch.zeros(num_keypoints, 3, self.task_vacab_endidx)
        infer_mask[:, :2, vocab_dict["coord_range"][0]: vocab_dict["coord_range"][1]] = 1
        infer_mask[:, 2, vocab_dict["visible_token"]] = 1
        infer_mask[:, 2, vocab_dict["invisible_token"]] = 1
        infer_mask = infer_mask.flatten(0, 1)
        kwargs["infer_mask"] = infer_mask

        # mask to separate the loss computation for different tasks
        if kwargs.pop("use_vocab_mask", False):
            vocab_mask = torch.ones(self.task_vacab_endidx)
            vocab_mask[vocab_dict["coord_range"][0]: vocab_dict["coord_range"][1]] = 0
            vocab_mask[vocab_dict["visible_token"]] = 0
            vocab_mask[vocab_dict["invisible_token"]] = 0
            vocab_mask = vocab_mask.bool()
        else:
            if "pred_token_shift" in vocab_dict:
                vocab_mask = torch.ones(self.task_vacab_endidx)
                vocab_mask[vocab_dict["pred_token_shift"]:] = 0
                vocab_mask = vocab_mask.bool()
            else:
                vocab_mask = None
        kwargs["vocab_mask"] = vocab_mask

        super(KeypointProcessor, self).__init__(**kwargs)

        self.num_bins = num_bins
        self.num_keypoints = num_keypoints
        self.use_invis_rand_coord = use_rand_coord

    def rewrite_param_default(self, **kwargs):
        kwargs["num_seq_train"] = kwargs.pop("num_seq_train", 10)
        kwargs["num_seq_test"] = kwargs.pop("num_seq_test", 100)
        kwargs["prompt_length"] = kwargs.pop("prompt_length", 6)
        return kwargs

    def process_training_input(self, batch_size, targets=None):
        assert targets is not None, "for keypoint detection, the input must contain box"
        keypoint_seq_size = self.num_keypoints * 3

        batched_prompt_seq = []
        batched_target_seq = []
        batched_valid_mask = []
        for targets_per_img in targets:
            gt_classes = targets_per_img.gt_classes
            gt_boxes = targets_per_img.norm_gt_boxes

            if gt_classes.shape[0] == 0:
                prompt_seq = torch.zeros([1, self.num_seq_train, self.prompt_length]).to(self.device)
                target_seq = torch.full(
                    (self.num_seq_train, keypoint_seq_size),
                    self.vocab_dict["invisible_token"],
                    device=self.device, dtype=torch.long)
                valid_mask = torch.zeros(self.num_seq_train, keypoint_seq_size, device=self.device)
            else:
                gt_keypoints = targets_per_img.gt_keypoints.tensor
                valid_flag = gt_keypoints[:, :, 2].gt(0)
                valid_flag_per_inst = valid_flag.any(dim=1)
                prompt_seq, indices = self.build_batched_prompt_seq(
                    1,
                    gt_boxes.unsqueeze(0),
                    gt_classes.unsqueeze(0),
                    valid_flag_per_inst.float().unsqueeze(0))
                indices = indices.squeeze(0)

                h, w = targets_per_img.image_size
                coord_scale = torch.as_tensor([w, h], dtype=torch.float, device=self.device)
                xy = gt_keypoints[:, :, :2] / coord_scale

                if self.use_invis_rand_coord:
                    rand_coord = torch.rand_like(xy)
                    box_xy = gt_boxes[:, None, :2]
                    box_wh = gt_boxes[:, None, 2:] - gt_boxes[:, None, :2]
                    rand_coord[:, :, 0] = box_xy[:, :, 0] + rand_coord[:, :, 0] * box_wh[:, :, 0]
                    rand_coord[:, :, 1] = box_xy[:, :, 1] + rand_coord[:, :, 1] * box_wh[:, :, 1]
                    xy = torch.where(valid_flag.unsqueeze(2).repeat(1, 1, 2), xy, rand_coord)

                xy = (xy * self.num_bins).floor().long().clamp(min=0, max=self.num_bins)
                point_label = (valid_flag * self.vocab_dict["visible_token"] +
                               ~valid_flag * self.vocab_dict["invisible_token"]).long().unsqueeze(2)
                target_seq = torch.cat([xy + self.vocab_dict["coord_range"][0], point_label], dim=2)

                target_weight = torch.zeros_like(target_seq)
                target_weight[:, :, 2] = 1
                target_weight[valid_flag] = 1
                target_weight[~valid_flag_per_inst] = 0

                target_seq = target_seq[indices].flatten(1, 2)
                target_weight = target_weight[indices].flatten(1, 2)
                valid_mask = indices.gt(-1).float().unsqueeze(1).repeat(1, keypoint_seq_size)
                valid_mask = valid_mask * target_weight

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

        bs, num_seq = pred_logits.shape[:2]
        if self.vocab_mask is not None:
            pred_logits = pred_logits.masked_fill(self.vocab_mask, float('-inf'))
        pred_prob = pred_logits.softmax(dim=3) * self.infer_mask + self.infer_mask
        pred_prob = pred_prob.reshape(bs, num_seq, self.num_keypoints, 3, -1)

        pred_seq = pred_prob.argmax(dim=4)
        keypoint_xy = ((pred_seq[:, :, :, :2] - self.vocab_dict["coord_range"][0]) / self.num_bins).clamp(min=0, max=1)
        vis_score = pred_prob[:, :, :, [2], self.vocab_dict["visible_token"]] - 1
        invis_score = pred_prob[:, :, :, [2], self.vocab_dict["invisible_token"]] - 1
        keypoint_score = vis_score / (vis_score + invis_score)
        keypoint_results = torch.cat([keypoint_xy, keypoint_score], dim=3)

        pred_keypoints_list = []
        for bs_i, (keypoint_per_img, img_size) in \
                enumerate(zip(keypoint_results, batched_img_sizes)):
            keypoint_per_img[:, :, 0] *= img_size[1]
            keypoint_per_img[:, :, 1] *= img_size[0]
            pred_keypoints_list.append(keypoint_per_img)

        result_dict["pred_keypoints"] = pred_keypoints_list

        return result_dict

    def process_pred_score(self, score, iter_idx):
        """
        Args:
            score: [batch_size, num_sequence, max_sequence_length - prompt_length]
        Returns:
            score
        """
        bs, num_seq, seq_length = score.shape
        score = score.reshape(bs, num_seq, self.num_keypoints, 3)
        if iter_idx == 1:
            score = score[:, :, :, None, 2].repeat(1, 1, 1, 3)
            score[:, :, :, 2] = 1

        score = score.flatten(2, 3)

        return score

    def build_batched_prompt_seq(self, batch_size, boxes, classes, gt_valid_flag=None):
        """
        :param
            batch_size:
            boxes: [bs, num_boxes, 4] in (0 - 1)
            classes: [bs, num_boxes,]
            gt_valid_flag: [bs, num_boxes]
        :return:
            out_prompts: [bs, num_instance_per_img, 5]
            instance_weight: [bs, num_instance_per_img] for loss computation
            indices: [bs, num_instance_per_img] indices of target for loss if the targets are randomly shuffled
        """
        bs, num_boxes, _ = boxes.shape
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
        num_sequence = self.num_seq_train
        if num_boxes < num_sequence:
            # pad zero boxes to the box list for batched computation
            padded_seq = torch.zeros(
                (bs, num_sequence - num_boxes, self.prompt_length), device=self.device)
            prompt_seq = torch.cat([prompt_seq, padded_seq], dim=1)
            indices = torch.arange(num_sequence).long().to(self.device)
            indices[num_boxes:] = -1
            indices = indices.unsqueeze(0).repeat(bs, 1)
        else:
            indices = gt_valid_flag.argsort(dim=1, descending=True)[:, :num_sequence]
            prompt_seq = torch.gather(
                prompt_seq, dim=1, index=indices.unsqueeze(2).repeat(1, 1, self.prompt_length))

        return prompt_seq, indices
