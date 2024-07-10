import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from detectron2.structures import Boxes
from .raw_task import TaskProcessor


def _get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


class DetectionProcessor(TaskProcessor):
    def __init__(
            self,
            *,
            num_bins: int = 500,
            num_queries: int = 100,
            use_skip_vocab: bool = False,
            **kwargs,
    ):
        kwargs = self.rewrite_param_default(**kwargs)
        vocab_dict = kwargs["vocab_dict"]
        prompt_length = kwargs["prompt_length"]

        kwargs["max_seq_length"] = int(num_queries * 5) + prompt_length

        # build vocab mask to ignore the loss computation of specific tokens
        self.task_vacab_endidx = vocab_dict["text_range"][0] if use_skip_vocab else vocab_dict["vocab_length"]

        infer_mask = torch.zeros(num_queries, 5, self.task_vacab_endidx)
        infer_mask[:, :4, vocab_dict["coord_range"][0]: vocab_dict["coord_range"][1]] = 1
        infer_mask[:, 4, vocab_dict["class_range"][0]: vocab_dict["class_range"][1]] = 1
        infer_mask = infer_mask.flatten(0, 1)
        kwargs["infer_mask"] = infer_mask

        # mask vocab to separate the loss computation for different tasks for efficiency
        if kwargs.pop("use_vocab_mask", False):
            vocab_mask = torch.ones(self.task_vacab_endidx)
            vocab_mask[vocab_dict["class_range"][0]: vocab_dict["class_range"][1]] = 0
            vocab_mask[vocab_dict["fake_class_token"]] = 0
            vocab_mask[vocab_dict["coord_range"][0]: vocab_dict["coord_range"][1]] = 0
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

        super(DetectionProcessor, self).__init__(**kwargs)

        self.num_bins = num_bins
        self.num_queries = num_queries

    def rewrite_param_default(self, **kwargs):
        kwargs["num_seq_train"] = kwargs.pop("num_seq_train", 1)
        kwargs["num_seq_test"] = kwargs.pop("num_seq_test", 1)
        kwargs["prompt_length"] = kwargs.pop("prompt_length", 6)
        return kwargs

    def process_training_input(self, batch_size, targets=None):
        """
        :param batch_size:
        :param targets:
        :return:
        """
        batched_target_seq = []
        for target_per_img in targets:
            gt_classes = target_per_img.gt_classes.unsqueeze(1)
            gt_boxes = target_per_img.norm_gt_boxes
            gt_boxes = (gt_boxes * self.num_bins).floor().long().clamp(min=0, max=self.num_bins)

            random_class = torch.randint(
                self.vocab_dict["class_range"][0], self.vocab_dict["class_range"][1],
                (self.num_queries, 1), device=self.device)
            random_box_x0y0 = torch.rand(self.num_queries, 2, device=self.device)
            random_box_wh = torch.rand(self.num_queries, 2, device=self.device)
            random_box_x1y1 = (random_box_x0y0 + random_box_wh).clamp(min=0, max=1)
            random_box = torch.cat([random_box_x0y0, random_box_x1y1], dim=1)
            random_box = (random_box * self.num_bins).floor().long().clamp(min=0, max=self.num_bins)
            random_box = random_box + self.vocab_dict["coord_range"][0]

            target_seq_per_img = torch.cat([random_box, random_class], dim=1)

            num_gt_per_img = gt_classes.shape[0]
            if num_gt_per_img > 0:
                gt_per_img = torch.cat([
                    gt_boxes + self.vocab_dict["coord_range"][0],
                    gt_classes + self.vocab_dict["class_range"][0]
                ], dim=1).long()
                rand_index = torch.rand(self.num_queries).argsort(dim=0)[:num_gt_per_img].to(self.device)
                target_seq_per_img[rand_index] = gt_per_img

            batched_target_seq.append(target_seq_per_img.flatten())

        batched_target_seq = torch.stack(batched_target_seq, dim=0).unsqueeze(1).long()
        batched_prompt_seq = self.task_seq.reshape(1, 1, -1).repeat(batch_size, 1, 1).long()

        all_input_seq = []
        all_target_seq = []
        all_seq_mask = []
        for mr in self.train_mask_ratio:
            noise = torch.rand(batched_target_seq.shape).to(self.device)
            input_seq, seq_mask = self.apply_masking(batched_target_seq, noise, mr)
            input_seq = torch.cat([batched_prompt_seq, input_seq], dim=2).long()

            all_input_seq.append(input_seq)
            all_target_seq.append(batched_target_seq)
            all_seq_mask.append(seq_mask)

        all_input_seq = torch.cat(all_input_seq, dim=1)
        all_target_seq = torch.cat(all_target_seq, dim=1)
        all_seq_mask = torch.cat(all_seq_mask, dim=1)

        return all_input_seq, all_target_seq, all_seq_mask

    def process_training_triplet(self, pred_logits, ori_tgt_seq, seq_mask, targets):
        """
        :param
            pred_logits: [num_decoder, bs, num_sequence, sequence_length, vocabulary_size]
            ori_tgt_seq: [bs, num_sequence, sequence_length - prompt_length]
            seq_mask: [bs, num_sequence, sequence_length - prompt_length]
            targets: target dict for batched images
        :return:
            input_sequence: [batch_size, 1, 1] (last two dim for num_seq and seq_length)
            target_seq:
            target_weight:
        """
        # first process the target dict
        gt_token_list = []
        for targets_per_img in targets:
            gt_classes = targets_per_img.gt_classes.unsqueeze(1)
            gt_boxes = targets_per_img.norm_gt_boxes
            gt_boxes = (gt_boxes * self.num_bins).floor().long().clamp(min=0, max=self.num_bins)
            if gt_classes.shape[0] == 0:
                gt_token_list.append(torch.zeros(0, 5, device=self.device, dtype=torch.long))
            else:
                target_token_per_img = torch.cat([
                    gt_boxes + self.vocab_dict["coord_range"][0],
                    gt_classes + self.vocab_dict["class_range"][0]
                ], dim=1).long()
                gt_token_list.append(target_token_per_img)

        if not self.use_aux_loss:
            pred_logits = pred_logits[[-1]]

        nd, bs, num_seq, seq_length, vocab_size = pred_logits.shape
        # assert num_seq == self.num_sequence, "the shape of the pred prob does not meet the requirement."
        assert seq_length == self.max_seq_length, "the shape of the pred prob does not meet the requirement."

        if self.vocab_mask is not None:
            pred_logits.masked_fill_(self.vocab_mask, float('-inf'))

        pred_logits = pred_logits[:, :, :, self.prompt_length:]
        raw_pred_logits = pred_logits.permute(0, 4, 1, 2, 3)  # permute diminsion for cross entropy loss

        detached_logits = pred_logits.clone().detach()
        tgt_seq_logits = torch.zeros_like(detached_logits) - 20
        ori_tgt_seq = ori_tgt_seq.unsqueeze(0).repeat(nd, 1, 1, 1).unsqueeze(-1)
        tgt_seq_logits = torch.scatter(tgt_seq_logits, dim=4, index=ori_tgt_seq, value=20)
        seq_mask = seq_mask.unsqueeze(0).unsqueeze(-1)
        detached_logits = torch.where(seq_mask.bool(), detached_logits, tgt_seq_logits)
        detached_logits = detached_logits.transpose(1, 2).reshape(
            int(nd * num_seq), int(bs * self.num_queries), 5, vocab_size)

        target_seq = []
        target_weight = []
        for logit in detached_logits:
            target_seq_per_layer, target_weights_per_layer = \
                self.match_and_build_target_sequence(logit, gt_token_list, bs)
            target_seq.append(target_seq_per_layer)
            target_weight.append(target_weights_per_layer)

        target_seq = torch.stack(target_seq, dim=0).reshape(nd, num_seq, bs, -1).transpose(1, 2)
        target_weight = torch.stack(target_weight, dim=0).reshape(nd, num_seq, bs, -1).transpose(1, 2)
        target_weight = target_weight * seq_mask.squeeze(-1)  # only compute loss on masked tokens

        if self.vocab_mask is not None:
            assert self.vocab_mask[target_seq].eq(0).all(), \
                "if vocab mask is adopted, all the target should avoid gather " \
                "the masked logits in case of nan loss"

        for token in self.weighted_token_list:
            token_mask = target_seq.eq(token)
            target_weight[token_mask] *= self.token_weight

        return raw_pred_logits, target_seq, target_weight

    def process_inference(self, result_dict, pred_logits, batched_img_sizes):
        """
        :param pred_logits: [bs, 1, sequence_length, vocab_size]
        :return:
        """
        bs = pred_logits.shape[0]
        if self.vocab_mask is not None:
            pred_logits.masked_fill(self.vocab_mask, float('-inf'))
        pred_seq_prob = pred_logits.softmax(dim=3) * self.infer_mask + self.infer_mask
        # we add self.cls_coord_mask incase that at the early stage of training,
        # the desire token might get zero score
        pred_seq_prob = pred_seq_prob.reshape(bs, self.num_queries, 5, -1)
        pred_seq_score, pred_seq = pred_seq_prob.max(dim=-1)
        output_score = pred_seq_score[:, :, 4] - 1
        output_cls = pred_seq[:, :, 4] - self.vocab_dict["class_range"][0]
        output_box = ((pred_seq[:, :, :4] - self.vocab_dict["coord_range"][0]) / self.num_bins).clamp(min=0, max=1)

        unscaled_boxes = output_box.clone()
        batched_box_list = []
        for box_per_img, img_size in zip(output_box, batched_img_sizes):
            box_per_img = Boxes(box_per_img)
            box_per_img.scale(scale_x=img_size[1], scale_y=img_size[0])
            box_per_img.clip(img_size)
            batched_box_list.append(box_per_img)

        result_dict["unscaled_boxes"] = unscaled_boxes
        result_dict["pred_boxes"] = batched_box_list
        result_dict["scores"] = output_score
        result_dict["pred_classes"] = output_cls
        return result_dict

    @torch.no_grad()
    def match_and_build_target_sequence(self, pred_logits, gt_token_list, batch_size):
        """
        :param pred_logits: [int(bs * num_instance), 5, vocabulary_size]
        :param gt_token_list: list of target tokens for each image
        :return:
            # num_instances equals 1 for detection
            target_seq: [bs, sequence_length(num_queries * 5)]
            target_weights: [bs, sequence_length(num_queries * 5)]
        """
        # matcher
        gt_tokens = torch.cat(gt_token_list, dim=0).long()
        total_num_gt = gt_tokens.shape[0]
        pred_size = pred_logits.shape[0]
        gt_tokens = gt_tokens.unsqueeze(0).repeat(pred_size, 1, 1)
        pred_logits = pred_logits.permute(0, 2, 1).unsqueeze(2).repeat(1, 1, total_num_gt, 1)
        cost_per_objects = F.cross_entropy(pred_logits, gt_tokens, reduction='none').mean(dim=-1)
        cost_per_objects = cost_per_objects.view(batch_size, self.num_queries, total_num_gt).cpu()

        sizes = [t.shape[0] for t in gt_token_list]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_per_objects.split(sizes, -1))]
        indices = [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

        # build the target sequence and weight based on matched indices
        target_seq = torch.full(
            (batch_size, self.num_queries, 5),
            self.vocab_dict["fake_class_token"],
            dtype=torch.int64, device=self.device,
        )

        batched_idx = _get_src_permutation_idx(indices)
        matched_tgt_tokens = torch.cat([gtt[J] for gtt, (_, J) in zip(gt_token_list, indices)], dim=0)
        target_seq[batched_idx] = matched_tgt_tokens

        target_weights = torch.zeros_like(target_seq)
        target_weights[:, :, 4] = 1
        target_weights[batched_idx] = 1

        target_seq = target_seq.flatten(1, 2)
        target_weights = target_weights.flatten(1, 2)

        return target_seq, target_weights
