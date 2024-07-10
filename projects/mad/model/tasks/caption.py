import torch
import torch.nn.functional as F
from .raw_task import TaskProcessor


class CaptionProcessor(TaskProcessor):
    def __init__(
            self,
            *,
            tokenizer=None,
            max_sentence_length: int = 20,
            num_target_per_img: int = 5,
            **kwargs,
    ):
        kwargs = self.rewrite_param_default(**kwargs)
        vocab_dict = kwargs["vocab_dict"]
        prompt_length = kwargs["prompt_length"]

        kwargs["max_seq_length"] = max_sentence_length + prompt_length

        # build vocab mask to ignore the loss computation of specific tokens
        self.task_vacab_endidx = None

        # mask to make sure that the prediction is in the proper format
        infer_mask = torch.zeros(vocab_dict["vocab_length"])
        infer_mask[vocab_dict["text_range"][0]: vocab_dict["text_range"][1]] = 1
        kwargs["infer_mask"] = infer_mask

        # mask vocab to separate the loss computation for different tasks
        if kwargs.pop("use_vocab_mask", False):
            vocab_mask = torch.ones(vocab_dict["vocab_length"])
            vocab_mask[vocab_dict["text_range"][0]: vocab_dict["text_range"][1]] = 0
            vocab_mask = vocab_mask.bool()
        else:
            if "pred_token_shift" in vocab_dict:
                # avoid compute loss for prompt tokens
                vocab_mask = torch.ones(vocab_dict["vocab_length"])
                vocab_mask[vocab_dict["pred_token_shift"]:] = 0
                vocab_mask = vocab_mask.bool()
            else:
                vocab_mask = None
        kwargs["vocab_mask"] = vocab_mask

        super(CaptionProcessor, self).__init__(**kwargs)

        self.tokenizer = tokenizer
        self.eos_idx = tokenizer.eos_idx
        self.pad_idx = tokenizer.padding_idx
        self.max_sent_length = max_sentence_length
        self.num_target_per_img = num_target_per_img

    def rewrite_param_default(self, **kwargs):
        kwargs["num_seq_train"] = kwargs.pop("num_seq_train", 1)
        kwargs["num_seq_test"] = kwargs.pop("num_seq_test", 1)
        kwargs["prompt_length"] = kwargs.pop("prompt_length", 6)
        kwargs["use_rand_infer_mask"] = kwargs.pop("use_rand_infer_mask", True)
        return kwargs

    def process_training_input(self, batch_size, targets=None):
        """
        Args:
            batch_size:
            targets: list(target_dict) the target_dict is expected to contain key "captions"
        Returns:
            input_seq: comprise of masked token or target tokens. [bs, num_seq, seq_length]
            seq_mask: 1 for masked token, 0 for target/prompt tokens which will not be considered in
                loss computation. shape: [bs, num_seq, seq_length]
            target_seq: shape: [bs, num_seq, seq_length]
        """
        batched_target_seq = []
        batched_valid_mask = []
        for gt_caption_tokens in targets:
            target_seq = gt_caption_tokens + self.vocab_dict["text_range"][0]
            assert target_seq.shape[1] == self.max_sent_length
            valid_mask = torch.ones_like(target_seq)
            if target_seq.shape[0] < self.num_target_per_img:
                pad_seq = torch.full(
                    (self.num_target_per_img - target_seq.shape[0], self.max_sent_length),
                    self.vocab_dict["text_pad_token"], device=self.device, dtype=torch.long)
                pad_mask = torch.zeros_like(pad_seq)
                target_seq = torch.cat([target_seq, pad_seq], dim=0)
                valid_mask = torch.cat([valid_mask, pad_mask], dim=0)
            else:
                target_seq = target_seq[:self.num_target_per_img]
                valid_mask = valid_mask[:self.num_target_per_img]

            batched_target_seq.append(target_seq)
            batched_valid_mask.append(valid_mask)

        batched_target_seq = torch.stack(batched_target_seq, dim=0)
        batched_valid_mask = torch.stack(batched_valid_mask, dim=0)
        batched_prompt_seq = self.task_seq.reshape(1, 1, -1).repeat(batch_size, batched_target_seq.shape[1], 1)

        all_input_seq = []
        all_target_seq = []
        all_seq_mask = []
        for mr in self.train_mask_ratio:
            noise = torch.rand(batched_target_seq.shape).to(self.device)
            input_seq, seq_mask = self.apply_masking(batched_target_seq, noise, mr)
            seq_mask = seq_mask * batched_valid_mask

            if self.use_noised_train:
                noise_token = torch.randint(
                    self.vocab_dict["text_range"][0], self.vocab_dict["text_range"][1],
                    size=input_seq.shape[:2], device=self.device).unsqueeze(-1)
                # add seq mask to avoid add noise on masked token
                noise = torch.rand_like(seq_mask.float()) + seq_mask
                noise_index = noise.argmin(dim=2, keepdim=True)
                input_seq = torch.scatter(input_seq, dim=2, index=noise_index, src=noise_token)

            input_seq = torch.cat([batched_prompt_seq, input_seq], dim=2).long()

            if mr == 1.0:
                all_input_seq.append(input_seq[:, None, 0])
                all_target_seq.append(batched_target_seq[:, None, 0])
                all_seq_mask.append(seq_mask[:, None, 0])
            else:
                all_input_seq.append(input_seq)
                all_target_seq.append(batched_target_seq)
                all_seq_mask.append(seq_mask)

        all_input_seq = torch.cat(all_input_seq, dim=1)
        all_target_seq = torch.cat(all_target_seq, dim=1)
        all_seq_mask = torch.cat(all_seq_mask, dim=1)

        return all_input_seq, all_target_seq, all_seq_mask

    def process_training_triplet(self, pred_logits, target_seq, seq_mask, targets):
        """
        Args:
        if using masked trainig, the first sequence will be fully masked sequence, while the
        followings are partly masked sequence.
            pred_logits: [bs, num_seq, seq_length, vocab_size]
            target_seq: [bs, num_seq, seq_length]
            seq_mask: [bs, num_seq, seq_length]
            targets:

        Returns:

        """
        batched_gt_tokens = []
        batched_pad_flag = []
        for _gt_caption_tokens in targets:
            gt_caption_tokens = _gt_caption_tokens.clone() + self.vocab_dict["text_range"][0]
            pad_flag = torch.zeros(self.num_target_per_img, device=self.device)
            assert gt_caption_tokens.shape[1] == self.max_sent_length
            if gt_caption_tokens.shape[0] < self.num_target_per_img:
                pad_seq = torch.full(
                    (self.num_target_per_img - gt_caption_tokens.shape[0], self.max_sent_length),
                    self.vocab_dict["text_pad_token"], device=self.device, dtype=torch.long)
                pad_flag[gt_caption_tokens.shape[0]:] = 1
                gt_caption_tokens = torch.cat([gt_caption_tokens, pad_seq], dim=0)
            else:
                gt_caption_tokens = gt_caption_tokens[:self.num_target_per_img]
                pad_flag = pad_flag[:self.num_target_per_img]

            batched_gt_tokens.append(gt_caption_tokens)
            batched_pad_flag.append(pad_flag)

        # shape: [bs, num_masked_seq_train, (sequence_length)]
        batched_gt_tokens = torch.stack(batched_gt_tokens, dim=0)
        batched_pad_flag = torch.stack(batched_pad_flag, dim=0)

        if not self.use_aux_loss:
            pred_logits = pred_logits[[-1]]

        nd, bs, num_seq, seq_length, vocab_size = pred_logits.shape
        assert seq_length == self.max_seq_length, \
            "the shape of the pred prob does not meet the requirement."

        if self.vocab_mask is not None:
            pred_logits.masked_fill_(self.vocab_mask, float('-inf'))

        # permute diminsion for cross entropy loss
        pred_logits = pred_logits[:, :, :, self.prompt_length:].permute(0, 4, 1, 2, 3)

        matched_target_seq = []
        matched_valid_flag = []
        for logit in pred_logits[:, :, :, 0, :]:
            target_seq_per_layer, valid_flag_per_layer = self.match_and_build_target_sequence(
                logit, batched_gt_tokens, batched_pad_flag)
            matched_target_seq.append(target_seq_per_layer)
            matched_valid_flag.append(valid_flag_per_layer)
        matched_target_seq = torch.stack(matched_target_seq, dim=0).long()
        matched_valid_flag = torch.stack(matched_valid_flag, dim=0)

        target_seq = target_seq.unsqueeze(0).repeat(nd, 1, 1, 1).long()
        target_seq[:, :, 0, :] = matched_target_seq
        target_weight = seq_mask.unsqueeze(0).repeat(nd, 1, 1, 1)
        target_weight[:, :, 0, :] = matched_valid_flag

        if self.vocab_mask is not None:
            assert self.vocab_mask[target_seq].eq(0).all(), \
                "if vocab mask is adopted, all the target should avoid gather " \
                "the masked logits in case of nan loss"

        for token in self.weighted_token_list:
            token_mask = target_seq.eq(token)
            target_weight[token_mask] *= self.token_weight

        return pred_logits, target_seq, target_weight

    def process_inference(self, result_dict, pred_logits, batched_img_sizes):
        """
        Args:
            result_dict:
            pred_logits: [bs, num_sequence, sequence_length, vocab_size]
            batched_img_sizes:

        Returns:
            result_dict:
        """
        pred_logits = pred_logits.squeeze(1)  # assert that we only predict one caption for each image
        if self.vocab_mask is not None:
            pred_logits = pred_logits.masked_fill(self.vocab_mask, float('-inf'))
        pred_prob = pred_logits.softmax(dim=2) * self.infer_mask + self.infer_mask
        pred_tokens = pred_prob.argmax(dim=2) - self.vocab_dict["text_range"][0]

        pred_texts = []
        for tokens in pred_tokens:
            eos_index = tokens.eq(self.eos_idx).nonzero()
            if len(eos_index) > 0:
                eos_index = eos_index.min()
                tokens[eos_index] = self.eos_idx
                tokens[eos_index + 1:] = self.pad_idx
            pred_texts.append(self.tokenizer.decode(tokens.cpu().tolist()))

        result_dict["pred_captions"] = pred_texts

        return result_dict

    @torch.no_grad()
    def match_and_build_target_sequence(self, _pred_logits, gt_tokens, pad_flag):
        """
        :param _pred_logits: [vocab_size, bs, seq_length]
        :param gt_tokens: [bs, num_target_seq, seq_length]
        :param pad_flag: [bs, num_target_seq]
        :return:
            target_seq: [bs, sequence_length]
            target_weights: [bs, sequence_length]
        """
        # matcher
        pred_logits = _pred_logits.clone().permute(1, 0, 2).unsqueeze(2).repeat(1, 1, self.num_target_per_img, 1)
        cost = F.cross_entropy(pred_logits, gt_tokens, reduction='none').mean(dim=-1)
        cost = cost + pad_flag * 100  # cost shape [bs, num_masked_seq_train]
        indices = cost.argmin(dim=1)

        batch_size = gt_tokens.shape[0]
        target_seq = gt_tokens[range(batch_size), indices]
        valid_flag = (1 - pad_flag[range(batch_size), indices]).unsqueeze(1).repeat(1, target_seq.shape[1])

        return target_seq, valid_flag
