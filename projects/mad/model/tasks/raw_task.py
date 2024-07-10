import torch
import torch.nn as nn
from abc import abstractmethod


class TaskProcessor(nn.Module):
    """
        Task Processor:
        function "__init__" and "rewrite_param_default" for defininig a task-specific processor
        function "build_target_sequence" and "process_training_input" for building input during training
        function
    """
    def __init__(
            self,
            vocab_dict: dict,
            num_seq_train: int,
            num_seq_test: int,
            prompt_length: int,
            max_seq_length: int,
            task_seq: list,
            infer_mask: torch.Tensor,
            vocab_mask: torch.Tensor = None,
            train_mask_ratio: tuple[float] = (0.7,),
            test_mask_ratio: tuple[float] = (1.0,),
            use_noised_train: bool = False,
            use_rand_infer_mask: bool = False,
            use_aux_loss: bool = True,
            weighted_token_list: tuple[int] = (),
            token_weight: float = 0.1,
            device: str = "cuda",
    ):
        """
        Args:
            vocab_dict: vocabulary dictionary that contains all task tokens.
            num_seq_train: number of target sequences during training. the sequences are masked with different
                ratios, leading to num_seq_train * number of mask ratios trained sequences.
            num_seq_test: number of generated sequences during inference.
            prompt_length: length of task prompt sequence
            max_seq_length: sequence length for current task
            task_seq: task prompt tokens/sequences
            infer_mask: vocabulary mask for decoding pred logits into special format of tasks
            vocab_mask: vocabulary mask for pointing out involved vocabulary for current task
            train_mask_ratio: mask ratio tuple for training
            test_mask_ratio: mask ratio tuple for inference. multistep masked inference
            use_rand_infer_mask: whether to use random masking during masked inference
            use_aux_loss: auxiliary loss after each decoder layer
            token_weight_dict: dict contains tokens that need to be weighted and corresponding weights.
            device:
        """
        super(TaskProcessor, self).__init__()
        self.vocab_dict = vocab_dict
        self.num_seq_train = num_seq_train
        self.num_seq_test = num_seq_test
        self.prompt_length = prompt_length
        self.max_seq_length = max_seq_length
        self.register_buffer("task_seq", torch.tensor(task_seq), persistent=False)
        self.register_buffer("infer_mask", infer_mask, persistent=False)
        self.register_buffer("vocab_mask", vocab_mask, persistent=False)

        self.train_mask_ratio = train_mask_ratio
        assert len(self.train_mask_ratio) > 0, "training mask ratio must be greater than 0."
        self.test_mask_ratio = test_mask_ratio
        self.use_noised_train = use_noised_train
        self.use_rand_infer_mask = use_rand_infer_mask
        self.use_aux_loss = use_aux_loss
        self.weighted_token_list = weighted_token_list
        self.token_weight = token_weight

        self.num_infer_stage = len(test_mask_ratio)
        self.device = torch.device(device)

    def rewrite_param_default(self, **kwargs):
        return kwargs

    @abstractmethod
    def process_training_input(self, batch_size, targets=None):
        """
        :param batch_size:
        :param targets:
        :return:
            input_seq_all: input sequence with prompt [bs, num_seq, seq_length]
            target_seq_all: target sequence [bs, num_seq, seq_length - prompt_length]
            seq_mask_all: seq mask, 1 for tokens that should be considered in loss,
                    0 for unmasked tokens or padded tokens. [bs, num_seq, seq_length - prompt_length]
        """
        pass

    def process_inference_input(
            self,
            batch_size,
            iter_idx,
            pred_dict,
            last_pred_logits=None,
            last_seq_mask=None,
            pred_seq_logits=None):
        """
        Args:
            batch_size:
            iter_idx: infer stage
            pred_dict: predictions from other tasks
            last_pred_logits: [batch_size, num_sequence, max_sequence_length - prompt_length, vocab_size]
            last_seq_mask: [batch_size, num_sequence, max_sequence_length - prompt_length]
            pred_seq_logits: [batch_size, num_sequence, max_sequence_length, vocab_size]
        Returns:
            masked_seq:
            seq_mask:
            ensembled_seq_logits:
        """
        box = pred_dict.get("unscaled_boxes", None)
        cls = pred_dict.get("pred_classes", None)
        batched_prompt_seq = self.build_batched_prompt_seq(batch_size, box, cls)

        if pred_seq_logits is None or iter_idx == 0:
            masked_seq = torch.full(
                (batch_size, self.num_seq_test, self.max_seq_length,),
                self.vocab_dict["mask_token"],
                dtype=torch.long, device=self.device)
            masked_seq[:, :, :self.prompt_length] = batched_prompt_seq
            seq_mask = torch.ones_like(masked_seq[:, :, self.prompt_length:]).bool()
            return masked_seq, seq_mask, None

        pred_seq_logits = pred_seq_logits[:, :, self.prompt_length:]

        if last_pred_logits is not None:
            last_seq_mask = last_seq_mask.bool().unsqueeze(-1)
            ensembled_seq_logits = torch.where(
                last_seq_mask, (last_pred_logits + pred_seq_logits) / 2, last_pred_logits)
        else:
            ensembled_seq_logits = pred_seq_logits

        if self.vocab_mask is not None:
            seq_logits = ensembled_seq_logits.masked_fill(self.vocab_mask, float('-inf'))
        else:
            seq_logits = ensembled_seq_logits

        # we add self.cls_coord_mask incase that at the early stage of training,
        # the desire token might get zero score
        pred_seq_prob = seq_logits.softmax(dim=3) * self.infer_mask + self.infer_mask
        pred_seq_score, pred_seq = pred_seq_prob.max(dim=3)
        pred_seq_score = pred_seq_score - 1

        if self.use_rand_infer_mask:
            pred_seq_score = torch.rand_like(pred_seq_score)

        pred_seq_score = self.process_pred_score(pred_seq_score, iter_idx)

        masked_seq, seq_mask = self.apply_masking(
            pred_seq, pred_seq_score, self.test_mask_ratio[iter_idx])
        masked_seq = torch.cat([batched_prompt_seq, masked_seq], dim=2).long()

        return masked_seq, seq_mask, ensembled_seq_logits

    def process_pred_score(self, score, iter_idx):
        """
        Args:
            score: [batch_size, num_sequence, max_sequence_length - prompt_length]
        Returns:
            score
        """
        return score

    def build_batched_prompt_seq(self, batch_size, boxes=None, classes=None):
        """
        :param batch_size:
        :param boxes: [bs, num_boxes, 4]
        :param classes: [bs, num_boxes]
        :return:
            batched_prompt_seq: [bs, num_seq, prompt_length]
        """
        batched_prompt_seq = self.task_seq.reshape(1, 1, -1).repeat(batch_size, self.num_seq_test, 1)
        return batched_prompt_seq

    def process_training_triplet(self, pred_logits, target_seq, seq_mask, targets):
        """
        :param
            pred_logits: [num_decoder, bs, num_sequence, sequence_length, vocabulary_size]
            target_seq: [bs, num_sequence, sequence_length]
            seq_mask: [bs, num_sequence, sequence_length]
        :return:
            input_sequence: [batch_size, 1, 1] (last two dim for num_seq and seq_length)
        """
        # first process the target dict
        target_seq = target_seq.unsqueeze(0)
        target_weight = seq_mask.unsqueeze(0)

        if not self.use_aux_loss:
            pred_logits = pred_logits[[-1]]

        nd, bs, num_seq, seq_length, vocab_size = pred_logits.shape
        assert seq_length == self.max_seq_length, \
            "the shape of the pred prob does not meet the requirement."

        if self.vocab_mask is not None:
            pred_logits.masked_fill_(self.vocab_mask, float('-inf'))

        # permute diminsion for cross entropy loss
        pred_logits = pred_logits[:, :, :, self.prompt_length:].permute(0, 4, 1, 2, 3)

        target_seq = target_seq.repeat(nd, 1, 1, 1)
        target_weight = target_weight.repeat(nd, 1, 1, 1)

        if self.vocab_mask is not None:
            assert self.vocab_mask[target_seq].eq(0).all(), \
                "if vocab mask is adopted, all the target should avoid gather " \
                "the masked logits in case of nan loss"

        for token in self.weighted_token_list:
            token_mask = target_seq.eq(token)
            target_weight[token_mask] *= self.token_weight

        return pred_logits, target_seq, target_weight

    @abstractmethod
    def process_inference(self, result_dict, pred_logits, batched_img_sizes):
        """
            process prediction for inference
        """
        pass

    def apply_masking(self, input_seq, seq_score, mask_ratio):
        """
        mask the token with the lowest scores
        Args:
            input_seq: [batch_size, num_sequence, max_sequence_length - prompt_length]
            seq_score: [batch_size, num_sequence, max_sequence_length - prompt_length]
        Returns:
            masked_seq: [batch_size, num_sequence, max_sequence_length - prompt_length]
            seq_mask: [batch_size, num_sequence, max_sequence_length - prompt_length]
        """
        num_mask = int(input_seq.shape[2] * mask_ratio)
        masked_index = torch.argsort(seq_score, dim=2)[:, :, :num_mask]
        masked_seq = torch.scatter(input_seq, dim=2, index=masked_index, value=self.vocab_dict["mask_token"])
        seq_mask = torch.zeros_like(masked_seq)
        seq_mask = torch.scatter(seq_mask, dim=2, index=masked_index, value=1).bool()
        return masked_seq, seq_mask
