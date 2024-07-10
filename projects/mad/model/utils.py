import numpy as np
from math import pi

import torch
from torch import nn

from einops import rearrange, repeat


class KNNClassifier(nn.Module):
    def __init__(
            self,
            num_embeddings,
            embed_dim,
            predictor=None,
            amp_infer=False,
            logit_scale: float = 0.07,
            use_outbias=False):
        super().__init__()
        self.num_embedings = num_embeddings
        self.embeddings = nn.Embedding(num_embeddings, embed_dim)
        self.predictor = predictor
        self.amp_infer = amp_infer

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / logit_scale))
        self.use_outbias = use_outbias
        if self.use_outbias:
            self.output_bias = nn.Parameter(torch.empty(num_embeddings))
            nn.init.trunc_normal_(self.output_bias, mean=0.0, std=0.02)

    def forward(self, x, end_ind: int = None):
        """
        Args:
            x: [nd, bs, num_seq, seq_length, dim]
            end_ind: skip the vocab after ending index to speed up. start index is set to 0 to ensure that
             the target index is consistent.
        Returns:
            x: [nd, bs, num_seq, seq_length, vocab_size]
        """
        if self.predictor is not None:
            x = self.predictor(x)
        nd, bs, num_seq, seq_length, _ = x.shape

        # skip some vocab for efficiency
        if end_ind is None:
            end_ind = self.num_embedings
        vocab_embed = self.embeddings.weight[:end_ind]

        # normalize the embeddings
        x = x / x.norm(dim=-1, keepdim=True)
        vocab_embed = vocab_embed / vocab_embed.norm(dim=-1, keepdim=True)

        if not self.training and self.amp_infer:
            with torch.cuda.amp.autocast():
                logits = x @ vocab_embed.t()
        else:
            logits = x @ vocab_embed.t()

        logits = self.logit_scale.exp() * logits

        if self.use_outbias:
            logits = logits + self.output_bias[:end_ind]

        return logits


class List2ModuleDict(nn.ModuleDict):
    def __init__(self, module_name, module_list):
        assert len(module_name) == len(module_list), "unmatched numbers of module names and modules."
        assert all([isinstance(name, str) for name in module_name]), "module names must be strings."
        module_dict = dict()
        for name, module in zip(module_name, module_list):
            module_dict[name] = module
        super().__init__(module_dict)


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all(
        [*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
            self,
            dim,
            pt_seq_len=16,
            ft_seq_len=None,
            custom_freqs=None,
            freqs_for='lang',
            theta=10000,
            max_freq=10,
            num_freqs=1,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        if ft_seq_len is None: ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs = torch.einsum('..., f -> ... f', t, freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r=2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)

        freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1])

        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

        print('======== shape of rope freq', self.freqs_cos.shape, '========')

    def forward(self, t):
        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin
