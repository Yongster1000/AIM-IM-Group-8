import copy
from typing import Optional, Any, Union, Callable, Tuple

import torch
import warnings
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.module import Module
#from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
#from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.activation import _is_make_fx_tracing, _check_arg_device, _arg_requires_grad
import warnings
#from typing import Optional, Tuple

from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from decompx_utils import DecompXConfig, DecompXOutput
import math


__all__ = ['Transformer', 'TransformerEncoder', 'TransformerDecoder', 'TransformerEncoderLayer', 'TransformerDecoderLayer']


def _generate_square_subsequent_mask(
        sz: int,
        device: torch.device = torch.device(torch._C._get_default_device()),  # torch.device('cpu'),
        dtype: torch.dtype = torch.get_default_dtype(),
) -> Tensor:
    r"""Generate a square causal mask for the sequence.

    The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
    """
    return torch.triu(
        torch.full((sz, sz), float('-inf'), dtype=dtype, device=device),
        diagonal=1,
    )


def _get_seq_len(
        src: Tensor,
        batch_first: bool
) -> Optional[int]:

    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            # unbatched: S, E
            return src_size[0]
        else:
            # batched: B, S, E if batch_first else S, B, E
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]

class MultiheadAttention(Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces as described in the paper:
    `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    Multi-Head Attention is defined as:

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    ``nn.MultiHeadAttention`` will use the optimized implementations of
    ``scaled_dot_product_attention()`` when possible.

    In addition to support for the new ``scaled_dot_product_attention()``
    function, for speeding up Inference, MHA will use
    fastpath inference with support for Nested Tensors, iff:

    - self attention is being computed (i.e., ``query``, ``key``, and ``value`` are the same tensor).
    - inputs are batched (3D) with ``batch_first==True``
    - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor argument ``requires_grad``
    - training is disabled (using ``.eval()``)
    - ``add_bias_kv`` is ``False``
    - ``add_zero_attn`` is ``False``
    - ``batch_first`` is ``True`` and the input is batched
    - ``kdim`` and ``vdim`` are equal to ``embed_dim``
    - if a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ is passed, neither ``key_padding_mask``
      nor ``attn_mask`` is passed
    - autocast is disabled

    If the optimized inference fastpath implementation is in use, a
    `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be passed for
    ``query``/``key``/``value`` to represent padding more efficiently than using a
    padding mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_
    will be returned, and an additional speedup proportional to the fraction of the input
    that is padding can be expected.

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
            Default: ``False``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    Examples::

        >>> # xdoctest: +SKIP
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)

    .. _`FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`:
         https://arxiv.org/abs/2205.14135

    """

    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
            self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super().__setstate__(state)

    def transpose_for_scores_for_decomposed(self, x):
        # x: (B, N, N, H*V)
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        # x: (B, N, N, H, V)
        x = x.view(new_x_shape)
        # x: (B, H, N, N, V)
        return x.permute(0, 3, 1, 2, 4)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    def bias_decomposer(self, bias, attribution_vectors, bias_decomp_type="absdot"):
        # Decomposes the input bias based on similarity to the attribution vectors
        # Args:
        #   bias: a bias vector (all_head_size)
        #   attribution_vectors: the attribution vectors from token j to i (b, i, j, all_head_size) :: (batch, seq_length, seq_length, all_head_size) 

        if bias_decomp_type == "absdot":
            weights = torch.abs(torch.einsum("bskd,d->bsk", attribution_vectors, bias))
        elif bias_decomp_type == "abssim":
            weights = torch.abs(torch.nn.functional.cosine_similarity(attribution_vectors, bias, dim=-1))
            weights = (torch.norm(attribution_vectors, dim=-1) != 0) * weights
        elif bias_decomp_type == "norm":
            weights = torch.norm(attribution_vectors, dim=-1)
        elif bias_decomp_type == "equal":
            weights = (torch.norm(attribution_vectors, dim=-1) != 0) * 1.0
        elif bias_decomp_type == "cls":
            weights = torch.zeros(attribution_vectors.shape[:-1], device=attribution_vectors.device)
            weights[:,:,0] = 1.0
        elif bias_decomp_type == "dot":
            weights = torch.einsum("bskd,d->bsk", attribution_vectors, bias)
        elif bias_decomp_type == "biastoken":
            attrib_shape = attribution_vectors.shape
            if attrib_shape[1] == attrib_shape[2]:
                attribution_vectors = torch.concat([attribution_vectors, torch.zeros((attrib_shape[0], attrib_shape[1], 1, attrib_shape[3]), device=attribution_vectors.device)], dim=-2)
            attribution_vectors[:,:,-1] = attribution_vectors[:,:,-1] + bias
            return attribution_vectors

        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-12)
        weighted_bias = torch.matmul(weights.unsqueeze(dim=-1), bias.unsqueeze(dim=0))
        return attribution_vectors + weighted_bias
    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True,
            is_causal : bool = False, 
            attribution_vectors: Optional[torch.FloatTensor] = None,
            decompx_config: Optional[DecompXConfig] = None
            ) -> Tuple[Tensor, Optional[Tensor]]: #modified here
        r"""
    Args:
        query: Query embeddings of shape :math:`(L, E_q)` for unbatched input, :math:`(L, N, E_q)` when ``batch_first=False``
            or :math:`(N, L, E_q)` when ``batch_first=True``, where :math:`L` is the target sequence length,
            :math:`N` is the batch size, and :math:`E_q` is the query embedding dimension ``embed_dim``.
            Queries are compared against key-value pairs to produce the output.
            See "Attention Is All You Need" for more details.
        key: Key embeddings of shape :math:`(S, E_k)` for unbatched input, :math:`(S, N, E_k)` when ``batch_first=False``
            or :math:`(N, S, E_k)` when ``batch_first=True``, where :math:`S` is the source sequence length,
            :math:`N` is the batch size, and :math:`E_k` is the key embedding dimension ``kdim``.
            See "Attention Is All You Need" for more details.
        value: Value embeddings of shape :math:`(S, E_v)` for unbatched input, :math:`(S, N, E_v)` when
            ``batch_first=False`` or :math:`(N, S, E_v)` when ``batch_first=True``, where :math:`S` is the source
            sequence length, :math:`N` is the batch size, and :math:`E_v` is the value embedding dimension ``vdim``.
            See "Attention Is All You Need" for more details.
        key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
            to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`.
            Binary and float masks are supported.
            For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
            the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.
        need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
            Set ``need_weights=False`` to use the optimized ``scaled_dot_product_attention``
            and achieve the best performance for MHA.
            Default: ``True``.
        attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
            :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
            :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
            broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
            Binary and float masks are supported. For a binary mask, a ``True`` value indicates that the
            corresponding position is not allowed to attend. For a float mask, the mask values will be added to
            the attention weight.
            If both attn_mask and key_padding_mask are supplied, their types should match.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
            heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
            effect when ``need_weights=True``. Default: ``True`` (i.e. average weights across heads)
        is_causal: If specified, applies a causal mask as attention mask.
            Default: ``False``.
            Warning:
            ``is_causal`` provides a hint that ``attn_mask`` is the
            causal mask. Providing incorrect hints can result in
            incorrect execution, including forward and backward
            compatibility.

    Outputs:
        - **attn_output** - Attention outputs of shape :math:`(L, E)` when input is unbatched,
          :math:`(L, N, E)` when ``batch_first=False`` or :math:`(N, L, E)` when ``batch_first=True``,
          where :math:`L` is the target sequence length, :math:`N` is the batch size, and :math:`E` is the
          embedding dimension ``embed_dim``.
        - **attn_output_weights** - Only returned when ``need_weights=True``. If ``average_attn_weights=True``,
          returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
          head of shape :math:`(\text{num\_heads}, L, S)` when input is unbatched or :math:`(N, \text{num\_heads}, L, S)`.

        .. note::
            `batch_first` argument is ignored for unbatched inputs.
        """
        
        why_not_fast_path = ''
        if ((attn_mask is not None and torch.is_floating_point(attn_mask))
           or (key_padding_mask is not None) and torch.is_floating_point(key_padding_mask)):
            why_not_fast_path = "floating-point masks are not supported for fast path."

        is_batched = query.dim() == 3

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        #print('attn_mask = \n', attn_mask) #None
        if not is_batched:
            why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif self.in_proj_weight is None:
            why_not_fast_path = "in_proj_weight was None"
        elif query.dtype != self.in_proj_weight.dtype:
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif (self.num_heads % 2) != 0:
            why_not_fast_path = "self.num_heads is not even"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif query.is_nested and (key_padding_mask is not None or attn_mask is not None):
            why_not_fast_path = "supplying both src_key_padding_mask and src_mask at the same time \
                                 is not supported with NestedTensor input"
        elif torch.is_autocast_enabled():
            why_not_fast_path = "autocast is enabled"
        
        #print(why_not_fast_path)
        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif _is_make_fx_tracing():
                why_not_fast_path = "we are running make_fx tracing"
            elif not all(_check_arg_device(x) for x in tensor_args):
                why_not_fast_path = ("some Tensor argument's device is neither one of "
                                     f"cpu, cuda or {torch.utils.backend_registration._privateuse1_backend_name}")
            elif torch.is_grad_enabled() and any(_arg_requires_grad(x) for x in tensor_args):
                why_not_fast_path = ("grad is enabled and at least one of query or the "
                                     "input/output projection weights or biases requires_grad")
            #print(why_not_fast_path)
            if not why_not_fast_path:
                merged_mask, mask_type = self.merge_masks(attn_mask, key_padding_mask, query)

                if self.in_proj_bias is not None and self.in_proj_weight is not None:
                    return torch._native_multi_head_attention(
                        query,
                        key,
                        value,
                        self.embed_dim,
                        self.num_heads,
                        self.in_proj_weight,
                        self.in_proj_bias,
                        self.out_proj.weight,
                        self.out_proj.bias,
                        merged_mask,
                        need_weights,
                        average_attn_weights,
                        mask_type)
        
        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, ("MultiheadAttention does not support NestedTensor outside of its fast path. " +
                                f"The fast path was not hit because {why_not_fast_path}")
        

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))
        
        decomposed_value_layer = attribution_vectors
        
        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal)
        else:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal)
        if self.batch_first and is_batched:
            decompx_ready = decompx_config is not None
            if decompx_ready:
                #print('here?')
                E = query.size(-1)
                if attribution_vectors is not None:
                    #print('type = ', type(attribution_vectors))
                    decomposed_value_layer = torch.einsum("bijd,vd->bijv", attribution_vectors, self.in_proj_weight[2 * E:, :])
                    decomposed_value_layer = self.transpose_for_scores_for_decomposed(decomposed_value_layer) #size (B, H, N, N, V)
                
                bias_decomp_type = "biastoken" if decompx_config.include_bias_token else decompx_config.bias_decomp_type

                
                #print('w, ', self.out_proj.weight.shape)
                headmixing_weight = self.out_proj.weight.view(self.embed_dim, self.num_heads, 
                                                              self.head_dim)
                
                value = torch.matmul(value, self.in_proj_weight[2 * E:, :].transpose(0, 1)) + self.in_proj_bias[2 * E:]
                #print(value.shape) #(S, N, V)
                if decomposed_value_layer is None or decompx_config.aggregation != "vector":
                    
                    transformed_layer = torch.einsum('bhsv,dhv->bhsd', self.transpose_for_scores(value.transpose(0, 1)), headmixing_weight)  # V * W^o  (z=(qk)v)
                    # Make weighted vectors αf(x) from transformed vectors (transformed_layer)
                    # and attention weights (attentions):
                    # (batch, num_heads, seq_length, seq_length, all_head_size)
                    #print(attn_output_weights.shape) #(B, num_heads, target_seq_len, source_seq_len)

                    ###for validation
                    '''
                    sh = attn_output_weights.shape[-1]
                    y = self.transpose_for_scores(value.transpose(0, 1))
                    #print('y, ', y.shape)
                    a = torch.einsum('bcd, bde ->bce', attn_output_weights.contiguous().view(-1, sh, sh), y.contiguous().view(-1, y.shape[2], y.shape[3]))
                    
                    a = a.transpose(0, 1).contiguous().view(-1, y.shape[3] * y.shape[1])
                    #print(a.shape, self.out_proj.weight.transpose(0, 1).shape)
                    #print('self.out_proj.bias, ', self.out_proj.bias.shape)
                    a = torch.matmul(a, self.out_proj.weight.transpose(0, 1)) + self.out_proj.bias
                    a = a.view(y.shape[2], y.shape[0], -1)'''
                    ###

                    # Sum each weighted vectors αf(x) over all heads:
                    # (batch, seq_length, seq_length, all_head_size)
                    '''
                    weighted_layer = torch.einsum('bhks,bhsd->bhksd', attn_output_weights.to("cpu"),
                                            transformed_layer.to("cpu")) 
                    summed_weighted_layer = weighted_layer.sum(dim=1)  # sum over heads
                    '''
                    weighted_layer = torch.einsum('bhks,bhsd->bhksd', attn_output_weights,
                                            transformed_layer) 
                    summed_weighted_layer = weighted_layer.sum(dim=1)  # sum over heads
                    ## my own
                    '''
                    weighted_layer = torch.einsum('bhcr, bhre ->bhcre', attn_output_weights, y)
                    weighted_layer = torch.einsum('bhkrv, dhv -> bhkrd', weighted_layer, headmixing_weight)
                    summed_weighted_layer = weighted_layer.sum(dim = 1)'''
                    # Make residual matrix (batch, seq_length, seq_length, all_head_size)
                    
                    hidden_shape = attn_output.size() 
                    #print('hidden = ', hidden_shape) # (seq_length, batch, all_head_size)
                    device = attn_output.device
                    
                    accumulated_bias = self.out_proj.bias
                    residual_weighted_layer = summed_weighted_layer.to(device)

                    #b = self.bias_decomposer(accumulated_bias, b, bias_decomp_type)
                    #print(b.shape)
                    '''
                    residual = torch.einsum('sk,bsd->bskd', torch.eye(hidden_shape[1]).to(device),
                                            attn_output.transpose(1, 0))  # diagonal representations (hidden states)
                    '''
                else:
                    transformed_layer = torch.einsum('bhsqv,dhv->bhsqd', decomposed_value_layer.to("cpu"), headmixing_weight.to("cpu"))
                    expanded_bias = torch.einsum('hv, dhv -> d',self.in_proj_bias[2 * E:].view(self.num_heads, self.head_dim), headmixing_weight)

                    weighted_layer = torch.einsum('bhks,bhsqd->bhkqd', attn_output_weights.to("cpu"),
                                            transformed_layer)  # attention_probs(Q*K^t) * V * W^o
                    '''
                    transformed_layer = torch.einsum('bhsqv,dhv->bhsqd', decomposed_value_layer, headmixing_weight)

                    weighted_layer = torch.einsum('bhks,bhsqd->bhkqd', attn_output_weights,
                                            transformed_layer)  # attention_probs(Q*K^t) * V * W^o
                    '''
                    summed_weighted_layer = weighted_layer.sum(dim=1)  # sum over heads

                    residual_weighted_layer = summed_weighted_layer.to(attribution_vectors.device)# + attribution_vectors
                    #print('attn_output_weights.shape =', attn_output_weights.shape)
                    #print('self.in_proj_bias[2 * E:].shape =', self.in_proj_bias[2 * E:].shape)
                    #print('self.out_proj.bias.shape =', self.out_proj.bias.shape)
                    #bias_weighted_layer = torch.einsum('bhks,bhsqd->bhkqd', attn_output_weights.to("cpu"),)
                    #noprint("--")
                    accumulated_bias = self.out_proj.bias + expanded_bias
                
                if decompx_config.include_biases:
                    residual_weighted_layer = self.bias_decomposer(accumulated_bias, residual_weighted_layer, bias_decomp_type)
                    device = attn_output.device
                    residual_weighted_layer = residual_weighted_layer.to(device)
                #print('attn_output.size() =', hidden_shape)
                #print(residual_weighted_layer.shape)
                #diff = max(abs(residual_weighted_layer.sum(dim = 2) - attn_output.transpose(1, 0)).flatten()) #0.0003
                #print('diff = ', diff)

                return attn_output.transpose(1, 0), attn_output_weights.to(device), value, residual_weighted_layer #B, S, S, E
            
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

    def merge_masks(self, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor],
                    query: Tensor) -> Tuple[Optional[Tensor], Optional[int]]:
        r"""
        Determine mask type and combine masks if necessary. If only one mask is provided, that mask
        and the corresponding mask type will be returned. If both masks are provided, they will be both
        expanded to shape ``(batch_size, num_heads, seq_len, seq_len)``, combined with logical ``or``
        and mask type 2 will be returned
        Args:
            attn_mask: attention mask of shape ``(seq_len, seq_len)``, mask type 0
            key_padding_mask: padding mask of shape ``(batch_size, seq_len)``, mask type 1
            query: query embeddings of shape ``(batch_size, seq_len, embed_dim)``
        Returns:
            merged_mask: merged mask
            mask_type: merged mask type (0, 1, or 2)
        """
        mask_type: Optional[int] = None
        merged_mask: Optional[Tensor] = None

        if key_padding_mask is not None:
            mask_type = 1
            merged_mask = key_padding_mask

        if attn_mask is not None:
            # In this branch query can't be a nested tensor, so it has a shape
            batch_size, seq_len, _ = query.shape
            mask_type = 2

            # Always expands attn_mask to 4D
            if attn_mask.dim() == 3:
                attn_mask_expanded = attn_mask.view(batch_size, -1, seq_len, seq_len)
            else:  # attn_mask.dim() == 2:
                attn_mask_expanded = attn_mask.view(1, 1, seq_len, seq_len).expand(batch_size, self.num_heads, -1, -1)
            merged_mask = attn_mask_expanded

            if key_padding_mask is not None:
                key_padding_mask_expanded = key_padding_mask.view(batch_size, 1, 1, seq_len).expand(-1, self.num_heads, -1, -1)
                merged_mask = attn_mask_expanded + key_padding_mask_expanded

        # no attn_mask and no key_padding_mask, returns None, None
        return merged_mask, mask_type


class Transformer(Module):
    r"""A transformer model.

    User is able to modify the attributes as needed. The architecture
    is based on the paper "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
    Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
    Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information
    Processing Systems, pages 6000-6010.

    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, encoder and decoder layers will perform LayerNorms before
            other attention and feedforward operations, otherwise after. Default: ``False`` (after).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.

    Examples::
        >>> transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
        >>> src = torch.rand((10, 32, 512))
        >>> tgt = torch.rand((20, 32, 512))
        >>> out = transformer_model(src, tgt)

    Note: A full example to apply nn.Transformer module for the word language model is available in
    https://github.com/pytorch/examples/tree/master/word_language_model
    """

    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first, norm_first,
                                                    bias, **factory_kwargs)
            encoder_norm = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first, norm_first,
                                                    bias, **factory_kwargs)
            decoder_norm = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.batch_first = batch_first

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None,
                src_is_causal: Optional[bool] = None, tgt_is_causal: Optional[bool] = None,
                memory_is_causal: bool = False) -> Tensor:
        r"""Take in and process masked source/target sequences.

        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
            src_mask: the additive mask for the src sequence (optional).
            tgt_mask: the additive mask for the tgt sequence (optional).
            memory_mask: the additive mask for the encoder output (optional).
            src_key_padding_mask: the Tensor mask for src keys per batch (optional).
            tgt_key_padding_mask: the Tensor mask for tgt keys per batch (optional).
            memory_key_padding_mask: the Tensor mask for memory keys per batch (optional).
            src_is_causal: If specified, applies a causal mask as ``src_mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``src_is_causal`` provides a hint that ``src_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            tgt_is_causal: If specified, applies a causal mask as ``tgt_mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            memory_is_causal: If specified, applies a causal mask as
                ``memory_mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

        Shape:
            - src: :math:`(S, E)` for unbatched input, :math:`(S, N, E)` if `batch_first=False` or
              `(N, S, E)` if `batch_first=True`.
            - tgt: :math:`(T, E)` for unbatched input, :math:`(T, N, E)` if `batch_first=False` or
              `(N, T, E)` if `batch_first=True`.
            - src_mask: :math:`(S, S)` or :math:`(N\cdot\text{num\_heads}, S, S)`.
            - tgt_mask: :math:`(T, T)` or :math:`(N\cdot\text{num\_heads}, T, T)`.
            - memory_mask: :math:`(T, S)`.
            - src_key_padding_mask: :math:`(S)` for unbatched input otherwise :math:`(N, S)`.
            - tgt_key_padding_mask: :math:`(T)` for unbatched input otherwise :math:`(N, T)`.
            - memory_key_padding_mask: :math:`(S)` for unbatched input otherwise :math:`(N, S)`.

            Note: [src/tgt/memory]_mask ensures that position i is allowed to attend the unmasked
            positions. If a BoolTensor is provided, positions with ``True``
            are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.
            [src/tgt/memory]_key_padding_mask provides specified elements in the key to be ignored by
            the attention. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.

            - output: :math:`(T, E)` for unbatched input, :math:`(T, N, E)` if `batch_first=False` or
              `(N, T, E)` if `batch_first=True`.

            Note: Due to the multi-head attention architecture in the transformer model,
            the output sequence length of a transformer is same as the input sequence
            (i.e. target) length of the decoder.

            where S is the source sequence length, T is the target sequence length, N is the
            batch size, E is the feature number

        Examples:
            >>> # xdoctest: +SKIP
            >>> output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        """
        is_batched = src.dim() == 3
        if not self.batch_first and src.size(1) != tgt.size(1) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        elif self.batch_first and src.size(0) != tgt.size(0) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.size(-1) != self.d_model or tgt.size(-1) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask,
                              is_causal=src_is_causal)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask,
                              tgt_is_causal=tgt_is_causal, memory_is_causal=memory_is_causal)
        return output

    @staticmethod
    def generate_square_subsequent_mask(
            sz: int,
            device: torch.device = torch.device(torch._C._get_default_device()),  # torch.device('cpu'),
            dtype: torch.dtype = torch.get_default_dtype(),
    ) -> Tensor:
        r"""Generate a square causal mask for the sequence.

        The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
        """
        return _generate_square_subsequent_mask(sz, dtype=dtype, device=device)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class TransformerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers.

    Users can build the BERT(https://arxiv.org/abs/1810.04805) model with corresponding parameters.

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``True`` (enabled).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """

    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        # this attribute saves the value providedat object construction
        self.enable_nested_tensor = enable_nested_tensor
        # this attribute controls whether nested tensors are used
        self.use_nested_tensor = enable_nested_tensor
        self.mask_check = mask_check

        enc_layer = "encoder_layer"
        why_not_sparsity_fast_path = ''
        if not isinstance(encoder_layer, torch.nn.TransformerEncoderLayer):
            why_not_sparsity_fast_path = f"{enc_layer} was not TransformerEncoderLayer"
        elif encoder_layer.norm_first :
            why_not_sparsity_fast_path = f"{enc_layer}.norm_first was True"
        elif not encoder_layer.self_attn.batch_first:
            why_not_sparsity_fast_path = (f"{enc_layer}.self_attn.batch_first was not True" +
                                          "(use batch_first for better inference performance)")
        elif not encoder_layer.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = f"{enc_layer}.self_attn._qkv_same_embed_dim was not True"
        elif not encoder_layer.activation_relu_or_gelu:
            why_not_sparsity_fast_path = f"{enc_layer}.activation_relu_or_gelu was not True"
        elif not (encoder_layer.norm1.eps == encoder_layer.norm2.eps) :
            why_not_sparsity_fast_path = f"{enc_layer}.norm1.eps was not equal to {enc_layer}.norm2.eps"
        elif encoder_layer.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = f"{enc_layer}.self_attn.num_heads is odd"

        if enable_nested_tensor and why_not_sparsity_fast_path:
            warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
            self.use_nested_tensor = False


    def forward(
            self,
            src: Tensor,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: Optional[bool] = None, 
            attribution_vectors : Optional[Tensor] = None,
            transformer_explainer = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``is_causal`` provides a hint that ``mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in Transformer class.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype
        )

        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        src_key_padding_mask_for_layers = src_key_padding_mask
        why_not_sparsity_fast_path = ''
        str_first_layer = "self.layers[0]"
        batch_first = first_layer.self_attn.batch_first
        if not hasattr(self, "use_nested_tensor"):
            why_not_sparsity_fast_path = "use_nested_tensor attribute not present"
        elif not self.use_nested_tensor:
            why_not_sparsity_fast_path = "self.use_nested_tensor (set in init) was not True"
        elif first_layer.training:
            why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif src_key_padding_mask is None:
            why_not_sparsity_fast_path = "src_key_padding_mask was None"
        elif (((not hasattr(self, "mask_check")) or self.mask_check)
                and not torch._nested_tensor_from_mask_left_aligned(src, src_key_padding_mask.logical_not())):
            why_not_sparsity_fast_path = "mask_check enabled, and src and src_key_padding_mask was not left aligned"
        elif output.is_nested:
            why_not_sparsity_fast_path = "NestedTensor input is not supported"
        elif mask is not None:
            why_not_sparsity_fast_path = "src_key_padding_mask and mask were both supplied"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"

        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                first_layer.self_attn.in_proj_weight,
                first_layer.self_attn.in_proj_bias,
                first_layer.self_attn.out_proj.weight,
                first_layer.self_attn.out_proj.bias,
                first_layer.norm1.weight,
                first_layer.norm1.bias,
                first_layer.norm2.weight,
                first_layer.norm2.bias,
                first_layer.linear1.weight,
                first_layer.linear1.bias,
                first_layer.linear2.weight,
                first_layer.linear2.bias,
            )
            _supported_device_type = ["cpu", "cuda", torch.utils.backend_registration._privateuse1_backend_name]
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif src.device.type not in _supported_device_type:
                why_not_sparsity_fast_path = f"src device is neither one of {_supported_device_type}"
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if (not why_not_sparsity_fast_path) and (src_key_padding_mask is not None):
                convert_to_nested = True
                output = torch._nested_tensor_from_mask(output, src_key_padding_mask.logical_not(), mask_check=False)
                src_key_padding_mask_for_layers = None

        
        seq_len = _get_seq_len(src, batch_first)
        is_causal = _detect_is_causal_mask(mask, is_causal, seq_len)
        #print('seq_len = ', seq_len)
        store = False
        #print('src_key_padding_mask = \n', src_key_padding_mask) #None
        #print('mask = \n', mask) #None
        for mod in self.layers:
            if transformer_explainer == None:
                output, _ = mod(output, src_mask=mask, is_causal=is_causal, src_key_padding_mask=src_key_padding_mask_for_layers, attribution_vectors = attribution_vectors, transformer_explainer = transformer_explainer)
            elif transformer_explainer.word in ["rollout"]:
                output, attn_matrix = mod(output, src_mask=mask, is_causal=is_causal, src_key_padding_mask=src_key_padding_mask_for_layers, attribution_vectors = attribution_vectors, transformer_explainer = transformer_explainer)
                if not store:
                    store_out = attn_matrix
                    store = True
                else:
                    store_out = torch.bmm(attn_matrix, store_out)
            elif transformer_explainer.word in ["globenc", "alti"]:
                if store == True:
                    device = output.device
                    hidden_shape = output.size()
                    attribution_vectors = torch.einsum('sk,bsd->bskd', torch.eye(hidden_shape[1]).to(device), output)  # diagonal representations (hidden states)
                output, attn_matrix = mod(output, src_mask=mask, is_causal=is_causal, src_key_padding_mask=src_key_padding_mask_for_layers, attribution_vectors = attribution_vectors, transformer_explainer = transformer_explainer)
                if not store:
                    store_out = attn_matrix
                    store = True
                else:
                    store_out = torch.bmm(attn_matrix, store_out)
            elif transformer_explainer.word == "decompx":
                output, attribution_vectors = mod(output, src_mask=mask, is_causal=is_causal, src_key_padding_mask=src_key_padding_mask_for_layers, attribution_vectors = attribution_vectors, transformer_explainer = transformer_explainer)
            

        if convert_to_nested:
            output = output.to_padded_tensor(0., src.size())

        if self.norm is not None:
            output = self.norm(output)

        if transformer_explainer == None:
            return output, None
        elif transformer_explainer.word in ["rollout", "globenc", "alti"]:
            #print(store_out.shape)
            return output, store_out
        elif transformer_explainer.word == "decompx":
            return output, attribution_vectors
            


class TransformerDecoder(Module):
    r"""TransformerDecoder is a stack of N decoder layers.

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """

    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, tgt_is_causal: Optional[bool] = None,
                memory_is_causal: bool = False) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            memory_is_causal: If specified, applies a causal mask as
                ``memory mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        seq_len = _get_seq_len(tgt, self.layers[0].self_attn.batch_first)
        tgt_is_causal = _detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask,
                         tgt_is_causal=tgt_is_causal,
                         memory_is_causal=memory_is_causal)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.

    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    TransformerEncoderLayer can handle either traditional torch.tensor inputs,
    or Nested Tensor inputs.  Derived classes are expected to similarly accept
    both input formats.  (Not all combinations of inputs are currently
    supported by TransformerEncoderLayer while Nested Tensor is in prototype
    state.)

    If you are implementing a custom layer, you may derive it either from
    the Module or TransformerEncoderLayer class.  If your custom layer
    supports both torch.Tensors and Nested Tensors inputs, make its
    implementation a derived class of TransformerEncoderLayer. If your custom
    Layer supports only torch.Tensor inputs, derive its implementation from
    Module.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectively. Otherwise it's done after. Default: ``False`` (after).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)

    Fast path:
        forward() will use a special optimized implementation described in
        `FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`_ if all of the following
        conditions are met:

        - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor
          argument ``requires_grad``
        - training is disabled (using ``.eval()``)
        - batch_first is ``True`` and the input is batched (i.e., ``src.dim() == 3``)
        - activation is one of: ``"relu"``, ``"gelu"``, ``torch.functional.relu``, or ``torch.functional.gelu``
        - at most one of ``src_mask`` and ``src_key_padding_mask`` is passed
        - if src is a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_, neither ``src_mask``
          nor ``src_key_padding_mask`` is passed
        - the two ``LayerNorm`` instances have a consistent ``eps`` value (this will naturally be the case
          unless the caller has manually modified one without modifying the other)

        If the optimized implementation is in use, a
        `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be
        passed for ``src`` to represent padding more efficiently than using a padding
        mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ will be
        returned, and an additional speedup proportional to the fraction of the input that
        is padding can be expected.

        .. _`FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`:
         https://arxiv.org/abs/2205.14135

    """

    __constants__ = ['norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout,
                                            bias=bias, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu


    def ln_decomposer(self, attribution_vectors, pre_ln_states, gamma, beta, eps, include_biases=True, bias_decomp_type="absdot"):
            mean = pre_ln_states.mean(-1, keepdim=True)  # (batch, seq_len, 1) m(y=Σy_j)
            var = (pre_ln_states - mean).pow(2).mean(-1, keepdim=True).unsqueeze(dim=2)  # (batch, seq_len, 1, 1)  s(y)

            each_mean = attribution_vectors.mean(-1, keepdim=True)  # (batch, seq_len, seq_len, 1) m(y_j)

            normalized_layer = torch.div(attribution_vectors - each_mean,
                                            (var + eps) ** (1 / 2))  # (batch, seq_len, seq_len, all_head_size)

            post_ln_layer = torch.einsum('bskd,d->bskd', normalized_layer,
                                            gamma)  # (batch, seq_len, seq_len, all_head_size)
            
            if include_biases:
                return self.bias_decomposer(beta, post_ln_layer, bias_decomp_type=bias_decomp_type).cuda()
            else:
                return post_ln_layer 
    def bias_decomposer(self, bias, attribution_vectors, bias_decomp_type="absdot"):
        #print('say hi to bias')
        # Decomposes the input bias based on similarity to the attribution vectors
        # Args:
        #   bias: a bias vector (all_head_size)
        #   attribution_vectors: the attribution vectors from token j to i (b, i, j, all_head_size) :: (batch, seq_length, seq_length, all_head_size) 

        if bias_decomp_type == "absdot":
            weights = torch.abs(torch.einsum("bskd,d->bsk", attribution_vectors, bias))
        elif bias_decomp_type == "abssim":
            weights = torch.abs(torch.nn.functional.cosine_similarity(attribution_vectors, bias, dim=-1))
            weights = (torch.norm(attribution_vectors, dim=-1) != 0) * weights
        elif bias_decomp_type == "norm":
            weights = torch.norm(attribution_vectors, dim=-1)
        elif bias_decomp_type == "equal":
            weights = (torch.norm(attribution_vectors, dim=-1) != 0) * 1.0
        elif bias_decomp_type == "cls":
            weights = torch.zeros(attribution_vectors.shape[:-1], device=attribution_vectors.device)
            weights[:,:,0] = 1.0
        elif bias_decomp_type == "dot":
            weights = torch.einsum("bskd,d->bsk", attribution_vectors, bias)
        elif bias_decomp_type == "biastoken":
            attrib_shape = attribution_vectors.shape
            if attrib_shape[1] == attrib_shape[2]:
                attribution_vectors = torch.concat([attribution_vectors, torch.zeros((attrib_shape[0], attrib_shape[1], 1, attrib_shape[3]), device=attribution_vectors.device)], dim=-2)
            attribution_vectors[:,:,-1] = attribution_vectors[:,:,-1] + bias
            return attribution_vectors

        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-12)
        weighted_bias = torch.matmul(weights.unsqueeze(dim=-1), bias.unsqueeze(dim=0))
        return attribution_vectors.cpu() + weighted_bias.cpu()
    
    def gelu_linear_approximation(self, intermediate_hidden_states, intermediate_output):
        def phi(x):
            return (1 + torch.erf(x / math.sqrt(2))) / 2.
        
        def normal_pdf(x):
            return torch.exp(-(x**2) / 2) / math.sqrt(2. * math.pi)

        def gelu_deriv(x):
            return phi(x)+x*normal_pdf(x)
        
        m = gelu_deriv(intermediate_hidden_states)
        b = intermediate_output - m * intermediate_hidden_states
        return m, b


    def gelu_decomposition(self, attribution_vectors, intermediate_hidden_states, intermediate_output, bias_decomp_type):
        m, b = self.gelu_linear_approximation(intermediate_hidden_states, intermediate_output)
        mx = attribution_vectors * m.unsqueeze(dim=-2)

        if bias_decomp_type == "absdot":
            weights = torch.abs(torch.einsum("bskl,bsl->bsk", mx, b))
        elif bias_decomp_type == "abssim":
            weights = torch.abs(torch.nn.functional.cosine_similarity(mx, b))
            weights = (torch.norm(mx, dim=-1) != 0) * weights
        elif bias_decomp_type == "norm":
            weights = torch.norm(mx, dim=-1)
        elif bias_decomp_type == "equal":
            weights = (torch.norm(mx, dim=-1) != 0) * 1.0
        elif bias_decomp_type == "cls":
            weights = torch.zeros(mx.shape[:-1], device=mx.device)
            weights[:,:,0] = 1.0

        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-12)
        weighted_bias = torch.einsum("bsl,bsk->bskl", b, weights)
        return mx + weighted_bias

    def gelu_zo_decomposition(self, attribution_vectors, intermediate_hidden_states, intermediate_output):
        m = intermediate_output / (intermediate_hidden_states + 1e-12)
        
        
        #print('attribution_vectors.shape = ', attribution_vectors.shape)
        
        #bias_removed_mask = torch.ones_like(attribution_vectors)
        '''
        pos_mask = intermediate_hidden_states > 0
        not_pos_mask = intermediate_hidden_states <= 0
        pos_attr_part_mask = (pos_mask.unsqueeze(dim = -2) * attribution_vectors) > 0
        pos_attr_part = attribution_vectors * pos_attr_part_mask

        
        not_pos_attr_part_mask = (not_pos_mask.unsqueeze(dim = -2) * attribution_vectors) <= 0
        not_pos_attr_part = attribution_vectors * not_pos_attr_part_mask
        attribution_vectors = pos_attr_part + not_pos_attr_part
        
        intermediate_hidden_states = attribution_vectors.sum(dim = -2)
        m = intermediate_output / (torch.pow(intermediate_hidden_states, 1) + 1e-12)
        '''
        #print(bias_removed_mask[0, 0, :, 0])
        #print('gelu_decomposer in RobertaLayer: norm for each attribution vectors')
        #print('before: ')
        #print(torch.norm(attribution_vectors[:2, 0], dim=-1))
        mx = attribution_vectors * m.unsqueeze(dim=-2)
        #print(sum(mx[0, 0, :, 0]))
        #mx = mx * bias_removed_mask
        #print(sum(mx[0, 0, :, 0]))
        
        #print('after:')
        #print(torch.norm(attribution_vectors[:2, 0], dim=-1))
        return mx
    
    def ffn_decomposer(self, attribution_vectors, intermediate_hidden_states, intermediate_output, include_biases=True, approximation_type="GeLU_LA", bias_decomp_type="absdot"):
        
        #post_first_layer = torch.einsum("ld,bskd->bskl", self.linear1.weight.to("cpu"), attribution_vectors.to("cpu"))
        post_first_layer = torch.einsum("ld,bskd->bskl", self.linear1.weight, attribution_vectors)
        #post_first_layer = post_first_layer.to(self.linear1.bias.device)
        if include_biases:
            post_first_layer = self.bias_decomposer(self.linear1.bias, post_first_layer, bias_decomp_type=bias_decomp_type)
            post_first_layer = post_first_layer.to(self.linear1.bias.device)

        if approximation_type == "ReLU":
            mask_for_gelu_approx = (intermediate_hidden_states > 0)
            post_act_first_layer = torch.einsum("bskl, bsl->bskl", post_first_layer, mask_for_gelu_approx)
            post_act_first_layer = post_first_layer * mask_for_gelu_approx.unsqueeze(dim=-2)
        elif approximation_type == "GeLU_LA":
            post_act_first_layer = self.gelu_decomposition(post_first_layer, intermediate_hidden_states, intermediate_output, bias_decomp_type=bias_decomp_type)
        elif approximation_type == "GeLU_ZO":
            post_act_first_layer = self.gelu_zo_decomposition(post_first_layer, intermediate_hidden_states, intermediate_output)

        post_second_layer = torch.einsum("bskl, dl->bskd", post_act_first_layer, self.linear2.weight)
        #post_second_layer = torch.einsum("bskl, dl->bskd", post_act_first_layer.to("cpu"), self.linear2.weight.to("cpu"))
        #torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
        #post_second_layer = post_second_layer.to(self.linear2.bias.device)
        if include_biases:
            post_second_layer = self.bias_decomposer(self.linear2.bias, post_second_layer, bias_decomp_type=bias_decomp_type)
            post_second_layer = post_second_layer.to(self.linear2.bias.device)
        
        
        return post_second_layer
    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False, 
            attribution_vectors = None,
            transformer_explainer = None
            )-> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``src mask``.
                Default: ``False``.
                Warning:
                ``is_causal`` provides a hint that ``src_mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in Transformer class.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        why_not_sparsity_fast_path = ''
        if not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif self.training:
            why_not_sparsity_fast_path = "training is enabled"
        elif not self.self_attn.batch_first :
            why_not_sparsity_fast_path = "self_attn.batch_first was not True"
        elif not self.self_attn._qkv_same_embed_dim :
            why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
        elif not self.activation_relu_or_gelu:
            why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
        elif not (self.norm1.eps == self.norm2.eps):
            why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
        elif src.is_nested and (src_key_padding_mask is not None or src_mask is not None):
            why_not_sparsity_fast_path = "neither src_key_padding_mask nor src_mask are not supported with NestedTensor input"
        elif self.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"
        #print(why_not_sparsity_fast_path)
        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )

            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            _supported_device_type = ["cpu", "cuda", torch.utils.backend_registration._privateuse1_backend_name]
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not all((x.device.type in _supported_device_type) for x in tensor_args):
                why_not_sparsity_fast_path = ("some Tensor argument's device is neither one of "
                                              f"{_supported_device_type}")
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")
            #print('why_not_sparsity_fast_path: ', why_not_sparsity_fast_path)
            if not why_not_sparsity_fast_path and transformer_explainer == None:
                merged_mask, mask_type = self.self_attn.merge_masks(src_mask, src_key_padding_mask, src)
                return torch._transformer_encoder_layer_fwd(
                    src,
                    self.self_attn.embed_dim,
                    self.self_attn.num_heads,
                    self.self_attn.in_proj_weight,
                    self.self_attn.in_proj_bias,
                    self.self_attn.out_proj.weight,
                    self.self_attn.out_proj.bias,
                    self.activation_relu_or_gelu == 2,
                    self.norm_first,
                    self.norm1.eps,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.linear1.weight,
                    self.linear1.bias,
                    self.linear2.weight,
                    self.linear2.bias,
                    merged_mask,
                    mask_type,
                ), None
        x = src 
        if transformer_explainer == None or transformer_explainer.word not in ["decompx", "globenc", "alti"]:
            if self.norm_first:
                x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
                x = x + self._ff_block(self.norm2(x))
            else: 
                #
                if transformer_explainer == None:
                    x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
                    x = self.norm2(x + self._ff_block(x))  
                elif transformer_explainer.method["rollout"]:
                    device = x.device
                    #print('rollout =', rollout)
                    sa_x, attn_matrix = self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal, transformer_explainer=transformer_explainer)
                    x = self.norm1(x + sa_x)
                    x = self.norm2(x + self._ff_block(x)) 
                    
                    redefined_attn_matrix = 0.5 * (attn_matrix + torch.eye(attn_matrix.shape[1]).to(device))
                    #print('redefined shape ', redefined_attn_matrix.shape)
                    return x, redefined_attn_matrix 
                                     
            return x, None
        else:
            decompx = transformer_explainer.method[transformer_explainer.word]
            
            #print('decompx =', decompx)
            device = x.device
            hidden_shape = x.size()
            residual = torch.einsum('sk,bsd->bskd', torch.eye(hidden_shape[1]).to(device), x)  # diagonal representations (hidden states)
            if decompx.bias_decomp_type == "biastoken":
                residual = torch.cat((residual, torch.zeros(hidden_shape[0], hidden_shape[1], 1, hidden_shape[2]).to(device)), dim = 2)
            if self.norm_first: #not repaired
                x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
                x = x + self._ff_block(self.norm2(x))
            else: 
                sa_to_re, de_x = self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal, attribution_vectors=attribution_vectors, transformer_explainer=transformer_explainer)
                
                pre_state = x + sa_to_re
                x = self.norm1(pre_state)
                de_x = de_x + residual
                if decompx.include_LN1:
                    de_x = self.ln_decomposer(de_x, pre_state, self.norm1.weight, self.norm1.bias, self.norm1.eps, include_biases=decompx.include_biases,
                                            bias_decomp_type=decompx.bias_decomp_type)
                #start here
                if transformer_explainer.method["alti"] != None:
                    sequence_length = x.shape[1]
                    x_expand = x.unsqueeze(-2).expand(-1, -1, sequence_length, -1)
                    #print('x_expand =', x_expand[0, :3, :5, :5])
                    #print('de_x =', de_x[0, :3, :5, :5])
                    importance_matrix = -F.pairwise_distance(de_x, x_expand, p=1)
                    x_norm = torch.norm(x, dim = -1, p = 1).unsqueeze(-1).expand(-1, -1, sequence_length)
                    #print('x_norm', x_norm[0, :5, :5])
                    #print(x_norm.shape, importance_matrix.shape)
                    summation = x_norm + importance_matrix
                    #print('summation = ', summation[0, :5, :5])
                    summation = torch.clamp(summation, min = 0)
                    d_importance_matrix = summation/(summation.sum(keepdim=True, dim = -1))

                    return x, d_importance_matrix
                
                ff_to_re, de_x_after_ff  = self._ff_block(x, decompx_config=decompx, attribution_vectors=de_x)
                pre_state = x + ff_to_re
                x = self.norm2(pre_state) 
                
                
                if transformer_explainer.method["globenc"] != None:
                    de_x = self.ln_decomposer(de_x, pre_state, self.norm2.weight, self.norm2.bias, self.norm2.eps, include_biases=decompx.include_biases,
                                            bias_decomp_type=decompx.bias_decomp_type)
                    return x, torch.norm(de_x, dim = -1)
                
                de_x = de_x + de_x_after_ff 

                if decompx.include_LN2:
                    de_x = self.ln_decomposer(de_x, pre_state, self.norm2.weight, self.norm2.bias, self.norm2.eps, include_biases=decompx.include_biases,
                                            bias_decomp_type=decompx.bias_decomp_type)
                #diff = max(abs(de_x.sum(dim = 2) - x).flatten()) 
                #print('diff = ', diff)
                #print('de_x.device =', de_x.device)
                
                torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
                return x, de_x
            return x, None
            

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False, attribution_vectors = None, transformer_explainer = None) -> Tensor:
        if transformer_explainer == None:
            x, *args = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False, is_causal=is_causal)
            #print('x.size(), ', x.size())
            return self.dropout1(x)
        elif transformer_explainer.word in ["decompx", "globenc", "alti"]:
            decompx = transformer_explainer.method[transformer_explainer.word]
            #print(x.size())
            x, *args = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True, is_causal=is_causal, decompx_config = decompx, average_attn_weights = False, attribution_vectors = attribution_vectors)
            #print('after', x.size())
            return self.dropout1(x), args[-1] #B, S, S, E
        elif transformer_explainer.word == "rollout":
            x, *args = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True, is_causal=is_causal, attribution_vectors = attribution_vectors)
            a = args[0].sum(dim = -1)
            return self.dropout1(x), args[0] #attention_average_weight
            

    # feed forward block
    def _ff_block(self, x: Tensor, decompx_config: Optional[DecompXConfig] = None, attribution_vectors = None) -> Tensor:
        if decompx_config == None:
            x = self.linear2(self.dropout(self.activation(self.linear1(x))))
            return self.dropout2(x)
        elif decompx_config.include_FFN:
            pre_act_hidden_states = self.linear1(x)
            intermediate_output = self.activation(pre_act_hidden_states)
            x = self.linear2(self.dropout(intermediate_output))
            post_ffn_layer = self.ffn_decomposer(attribution_vectors=attribution_vectors,
                    intermediate_hidden_states=pre_act_hidden_states,
                    intermediate_output=intermediate_output,
                    approximation_type=decompx_config.FFN_approx_type,
                    include_biases=decompx_config.include_biases,
                    bias_decomp_type=decompx_config.bias_decomp_type
                )
            return x, post_ffn_layer
        else:
            return x, attribution_vectors
        


class TransformerDecoderLayer(Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.

    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectively. Otherwise it's done after.
            Default: ``False`` (after).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)

    Alternatively, when ``batch_first`` is ``True``:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> memory = torch.rand(32, 10, 512)
        >>> tgt = torch.rand(32, 20, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    __constants__ = ['norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            bias=bias, **factory_kwargs)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 bias=bias, **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``False``.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            memory_is_causal: If specified, applies a causal mask as
                ``memory mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal))
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           is_causal=is_causal,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                is_causal=is_causal,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


def _detect_is_causal_mask(
        mask: Optional[Tensor],
        is_causal: Optional[bool] = None,
        size: Optional[int] = None,
) -> bool:
    """Return whether the given attention mask is causal.

    Warning:
    If ``is_causal`` is not ``None``, its value will be returned as is.  If a
    user supplies an incorrect ``is_causal`` hint,

    ``is_causal=False`` when the mask is in fact a causal attention.mask
       may lead to reduced performance relative to what would be achievable
       with ``is_causal=True``;
    ``is_causal=True`` when the mask is in fact not a causal attention.mask
       may lead to incorrect and unpredictable execution - in some scenarios,
       a causal mask may be applied based on the hint, in other execution
       scenarios the specified mask may be used.  The choice may not appear
       to be deterministic, in that a number of factors like alignment,
       hardware SKU, etc influence the decision whether to use a mask or
       rely on the hint.
    ``size`` if not None, check whether the mask is a causal mask of the provided size
       Otherwise, checks for any causal mask.
    """
    # Prevent type refinement
    make_causal = (is_causal is True)

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype)

        # Do not use `torch.equal` so we handle batched masks by
        # broadcasting the comparison.
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal
