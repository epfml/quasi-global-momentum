# -*- coding: utf-8 -*-
import copy
import math
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

"""training/inference utilities."""


class Mask:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=1, device=None):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad, device)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad, device):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1), device).type_as(tgt_mask.data)
        )
        return tgt_mask


def subsequent_mask(size, device):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    return (torch.from_numpy(subsequent_mask) == 0).to(device)


"""module utilities."""


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention module from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.
    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.
    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, head_count, model_dim, dropout=0.1):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None):
        """
        Compute the context vector and the attention vectors.
        Args:
           key (FloatTensor): set of `key_len`
               key vectors ``(batch, key_len, dim)``
           value (FloatTensor): set of `key_len`
               value vectors ``(batch, key_len, dim)``
           query (FloatTensor): set of `query_len`
               query vectors  ``(batch, query_len, dim)``
           mask: binary mask indicating which keys have
               non-zero attention ``(batch, query_len, key_len)``
        Returns:
           (FloatTensor, FloatTensor):
           * output context vectors ``(batch, query_len, dim)``
        """

        batch_size = key.size(0)

        def shape(x):
            """Projection."""
            return x.view(batch_size, -1, self.head_count, self.dim_per_head).transpose(
                1, 2
            )

        def unshape(x):
            """Compute context."""
            return (
                x.transpose(1, 2)
                .contiguous()
                .view(batch_size, -1, self.head_count * self.dim_per_head)
            )

        # 1) Project key, value, and query.
        key = self.linear_keys(key)
        value = self.linear_values(value)
        query = self.linear_query(query)
        # (batch, num_heads, key_len, dim_per_head)
        key = shape(key)
        value = shape(value)
        query = shape(query)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(self.dim_per_head)
        # batch x num_heads x query_len x key_len
        query_key = torch.matmul(query, key.transpose(2, 3))
        scores = query_key.float()

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            scores = scores.masked_fill(mask == 0, -1e18)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores).to(query.dtype)
        # (batch x num_heads x query_len x key_len)
        drop_attn = self.dropout(attn)

        # (batch x num_heads x query_len x dim_per_head)
        context_original = torch.matmul(drop_attn, value)
        # (batch x query_len x model_dimm)
        context = unshape(context_original)

        # (batch x query_len x model_dim)
        output = self.final_linear(context)

        return output


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.
    Args:
        d_model (int): the size of input for the first-layer of the FFN.
        d_ff (int): the hidden layer size of the second-layer
            of the FNN.
        dropout (float): dropout probability in :math:`[0, 1)`.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        """Layer definition.
        Args:
            x: ``(batch_size, input_len, model_dim)``
        Returns:
            (FloatTensor): Output ``(batch_size, input_len, model_dim)``.
        """

        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, : x.size(1)], requires_grad=False)
        return self.dropout(x)


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


"""Fundamental encoder/decoder layer, and encoder/decoder."""


class TransformerDecoderLayer(nn.Module):
    """
    Args:
      d_model (int): the dimension of keys/values/queries in
          :class:`MultiHeadedAttention`, also the input size of
          the first-layer of the :class:`PositionwiseFeedForward`.
      heads (int): the number of heads for MultiHeadedAttention.
      d_ff (int): the second-layer of the :class:`PositionwiseFeedForward`.
      dropout (float): dropout probability.
    """

    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(heads, d_model, dropout=dropout)
        self.context_attn = MultiHeadedAttention(heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, tgt_len, model_dim)``
            memory_bank (FloatTensor): ``(batch_size, src_len, model_dim)``
            src_pad_mask (LongTensor): ``(batch_size, 1, src_len)``
            tgt_pad_mask (LongTensor): ``(batch_size, 1, tgt_len, tgt_len)``
        Returns:
            (FloatTensor, FloatTensor):
            * output ``(batch_size, tgt_len, model_dim)``
        """

        input_norm = self.layer_norm_1(inputs)
        query = self.self_attn(input_norm, input_norm, input_norm, mask=tgt_pad_mask)
        query = self.drop(query) + inputs

        query_norm = self.layer_norm_2(query)
        mid = self.context_attn(memory_bank, memory_bank, query_norm, mask=src_pad_mask)
        output = self.feed_forward(self.drop(mid) + query)

        return output


class TransformerDecoder(nn.Module):
    """The Transformer decoder from "Attention is All You Need".
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`
    Args:
       num_layers (int): number of decoder layers.
       d_model (int): size of the model
       heads (int): number of heads
       d_ff (int): size of the inner FF layer
       dropout (float): dropout parameters
    """

    def __init__(self, num_layers, d_model, vocab, heads, d_ff, dropout):
        super(TransformerDecoder, self).__init__()
        self.transformer_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(d_model, heads, d_ff, dropout)
                for i in range(num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, tgt, memory_bank, src_pad_mask, tgt_pad_mask):
        for layer in self.transformer_layers:
            tgt = layer(tgt, memory_bank, src_pad_mask, tgt_pad_mask)

        output = self.layer_norm(tgt)
        return output


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.
    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, src_len, src_len)``
        Returns:
            (FloatTensor):
            * outputs ``(batch_size, src_len, model_dim)``
        """
        input_norm = self.layer_norm(inputs)
        context = self.self_attn(input_norm, input_norm, input_norm, mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class TransformerEncoder(nn.Module):
    """The Transformer encoder from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`
    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
    Returns:
        (torch.FloatTensor, torch.FloatTensor):
        * memory_bank ``(src_len, batch_size, model_dim)``
        * src_mask ``(w_batch, 1, w_len)``
    """

    def __init__(self, num_layers, d_model, heads, d_ff, dropout):
        super(TransformerEncoder, self).__init__()

        self.transformer = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, heads, d_ff, dropout)
                for i in range(num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src_embed, src_mask):
        # Run the forward pass of every layer of the tranformer.
        for layer in self.transformer:
            src_embed = layer(src_embed, src_mask)
        out = self.layer_norm(src_embed)

        return out


"""Transformer."""


class EncoderDecoder(nn.Module):
    """A standard Encoder-Decoder architecture."""

    def __init__(self, encoder, decoder, src_embed, tgt_embed, d_model, tgt_vocab_size):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.tgt_word_prj = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        output = self.decode(
            memory=self.encode(src, src_mask),
            src_mask=src_mask,
            tgt=tgt,
            tgt_mask=tgt_mask,
        )
        output = self.tgt_word_prj(output)
        return output.view(-1, output.size(2))

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


"""an entry to initialize the transformer."""

PAD_WORD = "<blank>"
UNK_WORD = "<unk>"
BOS_WORD = "<s>"
EOS_WORD = "</s>"


def make_model(
    src_vocab_size,
    tgt_vocab_size,
    share_embedding,
    num_layers=6,
    d_model=512,
    d_ff=2048,
    heads=8,
    dropout=0.1,
):
    "Construct a model from hyperparameters."
    c = copy.deepcopy
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        encoder=TransformerEncoder(num_layers, d_model, heads, d_ff, dropout),
        decoder=TransformerDecoder(
            num_layers, d_model, tgt_vocab_size, heads, d_ff, dropout
        ),
        src_embed=nn.Sequential(Embeddings(d_model, src_vocab_size), c(position)),
        tgt_embed=nn.Sequential(Embeddings(d_model, tgt_vocab_size), c(position)),
        d_model=d_model,
        tgt_vocab_size=tgt_vocab_size,
    )

    # shared embeddings (only if we have a shared vocabulary).
    if share_embedding:
        # Share the weight matrix between source & target word embeddings.
        model.src_embed[0].lut.weight = model.tgt_embed[0].lut.weight
        # Share the weight matrix between target word embedding & the final logit dense layer.
        model.tgt_word_prj.weight = model.tgt_embed[0].lut.weight

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
