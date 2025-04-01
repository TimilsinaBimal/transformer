import copy

import numpy as np
import torch
from torch import nn


def clone_modules(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# embedding layer [input]
class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512) -> None:
        super(InputEmbeddings, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)

    def forward(self, x):
        return self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32, device=x.device))


# Positional encodings
class PositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model: int, n: int = 10000) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(seq_len, d_model)
        position_matrix = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(n) / d_model))
        final_multiplier = position_matrix * div_term

        pe[:, 1::2] = torch.cos(final_multiplier)
        pe[:, 0::2] = torch.sin(final_multiplier)

        self.register_buffer("pe", pe)

    def __call__(self):
        return self.pe

    def forward(self):
        return self.pe


class DotProductAttention(nn.Module):
    def __init__(self, d_model: int, mask: bool):
        super(DotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.d_model = d_model
        self.mask = mask

    def forward(self, Q, K, V):
        x = torch.matmul(Q, torch.transpose(K, -1, -2))
        # do not transpose the batch_size
        # change position of last and second last elements for multiplication in
        # 4 dimensions
        x = x / np.sqrt(self.d_model)
        # mask here
        if self.mask:
            look_ahead_mask = torch.tril(torch.ones(x.size(), device=x.device), diagonal=0)
            x = x.masked_fill(look_ahead_mask == 0, -torch.inf)
        x = self.softmax(x)
        x = torch.matmul(x, V)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, h: int = 8, d_model: int = 512, mask: bool = False) -> None:
        super(MultiHeadAttention, self).__init__()
        if d_model % h != 0:
            raise ValueError(
                "The value of h must be divisible by d_model i.e. d_model%h == 0"
            )
        self.d_model = d_model
        self.h = h
        self.d_k = self.d_q = self.d_v = self.d_model // self.h
        self.linear_q = nn.Linear(self.d_model, self.d_model)
        self.linear_k = nn.Linear(self.d_model, self.d_model)
        self.linear_v = nn.Linear(self.d_model, self.d_model)
        self.attention = DotProductAttention(self.d_model, mask=mask)
        self.linear_o = nn.Linear(self.h * self.d_k, d_model)

    def forward(self, Q, K, V):
        batch_size = Q.size(0)
        seq_length = Q.size(1)

        # Linear transformations for Q, K, V and reshape to
        # "Batch x Heads x Seq_length x Depth"
        q_out = (
            self.linear_q(Q)
            .view(batch_size, seq_length, self.h, self.d_q)  # split into hxd dimension
            .transpose(1, 2)  # change dimension 1 to pos 2 and vice versa
            # but now change h since we want to multiply each head separately
            # and not along with other data
        )
        k_out = (
            self.linear_k(K)
            .view(batch_size, seq_length, self.h, self.d_k)
            .transpose(1, 2)
        )
        v_out = (
            self.linear_v(V)
            .view(batch_size, seq_length, self.h, self.d_v)
            .transpose(1, 2)
        )

        # Compute attention scores using all heads in parallel
        attention_output = self.attention(q_out, k_out, v_out)

        # Concatenate output of all the heads
        concat_out = (
            attention_output.transpose(1, 2)  # Now convert back, move head to
            # second last dimension since now we want to change back to
            # original position
            .contiguous().view(
                batch_size, seq_length, self.d_model
            )  # change to original size
        )

        # Final linear layer
        linear_out = self.linear_o(concat_out)
        return linear_out


class AddAndNormalization(nn.Module):
    def __init__(self, d_model: int = 512):
        super(AddAndNormalization, self).__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, residual_inp, x):
        return self.norm(torch.add(residual_inp, x))


class FeedForward(nn.Module):
    def __init__(self, d_model: int = 512, d_ff: int = 2048):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=True)
        self.linear2 = nn.Linear(d_ff, d_model, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        # ffn = relu(xW1 + b1)W2 + b2
        x = self.linear1(x)
        x = self.relu(x)
        return self.linear2(x)


class Embeddings(nn.Module):
    def __init__(self, vocab_size: int, seq_length: int, d_model: int = 512):
        super(Embeddings, self).__init__()
        self.token_embeddings = InputEmbeddings(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(
            seq_len=seq_length, d_model=d_model
        )

    def forward(self, x):
        # create embeddings
        inp_emb = self.token_embeddings(x)

        # create positional encodings
        pos_enc = self.positional_encoding()

        seq_len = inp_emb.shape[1]
        pos_enc = pos_enc[:seq_len, :].to(x.device)

        # add embs and pos encodings
        return torch.add(pos_enc, inp_emb)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int = 512, num_heads: int = 8, d_ff: int = 2048):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(h=num_heads, d_model=d_model)
        self.norm1 = AddAndNormalization(d_model=d_model)
        self.feed_forward = FeedForward(d_model=d_model, d_ff=d_ff)
        self.norm2 = AddAndNormalization(d_model=d_model)

    def forward(self, x):
        # multi head attention
        mha_out = self.multi_head_attention(x, x, x)

        # add and norm
        norm_out = self.norm1(x, mha_out)

        # feed forward
        ffn_out = self.feed_forward(norm_out)

        # add and norm
        x = self.norm2(norm_out, ffn_out)
        return x


class Encoder(nn.Module):
    def __init__(
        self, n_x: int = 6, num_heads: int = 8, d_model: int = 512, d_ff: int = 2048
    ) -> None:
        super(Encoder, self).__init__()
        self.encoder_layer = EncoderLayer(d_model, num_heads, d_ff)
        self.n_x = n_x
        self.encoder_layers = clone_modules(self.encoder_layer, self.n_x)

    def forward(self, x):
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, num_heads: int = 8, d_model: int = 512) -> None:
        super(DecoderLayer, self).__init__()
        self.masked_multi_head_attention = MultiHeadAttention(
            h=num_heads, d_model=d_model, mask=True
        )
        self.multi_head_attention = MultiHeadAttention(h=num_heads, d_model=d_model)
        self.norm1 = AddAndNormalization(d_model=d_model)
        self.norm2 = AddAndNormalization(d_model=d_model)
        self.norm3 = AddAndNormalization(d_model=d_model)
        self.feed_forward = FeedForward(d_model=d_model)

    def forward(self, x, enc_inp):
        # need to clone layers first if n_x > 1, otherwise same weight used which
        # is not correct
        # masked multi head attention
        mha_out = self.masked_multi_head_attention(x, x, x)

        # add and norm
        norm_out = self.norm1(x, mha_out)

        # multi head attention
        mha_out = self.multi_head_attention(norm_out, enc_inp, enc_inp)

        # add and norm
        norm_out = self.norm2(norm_out, mha_out)

        # feed forward
        ffn_out = self.feed_forward(norm_out)

        # add and norm
        x = self.norm3(norm_out, ffn_out)
        return x


class Decoder(nn.Module):
    def __init__(self, n_x: int = 6, num_heads: int = 8, d_model: int = 512) -> None:
        super(Decoder, self).__init__()
        self.n_x = n_x
        self.decoder_layer = DecoderLayer(num_heads=num_heads, d_model=d_model)
        self.decoder_layers = clone_modules(self.decoder_layer, self.n_x)

    def forward(self, x, enc_inp):
        for module in self.decoder_layers:
            x = module(x, enc_inp)
        return x


class Transformer(nn.Module):
    def __init__(self, config) -> None:
        super(Transformer, self).__init__()
        self.inp_embeddings = Embeddings(
            vocab_size=config.vocab_size,
            seq_length=config.seq_length,
        )
        self.out_embeddings = Embeddings(
            vocab_size=config.vocab_size,
            seq_length=config.seq_length,
        )
        self.encoder = Encoder(
            n_x=config.n_x,
            num_heads=config.num_heads,
            d_model=config.d_model,
        )
        self.decoder = Decoder(
            n_x=config.n_x,
            num_heads=config.num_heads,
            d_model=config.d_model,
        )
        self.linear = nn.Linear(config.d_model, config.vocab_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y):
        # input embeddings
        x = self.inp_embeddings(x)
        enc_out = self.encoder(x)

        # output embeddings
        y = self.out_embeddings(y)
        dec_out = self.decoder(y, enc_out)

        # linear layer
        linear_out = self.linear(dec_out)
        out = self.softmax(linear_out)
        return out
