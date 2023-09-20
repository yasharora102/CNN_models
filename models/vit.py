import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(512, 2048)
        self.relu_act = nn.ReLU()
        self.linear_2 = nn.Linear(2048, 512)
        self.layer_norm = nn.LayerNorm(512)

    def forward(self, x):
        out = self.linear_1(x)
        out = self.relu_act(out)
        out = self.linear_2(out)
        out = self.layer_norm(out)
        out = out + x
        return out


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()

        self.d_k = d_k

    def forward(self, q, k, v):
        dot_product_attention = torch.matmul(q, k.transpose(0, 1)) / math.sqrt(self.d_k)

        # Add MASKING

        dot_product_attention = F.softmax(dot_product_attention)
        dot_product_attention = torch.matmul(dot_product_attention, v)

        return dot_product_attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_k=64, d_v=64, h=8):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.q_linear = nn.Linear(512, h * d_k)
        self.v_linear = nn.Linear(512, h * d_v)
        self.k_linear = nn.Linear(512, h * d_k)

        self.scaled_dot_product_attention = ScaledDotProductAttention(self.d_k)
        self.layer_norm = nn.LayerNorm(512)

    def forward(self, x):
        attention = self.scaled_dot_product_attention()

        return None


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention()
        self.feed_forward = FeedForward()

    def forward(self, encoder_input):
        encoder_out = self.multi_head_attention(encoder_input)
        encoder_out = self.feed_forward(encoder_out)

        return encoder_out


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.multi_head_attention_dec = MultiHeadAttention()
        self.multi_head_attention_enc = MultiHeadAttention()
        self.feed_forward = FeedForward()

    def forward(self, decoder_in, encoder_out):
        dec_out = self.multi_head_attention_dec(decoder_in)
        dec_out = self.multi_head_attention_enc(encoder_out)
        dec_out = self.feed_forward(dec_out)

        return dec_out
