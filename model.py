# coding: utf-8
import torch
from torch import nn
from modules import Conv1d, Conv1dGLU, Embedding, Linear, expand_speaker_embed
from torch.nn import functional as F
import math
import torch.nn.init as init

class EVCModel(nn.Module):
    """Attention seq2seq model + post processing network
    """
    def __init__(self, GST, NEU2EMO, MEL2LIN, mel_dim=80, linear_dim=513,
                 n_speakers=1, speaker_embed_dim=16, speaker_embedding_weight_std=0.01):
        super(EVCModel, self).__init__()
        self.gst = GST
        self.seq2seq = NEU2EMO
        self.postnet = MEL2LIN  # referred as "Converter" in DeepVoice3
        self.mel_dim = mel_dim
        self.linear_dim = linear_dim
        #self.fc = Linear(mel_dim, gru_unit)
        if n_speakers > 1 :
            self.embed_speakers = Embedding(n_speakers, speaker_embed_dim, padding_idx=None, std=speaker_embedding_weight_std)
        self.n_speakers = n_speakers
        self.speaker_embed_dim = speaker_embed_dim
    def make_generation_fast_(self):

        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return
        self.apply(remove_weight_norm)


    def forward(self, ling, mel=None, speaker_ids=None):
        # Apply seq2seq
        # (B, T, mel_dim)
        if speaker_ids is not None:
            assert self.n_speakers > 1
            speaker_embed = self.embed_speakers(speaker_ids)
            speaker_embed = speaker_embed.unsqueeze(1)
        else:
            speaker_embed = None
        if self.gst is not None:
            style_embed = self.gst(mel)
        else:
            style_embed = None
        mel_outputs, decoder_states = self.seq2seq(ling, style_embed, speaker_embed)
        if self.postnet is not None:
            linear_outputs = self.postnet(decoder_states, style_embed, speaker_embed)
        else:
            linear_outputs= None

        return style_embed, mel_outputs, linear_outputs



class NEU2EMO(nn.Module):
    def __init__(self, in_dim=342, out_dim=80, style_embed_dim=256, convolutions=((128, 5, 1),) * 4,
                 dropout=0.1, speaker_embedding=True, style_embedding=True, speaker_embed_dim=16):
        super(NEU2EMO, self).__init__()
        self.dropout = dropout
        self.in_dim = in_dim
        self.style_embed_dim = style_embed_dim
        self.speaker_embed_dim = speaker_embed_dim
        self.speaker_embedding = speaker_embedding
        self.style_embedding = style_embedding
        # Prenet: causal convolution blocks
        self.convolutions = nn.ModuleList()
        in_channels = in_dim
        std_mul = 1.0
        self.input_proj = Linear(in_dim, convolutions[0][0])

        for out_channels, kernel_size, dilation in convolutions:
            if in_channels != out_channels:
                # Conv1d + ReLU
                self.convolutions.append(
                    Conv1d(in_channels, out_channels, kernel_size=1, padding=0, dilation=1, std_mul=std_mul))
                self.convolutions.append(nn.ReLU(inplace=True))
                in_channels = out_channels
                std_mul = 2.0
            self.convolutions.append(
                Conv1dGLU(in_channels, out_channels, kernel_size, causal=True, dilation=dilation, dropout=dropout,
                          std_mul=std_mul, residual=True, style_embed_dim=style_embed_dim, speaker_embed_dim=speaker_embed_dim,
                          style_embedding=style_embedding, speaker_embedding=speaker_embedding))
            in_channels = out_channels
            std_mul = 4.0
        """
        for out_channels, kernel_size, dilation in encoder:
            if in_channels != out_channels:
                # Conv1d + ReLU
                self.preattention.append(
                    Conv1d(in_channels, out_channels, kernel_size=1, padding=0, dilation=1, std_mul=std_mul))
                self.preattention.append(nn.ReLU(inplace=True))
                in_channels = out_channels
                std_mul = 2.0
            self.preattention.append(
                Conv1dGLU(in_channels, out_channels, kernel_size, causal=True, dilation=dilation, dropout=dropout,
                          std_mul=std_mul, residual=True, style_embed_dim=style_embed_dim, speaker_embed_dim=speaker_embed_dim,
                          style_embedding=True, speaker_embedding=True))
            in_channels = out_channels
            std_mul = 4.0

        # Causal convolution blocks + attention layers
        self.decoder = nn.ModuleList()

        for i, (out_channels, kernel_size, dilation) in enumerate(decoder):
            assert in_channels == out_channels
            self.decoder.append(
                Conv1dGLU(in_channels, out_channels, kernel_size, causal=True, dilation=dilation, dropout=dropout,
                          std_mul=std_mul, residual=False, style_embed_dim=style_embed_dim, speaker_embed_dim=speaker_embed_dim,
                          style_embedding=True, speaker_embedding=True))
            in_channels = out_channels
            std_mul = 4.0
        """
        # Last 1x1 convolution
        self.last_conv = Conv1d(in_channels, out_dim, kernel_size=1,
                                padding=0, dilation=1, std_mul=std_mul,
                                dropout=dropout)

    def forward(self, inputs, style_embed=None, speaker_embed=None):
        # Grouping multiple frames if necessary
        assert inputs.size(-1) == self.in_dim
        x = self.input_proj(inputs)
        x = F.dropout(x, p=self.dropout, training=self.training)


        if speaker_embed is None:
            speaker_embed_btc = None
        else:
            speaker_embed_btc = speaker_embed.expand(speaker_embed.size(0), x.size(1), speaker_embed.size(2))
            speaker_embed_btc = F.dropout(speaker_embed_btc, p=self.dropout, training=self.training)
        if style_embed is None:
            style_embed_btc = None
        else:
            style_embed_btc = style_embed.expand(x.size(0), x.size(1), style_embed.size(2))
            style_embed_btc = F.dropout(style_embed_btc, p=self.dropout, training=self.training)

        # Generic case: B x T x C -> B x C x T
        x = inputs.transpose(1, 2)
        # Prenet
        """
        for f in self.encoder:
            x = f(x, style_embed_btc, speaker_embed_btc) if isinstance(f, Conv1dGLU) else f(x)

        # Casual convolutions + Multi-hop attentions
        for f in self.convolutions:
            residual = x
            x = f(x, style_embed_btc) if isinstance(f, Conv1dGLU) else f(x)
            if isinstance(f, Conv1dGLU):
                x = (x + residual) * math.sqrt(0.5)
        """
        for f in self.convolutions:
            x = f(x, style_embed_btc, speaker_embed_btc) if isinstance(f, Conv1dGLU) else f(x)
        decoder_states = x.transpose(1, 2).contiguous()
        x = self.last_conv(x)

        # Back to B x T x C
        x = x.transpose(1, 2)

        # project to mel-spectorgram
        outputs = F.sigmoid(x)
        return outputs, decoder_states



class MEL2LIN(nn.Module):
    def __init__(self, in_dim=256, style_embed_dim=256, speaker_embed_dim=16, out_dim=513, convolutions=((256, 5, 1),) * 4,
                 speaker_embedding=True, style_embedding=True, dropout=0.1):
        super(MEL2LIN, self).__init__()
        self.dropout = dropout
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.style_embed_dim = style_embed_dim
        self.speaker_embed_dim = speaker_embed_dim
        self.speaker_embedding = speaker_embedding
        self.style_embedding = style_embedding
        # Non causual convolution blocks
        in_channels = convolutions[0][0]

        self.convolutions = nn.ModuleList([
            # 1x1 convolution first
            Conv1d(in_dim, in_channels, kernel_size=1, padding=0, dilation=1,
                   std_mul=1.0),
            Conv1dGLU(in_channels, in_channels, kernel_size=3, causal=False, dilation=3, dropout=dropout,
                      std_mul=4.0, residual=True, style_embed_dim=style_embed_dim,
                      speaker_embed_dim=speaker_embed_dim,
                      style_embedding=style_embedding, speaker_embedding=speaker_embedding)
        ])

        std_mul = 4.0
        for (out_channels, kernel_size, dilation) in convolutions:
            if in_channels != out_channels:
                self.convolutions.append(
                    Conv1d(in_channels, out_channels, kernel_size=1, padding=0,
                           dilation=1, std_mul=std_mul))
                self.convolutions.append(nn.ReLU(inplace=True))
                in_channels = out_channels
                std_mul = 2.0
            self.convolutions.append(
                Conv1dGLU(in_channels, out_channels, kernel_size, causal=False, dilation=dilation, dropout=dropout,
                          std_mul=std_mul, residual=True, style_embed_dim=style_embed_dim,
                          speaker_embed_dim=speaker_embed_dim,
                          style_embedding=style_embedding, speaker_embedding=speaker_embedding))
            in_channels = out_channels
            std_mul = 4.0
        # Last 1x1 convolution
        self.convolutions.append(Conv1d(in_channels, out_dim, kernel_size=1,
                                        padding=0, dilation=1, std_mul=std_mul,
                                        dropout=dropout))

    def forward(self, x, style_embed=None, speaker_embed=None):

        if speaker_embed is None:
            speaker_embed_btc = None
        else:
            speaker_embed_btc = speaker_embed.expand(speaker_embed.size(0), x.size(1), speaker_embed.size(2))
            speaker_embed_btc = F.dropout(speaker_embed_btc, p=self.dropout, training=self.training)
        if style_embed is None:
            style_embed_btc = None
        else:
            style_embed_btc = style_embed.expand(x.size(0), x.size(1), style_embed.size(2))
            style_embed_btc = F.dropout(style_embed_btc, p=self.dropout, training=self.training)

        # Generic case: B x T x C -> B x C x T
        x = x.transpose(1, 2)


        for f in self.convolutions:
            x = f(x, style_embed_btc, speaker_embed_btc) if isinstance(f, Conv1dGLU) else f(x)
        # Back to B x T x C
        x = x.transpose(1, 2)

        return F.sigmoid(x)


class GST(nn.Module):

    def __init__(self, in_dim=80, gru_unit=128, style_att_dim=128, num_gst=10, num_heads=8,
                 convolutions=((32, (3, 3), 2),) * 4):
        super().__init__()
        #self.encoder = ReferenceEncoder(in_dim=in_dim, gru_unit=gru_unit, dropout=dropout, convolutions=convolutions)
        #self.stl = STL(num_gst=num_gst, style_embed_depth=style_embed_depth, style_att_dim=style_att_dim,
        #               gru_unit=gru_unit, num_heads=num_heads)
        self.encoder = ReferenceEncoder(in_dim=in_dim, gru_unit=gru_unit, convolutions=convolutions)
        self.stl = STL(num_gst=num_gst, style_att_dim=style_att_dim, gru_unit=gru_unit, num_heads=num_heads)


    def forward(self, inputs):
        enc_out = self.encoder(inputs) # batch, gru_unit(80)
        style_embed = self.stl(enc_out)

        return style_embed


class ReferenceEncoder(nn.Module):
    '''
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    '''

    def __init__(self, in_dim=80, gru_unit=80, convolutions=((32, (3, 3), 2),) * 4):

        super().__init__()
        self.in_dim = in_dim
        K = len(convolutions)
        filters = [1] + convolutions
        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=(3, 3),
                           stride=(2, 2),
                           padding=(1, 1)) for i in range(K)]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features=convolutions[i]) for i in range(K)])

        out_channels = self.calculate_channels(in_dim, 3, 2, 1, K)
        self.gru = nn.GRU(input_size=convolutions[-1] * out_channels,
                          hidden_size=gru_unit // 2,
                          batch_first=True)

    def forward(self, inputs):
        N = inputs.size(0)
        out = inputs.view(N, 1, -1, self.in_dim)  # [N, 1, Ty, n_mels]
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        memory, out = self.gru(out)  # out --- [1, N, E//2]

        return out.squeeze(0)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class STL(nn.Module):
    '''
    inputs --- [N, E//2]
    '''

    def __init__(self, num_gst=10, style_att_dim=80, gru_unit=80, num_heads=8):

        super().__init__()
        self.embed = nn.Parameter(torch.FloatTensor(num_gst, gru_unit // num_heads))
        d_q = style_att_dim // 2
        d_k = style_att_dim// num_heads
        # self.attention = MultiHeadAttention(hp.num_heads, d_model, d_q, d_v)
        self.attention = MultiHeadAttention(query_dim=d_q, key_dim=d_k, num_units=style_att_dim, num_heads=num_heads)

        init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, inputs):
        N = inputs.size(0)
        query = inputs.unsqueeze(1)  # [N, 1, E//2]
        keys = F.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, E // num_heads]
        style_embed = self.attention(query, keys)

        return style_embed


class MultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    '''

    def __init__(self, query_dim, key_dim, num_units, num_heads):

        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)
        scores = F.softmax(scores, dim=3)

        # out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out



