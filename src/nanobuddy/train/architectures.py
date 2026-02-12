"""All 11 wake word model architectures as factory functions.

Each factory: (n_features: int, config: dict) -> nn.Module
The trainer adds the final classifier head (Linear → 1).

Config keys are architecture-specific. See each factory's implementation
for expected keys and their defaults.
"""

import math

import torch
import torch.nn as nn
import numpy as np


# ── Shared helpers ─────────────────────────────────────────────────

class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class _Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def _get_activation(name: str = "relu") -> nn.Module:
    return {"relu": nn.ReLU(), "gelu": nn.GELU(), "swish": _Swish()}.get(name, nn.ReLU())


# ── DNN ────────────────────────────────────────────────────────────

class _FCNBlock(nn.Module):
    def __init__(self, dim, activation):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.act = activation
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.act(self.norm(self.fc(x)))


class _DNN(nn.Module):
    def __init__(self, input_shape, layer_dim, n_blocks, emb_dim, dropout, activation):
        super().__init__()
        self.flatten = nn.Flatten()
        self.proj = nn.Linear(input_shape[0] * input_shape[1], layer_dim)
        self.act = activation
        self.norm = nn.LayerNorm(layer_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([_FCNBlock(layer_dim, activation) for _ in range(n_blocks)])
        self.out = nn.Linear(layer_dim, emb_dim)

    def forward(self, x):
        x = self.act(self.norm(self.proj(self.flatten(x))))
        x = self.dropout(x)
        for b in self.blocks:
            x = b(x)
        return self.out(x)


def build_dnn(n_features: int, cfg: dict) -> nn.Module:
    return _DNN(
        input_shape=(n_features, 96),
        layer_dim=cfg.get("layer_dim", 128),
        n_blocks=cfg.get("n_blocks", 2),
        emb_dim=cfg.get("embedding_dim", 1),
        dropout=cfg.get("dropout", 0.1),
        activation=_get_activation(cfg.get("activation", "relu")),
    )


# ── CNN ────────────────────────────────────────────────────────────

class _CNN(nn.Module):
    def __init__(self, input_shape, emb_dim, dropout, activation):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.act1 = activation
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.act2 = activation
        self.pool2 = nn.MaxPool2d(2, 2)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *input_shape)
            out = self.pool2(self.act2(self.conv2(self.pool1(self.act1(self.conv1(dummy))))))
            flat_size = int(np.prod(out.shape))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flat_size, 128)
        self.act3 = activation
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, emb_dim)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.flatten(x)
        x = self.act3(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def build_cnn(n_features: int, cfg: dict) -> nn.Module:
    return _CNN(
        input_shape=(n_features, 96),
        emb_dim=cfg.get("embedding_dim", 1),
        dropout=cfg.get("dropout", 0.1),
        activation=_get_activation(cfg.get("activation", "relu")),
    )


# ── LSTM ───────────────────────────────────────────────────────────

class _LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, emb_dim, bidir, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True,
                            bidirectional=bidir, dropout=dropout if n_layers > 1 else 0)
        fc_in = hidden_dim * 2 if bidir else hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(fc_in, emb_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))


def build_lstm(n_features: int, cfg: dict) -> nn.Module:
    return _LSTM(
        input_dim=96,
        hidden_dim=cfg.get("hidden_dim", 64),
        n_layers=cfg.get("n_layers", 2),
        emb_dim=cfg.get("embedding_dim", 1),
        bidir=cfg.get("bidirectional", True),
        dropout=cfg.get("dropout", 0.1),
    )


# ── GRU ────────────────────────────────────────────────────────────

class _GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, emb_dim, bidir, dropout):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True,
                          bidirectional=bidir, dropout=dropout if n_layers > 1 else 0)
        fc_in = hidden_dim * 2 if bidir else hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(fc_in, emb_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(self.dropout(out[:, -1, :]))


def build_gru(n_features: int, cfg: dict) -> nn.Module:
    return _GRU(
        input_dim=96,
        hidden_dim=cfg.get("hidden_dim", 64),
        n_layers=cfg.get("n_layers", 2),
        emb_dim=cfg.get("embedding_dim", 1),
        bidir=cfg.get("bidirectional", True),
        dropout=cfg.get("dropout", 0.1),
    )


# ── RNN (bidirectional LSTM variant) ──────────────────────────────

class _RNN(nn.Module):
    def __init__(self, input_dim, n_blocks, emb_dim, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 64, num_layers=n_blocks, bidirectional=True,
                            batch_first=True, dropout=dropout if n_blocks > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, emb_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1]))


def build_rnn(n_features: int, cfg: dict) -> nn.Module:
    return _RNN(
        input_dim=96,
        n_blocks=cfg.get("n_blocks", 2),
        emb_dim=cfg.get("embedding_dim", 1),
        dropout=cfg.get("dropout", 0.1),
    )


# ── Transformer ───────────────────────────────────────────────────

class _Transformer(nn.Module):
    def __init__(self, input_dim, d_model, n_head, n_layers, emb_dim, dropout):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = _PositionalEncoding(d_model, dropout=dropout)
        layer = nn.TransformerEncoderLayer(d_model, n_head, d_model * 4, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.out = nn.Linear(d_model, emb_dim)
        self.d_model = d_model

    def forward(self, x):
        x = self.input_proj(x) * math.sqrt(self.d_model)
        x = self.pos_enc(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.encoder(x)
        return self.out(x.mean(dim=1))


def build_transformer(n_features: int, cfg: dict) -> nn.Module:
    return _Transformer(
        input_dim=96,
        d_model=cfg.get("d_model", 64),
        n_head=cfg.get("n_head", 4),
        n_layers=cfg.get("n_layers", 2),
        emb_dim=cfg.get("embedding_dim", 1),
        dropout=cfg.get("dropout", 0.1),
    )


# ── TCN ────────────────────────────────────────────────────────────

class _TemporalBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_in, n_out, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(n_out, n_out, kernel_size, padding=pad, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        self.relu = nn.ReLU()
        self.pad = pad

    def forward(self, x):
        out = self.drop1(self.relu1(self.conv1(x)[:, :, : -self.pad]))
        out = self.drop2(self.relu2(self.conv2(out)[:, :, : -self.pad]))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class _TCN(nn.Module):
    def __init__(self, input_dim, channels, emb_dim, kernel_size, dropout):
        super().__init__()
        layers = []
        for i, ch in enumerate(channels):
            inch = input_dim if i == 0 else channels[i - 1]
            layers.append(_TemporalBlock(inch, ch, kernel_size, 2**i, dropout))
        self.blocks = nn.Sequential(*layers)
        self.fc = nn.Linear(channels[-1], emb_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.blocks(x)
        return self.fc(x[:, :, -1])


def build_tcn(n_features: int, cfg: dict) -> nn.Module:
    return _TCN(
        input_dim=96,
        channels=cfg.get("channels", [64, 64, 64]),
        emb_dim=cfg.get("embedding_dim", 1),
        kernel_size=cfg.get("kernel_size", 3),
        dropout=cfg.get("dropout", 0.1),
    )


# ── QuartzNet ─────────────────────────────────────────────────────

class _QuartzNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dropout):
        super().__init__()
        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size, padding="same", groups=in_ch)
        self.pw = nn.Conv1d(in_ch, out_ch, 1)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.res = nn.Sequential(nn.Conv1d(in_ch, out_ch, 1), nn.BatchNorm1d(out_ch)) if in_ch != out_ch else None

    def forward(self, x):
        r = x
        x = self.drop(self.act(self.bn(self.pw(self.dw(x))) + (self.res(r) if self.res else r)))
        return x


class _QuartzNet(nn.Module):
    def __init__(self, input_dim, qconfig, emb_dim, dropout):
        super().__init__()
        layers = []
        in_ch = input_dim
        for ch, ks, reps in qconfig:
            for _ in range(reps):
                layers.append(_QuartzNetBlock(in_ch, ch, ks, dropout))
                in_ch = ch
        self.blocks = nn.Sequential(*layers)
        self.fc = nn.Linear(in_ch, emb_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.blocks(x)
        return self.fc(x.mean(dim=2))


def build_quartznet(n_features: int, cfg: dict) -> nn.Module:
    return _QuartzNet(
        input_dim=96,
        qconfig=cfg.get("quartznet_config", [(64, 11, 1), (64, 13, 1), (64, 15, 1)]),
        emb_dim=cfg.get("embedding_dim", 1),
        dropout=cfg.get("dropout", 0.1),
    )


# ── Conformer ─────────────────────────────────────────────────────

class _ConvModule(nn.Module):
    def __init__(self, d, ks=31):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.conv1 = nn.Conv1d(d, d * 2, 1)
        self.glu = nn.GLU(dim=1)
        self.dw = nn.Conv1d(d, d, ks, groups=d, padding="same")
        self.bn = nn.BatchNorm1d(d)
        self.swish = _Swish()
        self.conv2 = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.norm(x).permute(0, 2, 1)
        x = self.drop(self.conv2(self.swish(self.bn(self.dw(self.glu(self.conv1(x)))))))
        return x.permute(0, 2, 1)


class _FFModule(nn.Module):
    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d * 4)
        self.swish = _Swish()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d * 4, d)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop2(self.fc2(self.drop1(self.swish(self.fc1(self.norm(x))))))


class _ConformerBlock(nn.Module):
    def __init__(self, d, n_head, dropout):
        super().__init__()
        self.ff1 = _FFModule(d, dropout)
        self.attn = nn.MultiheadAttention(d, n_head, dropout=dropout, batch_first=True)
        self.conv = _ConvModule(d)
        self.ff2 = _FFModule(d, dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        x = x + 0.5 * self.ff1(x)
        a, _ = self.attn(x, x, x)
        x = x + a
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.norm(x)


class _Conformer(nn.Module):
    def __init__(self, input_dim, d_model, n_head, n_layers, emb_dim, dropout):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[_ConformerBlock(d_model, n_head, dropout) for _ in range(n_layers)])
        self.out = nn.Linear(d_model, emb_dim)

    def forward(self, x):
        x = self.drop(self.proj(x))
        x = self.blocks(x)
        return self.out(x.mean(dim=1))


def build_conformer(n_features: int, cfg: dict) -> nn.Module:
    return _Conformer(
        input_dim=96,
        d_model=cfg.get("d_model", 64),
        n_head=cfg.get("n_head", 4),
        n_layers=cfg.get("n_layers", 2),
        emb_dim=cfg.get("embedding_dim", 1),
        dropout=cfg.get("dropout", 0.1),
    )


# ── E-Branchformer ────────────────────────────────────────────────

class _Merger(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gate = nn.Linear(d, d)

    def forward(self, attn, conv):
        g = torch.sigmoid(self.gate(conv))
        return attn * g + conv * (1 - g)


class _EBranchformerBlock(nn.Module):
    def __init__(self, d, n_head, dropout):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_head, dropout=dropout, batch_first=True)
        self.conv = _ConvModule(d)
        self.merger = _Merger(d)
        self.norm = nn.LayerNorm(d)
        self.ffn = _FFModule(d, dropout)

    def forward(self, x):
        an = self.attn_norm(x)
        a, _ = self.attn(an, an, an)
        c = self.conv(x)
        x = self.norm(x + self.merger(a, c))
        return x + self.ffn(x)


class _EBranchformer(nn.Module):
    def __init__(self, input_dim, d_model, n_head, n_layers, emb_dim, dropout):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[_EBranchformerBlock(d_model, n_head, dropout) for _ in range(n_layers)])
        self.out = nn.Linear(d_model, emb_dim)

    def forward(self, x):
        x = self.drop(self.proj(x))
        x = self.blocks(x)
        return self.out(x.mean(dim=1))


def build_e_branchformer(n_features: int, cfg: dict) -> nn.Module:
    return _EBranchformer(
        input_dim=96,
        d_model=cfg.get("d_model", 64),
        n_head=cfg.get("n_head", 4),
        n_layers=cfg.get("n_layers", 2),
        emb_dim=cfg.get("embedding_dim", 1),
        dropout=cfg.get("dropout", 0.1),
    )


# ── CRNN ──────────────────────────────────────────────────────────

class _CRNN(nn.Module):
    def __init__(self, input_shape, rnn_type, rnn_hidden, n_rnn, cnn_channels, emb_dim, dropout, activation):
        super().__init__()
        layers = []
        in_ch = 1
        for out_ch in cnn_channels:
            layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), activation, nn.MaxPool2d(2, 2)]
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *input_shape)
            co = self.cnn(dummy)
            rnn_in = co.shape[1] * co.shape[2]
        RNN = nn.GRU if rnn_type.lower() == "gru" else nn.LSTM
        self.rnn = RNN(rnn_in, rnn_hidden, n_rnn, batch_first=True, bidirectional=True,
                       dropout=dropout if n_rnn > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(rnn_hidden * 2, emb_dim)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        c = self.cnn(x)
        B, C, H, W = c.shape
        r = c.view(B, C * H, W).permute(0, 2, 1)
        out, _ = self.rnn(r)
        return self.fc(self.dropout(out[:, -1, :]))


def build_crnn(n_features: int, cfg: dict) -> nn.Module:
    return _CRNN(
        input_shape=(n_features, 96),
        rnn_type=cfg.get("rnn_type", "lstm"),
        rnn_hidden=cfg.get("rnn_hidden_size", 64),
        n_rnn=cfg.get("n_rnn_layers", 2),
        cnn_channels=cfg.get("cnn_channels", [16, 32]),
        emb_dim=cfg.get("embedding_dim", 1),
        dropout=cfg.get("dropout", 0.1),
        activation=_get_activation(cfg.get("activation", "relu")),
    )


# ── Registry ──────────────────────────────────────────────────────

ARCHITECTURES: dict[str, callable] = {
    "dnn": build_dnn,
    "cnn": build_cnn,
    "lstm": build_lstm,
    "gru": build_gru,
    "rnn": build_rnn,
    "transformer": build_transformer,
    "tcn": build_tcn,
    "quartznet": build_quartznet,
    "conformer": build_conformer,
    "e_branchformer": build_e_branchformer,
    "crnn": build_crnn,
}
