import torch
from torch import nn
from typing import Union, Any
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence
)

from hw_asr.base import BaseModel


class DeepSpeech(BaseModel):
    """
    Relevant papers
        https://arxiv.org/pdf/1512.02595.pdf
        http://proceedings.mlr.press/v48/amodei16.pdf
        http://ceur-ws.org/Vol-2267/470-474-paper-90.pdf
        https://arxiv.org/pdf/1904.08779.pdf
    Nvidia Docs
        https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition/deepspeech2.html
    Libri Speech
        https://www.openslr.org/12
    """
    def __init__(
        self,
        n_feats: int,
        n_class: int,
        bidirectional: bool = True,
        num_layers: int = 5,
        hidden_size: int = 400,
    ):
        super(DeepSpeech, self).__init__(n_feats, n_class)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20),
        )

        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        input_size = self.transform_input_lengths(n_feats, 0) * 32
        self.gru = nn.Sequential(
            nn.GRU(
                input_size=input_size,
                num_layers=num_layers,
                hidden_size=hidden_size,
                bidirectional=self.bidirectional,
            )
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Hardtanh(0, 20),
            nn.Linear(hidden_size, n_class),
        )

    def forward(self, spectrogram, **batch) -> Union[torch.Tensor, dict]:
        x = spectrogram.unsqueeze(1)  # (N, C_in, N_ff, N_t)
        x = self.conv(x)  # (N, C_out, N_ff, N_t)

        x = x.view(x.size(0), x.size(1) * x.size(2), x.size(3))  # (N, C_out * N_ff, N_t)
        x = x.permute(2, 0, 1)  # (N_t, N, N_ff)

        lengths = self.transform_input_lengths(batch["spectrogram_length"])
        x = pack_padded_sequence(x, lengths, enforce_sorted=False)
        x, _ = self.gru(x)  # (N_t, N, D * N_hidden)
        x, _ = pad_packed_sequence(x)  # (N_t, N, D * N_hidden)

        if self.bidirectional:
            x = x[:, :, :self.hidden_size] + x[:, :, self.hidden_size:]  # (N_t, N, N_hidden)

        x = self.fc(x)  # (N_t, N, N_class)
        x = x.transpose(0, 1)  # (N, N_t, N_class)
        return x

    def transform_input_lengths(self, input_length: Any, dim: int = 1):
        """Calculate size after convolution layers via selected dimension."""
        for i, module in enumerate(self.conv):
            if i % 3 == 0:
                input_length = torch.div(
                    (input_length + 2 * module.padding[dim] - module.dilation[dim] * (module.kernel_size[dim] - 1) - 1),
                    module.stride[dim],
                    rounding_mode="trunc",
                ) + 1
        return input_length
