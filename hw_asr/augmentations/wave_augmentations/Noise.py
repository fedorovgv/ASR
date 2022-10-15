import torch

from torch_audiomentations import AddColoredNoise


class Noise(torch.nn.Module):
    """Fast version from torch_audiomentations."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = AddColoredNoise(*args, **kwargs)

    def forward(self, audio):
        aug_audio = self._aug(audio.unsqueeze(1)).squeeze(1)
        assert aug_audio.size() == audio.size()
        return aug_audio
