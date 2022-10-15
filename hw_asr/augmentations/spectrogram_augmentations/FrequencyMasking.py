from torch import Tensor
import torchaudio.transforms

from hw_asr.augmentations.base import AugmentationBase


class FrequencyMasking(AugmentationBase):
    def __init__(self, freq_mask_param, *args, **kwargs):
        super().__init__()
        self._aug = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)

    def __call__(self, data: Tensor):
        return self._aug(data)
