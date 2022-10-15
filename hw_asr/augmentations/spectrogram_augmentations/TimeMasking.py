from torch import Tensor
import torchaudio.transforms

from hw_asr.augmentations.base import AugmentationBase


class TimeMasking(AugmentationBase):
    def __init__(self, time_mask_param, *args, **kwargs):
        super().__init__()
        self._aug = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)

    def __call__(self, data: Tensor):
        return self._aug(data)
