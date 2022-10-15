import logging
import torch
from typing import List, Dict
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence


logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]) -> Dict:
    """
    Collate and pad fields in dataset items.
    Dataset_items object keys:
        "audio": torch.Tensor
        "spectrogram": torch.Tensor - N * N_freq_feat * N_time
        "duration": float,
        "text": str,
        "text_encoded": torch.Tensor, -
        "audio_path": str,
        ...
    Returns dict of collated samples and additional info.
    """
    fields = defaultdict(list)
    for item in dataset_items:
        for k, v in item.items():
            fields[k].append(v)

    batch = {k: fields[k] for k in fields if type(fields[k]) != torch.Tensor or k == "audio"}

    batch["spectrogram"] = [item.squeeze(0).transpose(1, 0) for item in fields["spectrogram"]]  # N_t * N_ff
    batch["spectrogram_length"] = torch.tensor([item.size(0) for item in batch["spectrogram"]])  # get N_t's
    batch["spectrogram"] = pad_sequence(batch["spectrogram"], batch_first=True)  # N * max(N_t) * N_ff
    batch["spectrogram"] = batch["spectrogram"].transpose(2, 1)  # N * N_ff * N_t

    batch["text_encoded"] = [torch.transpose(item, 1, 0) for item in fields["text_encoded"]]
    batch["text_encoded_length"] = torch.tensor([item.size(0) for item in batch["text_encoded"]])
    batch["text_encoded"] = pad_sequence(batch["text_encoded"], batch_first=True).squeeze(-1)

    return batch
