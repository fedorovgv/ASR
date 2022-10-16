#  from Nemo library "NeMo offline ASR" notebook example

import logging
import os, shutil
from pathlib import Path
from speechbrain.utils.data_utils import download_file

logger = logging.getLogger(__name__)

AVAILABLE_MODELS = [
    '3-gram.arpa',
    '3-gram.pruned.1e-7.arpa',
    '3-gram.pruned.3e-7.arpa',
    '4-gram.arpa',
]


def load_model(lm_dir: Path, lm_model: str):
    assert lm_model in AVAILABLE_MODELS

    lm_gzip_path = lm_model + '.gz'
    print(f'LM MODEL: {lm_model}  == ? 3-gram.pruned.1e-7.arpa, {os.path.exists(lm_model)}')
    if os.path.exists(lm_model):
        logger.info(f'We also have language model in {lm_dir}')
    else:
        logger.info(f'Downloading language model in {lm_dir}')
        lm_url = 'http://www.openslr.org/resources/11/' + lm_gzip_path
        download_file(lm_url, lm_dir / lm_model)

    uppercase_lm_path = lm_dir / lm_model
    if not os.path.exists(uppercase_lm_path):
        with open(lm_gzip_path, 'rb', encoding='utf-8') as f_zipped:
            with open(uppercase_lm_path, 'wb', encoding='utf-8') as f_unzipped:
                shutil.copyfileobj(f_zipped, f_unzipped)
        logger.info('Unzipped model.')

    lm_path = lm_dir / ('lowercase_' + str(lm_model))
    if not os.path.exists(lm_path):
        with open(uppercase_lm_path, 'r') as f_upper:
            with open(lm_path, 'w') as f_lower:
                for line in f_upper:
                    f_lower.write(line.lower())

    logger.info(f'Model available at {lm_path}')

    return lm_path
