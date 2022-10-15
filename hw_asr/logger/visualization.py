import logging
from enum import Enum

from .tensorboard import TensorboardWriter

logger = logging.getLogger(__name__)

try:
    import wandb
    from .wandb import WanDBWriter

    WANDB_AVAILABLE = True
except:
    logger.info('Wandb is not available.')


class VisualizerBackendType(str, Enum):
    tensorboard = "tensorboard"
    wandb = "wandb"


def get_visualizer(config, logger, backend: VisualizerBackendType):
    if backend == VisualizerBackendType.tensorboard:
        return TensorboardWriter(config.log_dir, logger, True)

    assert WANDB_AVAILABLE, 'Wandb is not available!'

    if backend == VisualizerBackendType.wandb:
        return WanDBWriter(config, logger)

    return None
