"""kiln_cnn — classificação de tiles de fornos de tijolo (CNNs do zero).

Pacote inspirado nas convenções do TorchGeo (dataset -> modelo -> treino,
*sample* como dict), mas sem modelos pré-treinados, conforme o edital.
"""

from .dataset import CLASSES_BINARY, CLASSES_TYPE, KilnTiles
from .models import DeeperCNN, SimpleCNN, build_model
from .transforms import TileAugment

__all__ = [
    "KilnTiles",
    "CLASSES_TYPE",
    "CLASSES_BINARY",
    "SimpleCNN",
    "DeeperCNN",
    "build_model",
    "TileAugment",
]
