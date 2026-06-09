"""Dataset de classificação de tiles de fornos de tijolo (SentinelKilnDB).

Segue as convenções do TorchGeo (``NonGeoDataset``): cada item é um *sample*
``dict`` com tensores, nunca uma tupla. As chaves usadas aqui são:

    {"image": Tensor[C, H, W], "label": Tensor escalar}

Os dados originais vêm com rótulos de *detecção* (YOLO / DOTA). Aqui convertemos
o problema para **classificação de tile**, exatamente como sugerido no ``idea.md``:
um tile recebe um único rótulo. Tiles sem arquivo de rótulo são "sem forno".
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Callable

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

# Mapeamento dos ids do formato YOLO (coluna 0 do .txt) para nome do forno.
# Descoberto a partir dos rótulos DOTA: 0=CFCBK (circular), 1=FCBK, 2=Zigzag.
YOLO_ID_TO_NAME = {0: "CFCBK", 1: "FCBK", 2: "Zigzag"}

# Esquema 4 classes (padrão): primeiro "sem forno", depois os três tipos.
CLASSES_TYPE = ["none", "CFCBK", "FCBK", "Zigzag"]
# Esquema binário: apenas "tem forno" / "não tem forno".
CLASSES_BINARY = ["none", "kiln"]

Sample = dict[str, Tensor]


class KilnTiles(Dataset[Sample]):
    """Tiles 128x128 RGB rotulados por tipo de forno (ou tem/não tem forno).

    Args:
        root: pasta que contém os splits ``train/`` ``val/`` ``test/``.
        split: ``"train"``, ``"val"`` ou ``"test"``.
        task: ``"type"`` (4 classes) ou ``"binary"`` (2 classes).
        transforms: função opcional aplicada ao *sample* dict.
        fraction: usa apenas uma fração do split (útil para testes rápidos).
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        task: str = "type",
        transforms: Callable[[Sample], Sample] | None = None,
        fraction: float = 1.0,
    ) -> None:
        assert split in {"train", "val", "test"}
        assert task in {"type", "binary"}
        self.root = root
        self.split = split
        self.task = task
        self.transforms = transforms
        self.classes = CLASSES_TYPE if task == "type" else CLASSES_BINARY

        self.image_dir = os.path.join(root, split, "images")
        self.label_dir = os.path.join(root, split, "yolo_aa_labels")
        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Pasta de imagens não encontrada: {self.image_dir}")

        # Constrói o índice (caminho, rótulo) uma única vez.
        self.samples: list[tuple[str, int]] = []
        names = sorted(os.listdir(self.image_dir))
        if fraction < 1.0:
            names = names[: int(len(names) * fraction)]
        for name in names:
            if not name.lower().endswith(".png"):
                continue
            label = self._label_for(name)
            self.samples.append((os.path.join(self.image_dir, name), label))

    def _label_for(self, image_name: str) -> int:
        """Lê o .txt de detecção e devolve o id da classe do tile."""
        txt = os.path.join(self.label_dir, os.path.splitext(image_name)[0] + ".txt")
        if not os.path.exists(txt) or os.path.getsize(txt) == 0:
            return 0  # "none" em ambos os esquemas

        if self.task == "binary":
            return 1  # qualquer rótulo presente => "kiln"

        # task == "type": escolhe o tipo de forno mais frequente no tile.
        ids = []
        with open(txt) as f:
            for line in f:
                parts = line.split()
                if parts:
                    ids.append(int(float(parts[0])))
        if not ids:
            return 0
        dominant = Counter(ids).most_common(1)[0][0]
        name = YOLO_ID_TO_NAME[dominant]
        return self.classes.index(name)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        path, label = self.samples[index]
        with Image.open(path) as img:
            img = img.convert("RGB")
            # [H, W, C] uint8 -> [C, H, W] float em [0, 1]
            tensor = torch.from_numpy(_to_array(img)).permute(2, 0, 1).float() / 255.0

        sample: Sample = {"image": tensor, "label": torch.tensor(label, dtype=torch.long)}
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def class_counts(self) -> list[int]:
        """Quantidade de tiles por classe (para inspecionar desbalanceamento)."""
        counts = [0] * len(self.classes)
        for _, label in self.samples:
            counts[label] += 1
        return counts


def _to_array(img: Image.Image):
    import numpy as np

    return np.array(img)  # cópia gravável (evita warning do torch.from_numpy)
