"""Data augmentation para tiles de satélite (combate ao overfitting).

Tiles vistos de cima (nadir) são invariantes a espelhamentos e a rotações de
90° (grupo de simetria D4): girar/espelhar um forno continua sendo o mesmo
forno. Isso multiplica o dataset efetivo sem custo de coleta.

A transform opera no formato *sample dict* que o ``KilnTiles`` já espera
(``{"image": Tensor[C, H, W], "label": Tensor}``) e deve ser aplicada
**apenas no split de treino** — val/test ficam sem augmentation.
"""

from __future__ import annotations

import torch
from torchvision.transforms import functional as TF

from .dataset import Sample


class TileAugment:
    """Augmentation para tiles nadir: flips H/V + rotações de 90° + jitter leve.

    Args:
        train: se ``False``, a transform é um *no-op* (passa a imagem adiante
            inalterada), útil para manter a mesma assinatura em val/test.
    """

    def __init__(self, train: bool = True) -> None:
        self.train = train

    def __call__(self, sample: Sample) -> Sample:
        if not self.train:
            return sample

        img = sample["image"]  # [C, H, W] float em [0, 1]
        if torch.rand(1).item() < 0.5:
            img = TF.hflip(img)
        if torch.rand(1).item() < 0.5:
            img = TF.vflip(img)
        k = int(torch.randint(0, 4, (1,)).item())  # 0 / 90 / 180 / 270 graus
        if k:
            img = torch.rot90(img, k, dims=(1, 2))
        # jitter fotométrico leve (variação de iluminação/atmosfera)
        img = TF.adjust_brightness(img, 1.0 + (torch.rand(1).item() - 0.5) * 0.3)
        img = TF.adjust_contrast(img, 1.0 + (torch.rand(1).item() - 0.5) * 0.3)

        sample["image"] = img.clamp(0.0, 1.0).contiguous()
        return sample
