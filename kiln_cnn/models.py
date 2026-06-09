"""Duas CNNs construídas do zero (sem pesos pré-treinados).

O edital exige duas arquiteturas próprias, treinadas do zero, para comparação.
- ``SimpleCNN``: rede rasa, 3 blocos conv+pool+dense. Baseline.
- ``DeeperCNN``: rede mais profunda, com BatchNorm e Dropout para regularização.

Ambas seguem a mesma assinatura para serem intercambiáveis no treino.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SimpleCNN(nn.Module):
    """CNN 1 — baseline raso (3 blocos conv), agora regularizado.

    Entrada esperada: [B, 3, 128, 128]. Após 3 max-pools: 16x16.

    Mudanças vs. versão original (combate ao overfitting — ver
    ``sugestões de melhorias.md``):
    - ``BatchNorm2d`` em cada bloco conv (estabiliza o treino do zero);
    - ``AdaptiveAvgPool2d(2)`` no lugar do ``Flatten`` direto de ``64*16*16``:
      a cabeça densa cai de ~2,1 M para ~33 k parâmetros (fim da memorização),
      mas mantém uma grade espacial 2x2 — útil para distinguir o padrão
      espalhado do ``Zigzag`` do ``FCBK`` compacto;
    - ``Dropout`` em torno de uma camada oculta de 128 (capacidade suficiente
      para aprender, sem voltar a decorar como a versão original de 2,1 M).

    Continua mais rasa e estreita que a ``DeeperCNN`` (3 blocos, máx. 64 canais),
    preservando a comparação CNN 1 × CNN 2 do edital.
    """

    def __init__(self, num_classes: int, in_channels: int = 3, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),  # 64
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),           # 32
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),           # 16
        )
        self.pool = nn.AdaptiveAvgPool2d(2)  # 64x16x16 -> 64x2x2 (retém info espacial)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64 * 2 * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.pool(self.features(x)))


class DeeperCNN(nn.Module):
    """CNN 2 — mais profunda, com BatchNorm + Dropout (para comparação).

    4 blocos conv. BatchNorm acelera/estabiliza o treino do zero e o Dropout
    reduz overfitting — exatamente o experimento pedido no passo 4 do idea.md.

    Atualização (ver ``sugestões de melhorias.md``): o pooling final usa
    ``AdaptiveAvgPool2d(2)`` em vez de ``(1)``, preservando uma grade espacial
    2x2 antes da cabeça densa. Diferente do GAP 1x1 (que colapsa toda a
    informação espacial em médias por canal), isso retém a *disposição* dos
    padrões no tile — útil para separar o padrão espacial do ``Zigzag`` do
    ``FCBK`` compacto, que é a principal confusão observada na matriz de
    confusão (FCBK <-> Zigzag).
    """

    def __init__(self, num_classes: int, in_channels: int = 3, dropout: float = 0.4) -> None:
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(in_channels, 32),  # 64
            block(32, 64),           # 32
            block(64, 128),          # 16
            block(128, 256),         # 8
        )
        self.pool = nn.AdaptiveAvgPool2d(2)  # 256x8x8 -> 256x2x2 (retém info espacial)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256 * 2 * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.pool(self.features(x)))


def build_model(name: str, num_classes: int) -> nn.Module:
    """Fábrica usada pelos scripts (``simple`` ou ``deep``)."""
    if name == "simple":
        return SimpleCNN(num_classes)
    if name == "deep":
        return DeeperCNN(num_classes)
    raise ValueError(f"Modelo desconhecido: {name!r}. Use 'simple' ou 'deep'.")


def last_conv_layer(model: nn.Module) -> nn.Conv2d:
    """Última camada convolucional — alvo do Grad-CAM."""
    conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            conv = module
    assert conv is not None
    return conv
