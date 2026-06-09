"""Grad-CAM: mostra onde a rede "olhou" para tomar a decisão.

Implementação mínima (sem dependências externas além de torch/numpy), usando
hooks na última camada convolucional. Usado tanto no notebook quanto no app.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .models import last_conv_layer


class GradCAM:
    """Calcula o mapa de ativação Grad-CAM para uma imagem e classe-alvo."""

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None) -> None:
        self.model = model.eval()
        self.layer = target_layer or last_conv_layer(model)
        self.activations: Tensor | None = None
        self.gradients: Tensor | None = None
        self.layer.register_forward_hook(self._save_activation)
        self.layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inp, out: Tensor) -> None:
        self.activations = out.detach()

    def _save_gradient(self, _module, _grad_in, grad_out) -> None:
        self.gradients = grad_out[0].detach()

    def __call__(self, image: Tensor, target_class: int | None = None) -> tuple[np.ndarray, int]:
        """Args:
            image: tensor [C, H, W] ou [1, C, H, W] em [0, 1].
        Returns:
            (heatmap HxW em [0,1], classe usada como alvo)
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        # Garante que a imagem esteja no mesmo device do modelo (ex.: GPU no Colab).
        image = image.to(next(self.model.parameters()).device)
        with torch.enable_grad():  # robusto mesmo se chamado dentro de no_grad()
            logits = self.model(image)
            if target_class is None:
                target_class = int(logits.argmax(1).item())
            self.model.zero_grad()
            logits[0, target_class].backward()

        # peso de cada canal = média espacial do gradiente
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, h, w]
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam, target_class


def overlay_heatmap(image: Tensor, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Sobrepõe o heatmap (vermelho) sobre a imagem RGB. Retorna HxWx3 em [0,1]."""
    img = image.detach().cpu()
    if img.dim() == 4:
        img = img[0]
    rgb = img.permute(1, 2, 0).numpy()
    heat = np.stack([cam, np.zeros_like(cam), 1 - cam], axis=-1)  # vermelho->azul
    return np.clip((1 - alpha) * rgb + alpha * heat, 0, 1)
