"""Avaliação no split de teste: accuracy, matriz de confusão e exemplos.

Exemplo:
    python -m kiln_cnn.evaluate --weights weights/deep_type.pt

Gera em ``outputs/``:
  - matriz de confusão (png)
  - relatório de precisão/recall por classe (texto no console)
  - grade com exemplos de acertos e erros (png)
"""

from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

from .dataset import KilnTiles
from .models import build_model


@torch.no_grad()
def collect_predictions(model, loader, device):
    y_true, y_pred = [], []
    model.eval()
    for batch in loader:
        logits = model(batch["image"].to(device))
        y_pred.extend(logits.argmax(1).cpu().tolist())
        y_true.extend(batch["label"].tolist())
    return y_true, y_pred


def main() -> None:
    p = argparse.ArgumentParser(description="Avalia um modelo treinado no teste.")
    p.add_argument("--root", default=".")
    p.add_argument("--weights", required=True, help="checkpoint .pt salvo pelo train.py")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--fraction", type=float, default=1.0)
    p.add_argument("--out", default="outputs")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    ckpt = torch.load(args.weights, map_location=device)
    classes = ckpt["classes"]
    model = build_model(ckpt["model"], len(classes)).to(device)
    model.load_state_dict(ckpt["state_dict"])

    test_ds = KilnTiles(args.root, "test", ckpt["task"], fraction=args.fraction)
    loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=args.workers)

    y_true, y_pred = collect_predictions(model, loader, device)

    cm = _confusion_matrix(y_true, y_pred, len(classes))
    acc = cm.trace() / cm.sum() if cm.sum() else 0.0
    print(f"\nAccuracy no teste: {acc:.4f} (meta do edital: 0.88)\n")
    _print_report(cm, classes)

    _plot_confusion(cm, classes, os.path.join(args.out, f"confusion_{ckpt['model']}_{ckpt['task']}.png"))
    _plot_examples(test_ds, y_true, y_pred, classes,
                   os.path.join(args.out, f"examples_{ckpt['model']}_{ckpt['task']}.png"))


def _confusion_matrix(y_true, y_pred, n: int):
    import numpy as np

    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def _print_report(cm, classes) -> None:
    """Precision/recall/F1 por classe a partir da matriz de confusão."""
    print(f"{'classe':>10} {'prec':>8} {'recall':>8} {'f1':>8} {'suporte':>8}")
    for i, name in enumerate(classes):
        tp = cm[i, i]
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        rec = tp / cm[i, :].sum() if cm[i, :].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        print(f"{name:>10} {prec:8.4f} {rec:8.4f} {f1:8.4f} {cm[i, :].sum():8d}")
    print()


def _plot_confusion(cm, classes, path: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("Predito"); ax.set_ylabel("Real"); ax.set_title("Matriz de confusão")
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, int(cm[i, j]), ha="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    print(f"Matriz de confusão salva em: {path}")


def _plot_examples(dataset, y_true, y_pred, classes, path: str, n: int = 8) -> None:
    import matplotlib.pyplot as plt

    hits = [i for i in range(len(y_true)) if y_true[i] == y_pred[i]]
    errs = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]
    picks = hits[: n // 2] + errs[: n // 2]
    if not picks:
        return
    cols = len(picks)
    fig, axes = plt.subplots(1, cols, figsize=(2.2 * cols, 2.8))
    if cols == 1:
        axes = [axes]
    for ax, idx in zip(axes, picks):
        img = dataset[idx]["image"].permute(1, 2, 0).numpy()
        ax.imshow(img); ax.axis("off")
        ok = y_true[idx] == y_pred[idx]
        ax.set_title(f"R:{classes[y_true[idx]]}\nP:{classes[y_pred[idx]]}",
                     color="green" if ok else "red", fontsize=8)
    fig.suptitle("Exemplos — acertos (verde) e erros (vermelho)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    print(f"Exemplos salvos em: {path}")


if __name__ == "__main__":
    main()
