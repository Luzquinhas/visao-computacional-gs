"""Treino de uma das CNNs do zero, com métricas por época.

Exemplos:
    # treino rápido com 10% dos dados para validar o pipeline
    python -m kiln_cnn.train --model simple --epochs 3 --fraction 0.1

    # treino completo da CNN profunda
    python -m kiln_cnn.train --model deep --epochs 20 --batch-size 128

Salva: pesos do melhor modelo (val accuracy) em ``weights/`` e a curva de
loss/accuracy por época em ``outputs/``.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import KilnTiles
from .models import build_model
from .transforms import TileAugment


def make_balanced_sampler(dataset: KilnTiles) -> WeightedRandomSampler:
    """Amostragem balanceada: compensa o desbalanceamento entre classes.

    Importante aqui porque CFCBK (circular) é raríssimo (~3% dos positivos).
    """
    counts = dataset.class_counts()
    per_class_w = [0.0 if c == 0 else 1.0 / c for c in counts]
    weights = [per_class_w[label] for _, label in dataset.samples]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def run_epoch(model, loader, criterion, device, optimizer=None, desc="") -> tuple[float, float]:
    """Roda uma época. Se ``optimizer`` for None, é avaliação (sem grad)."""
    from tqdm import tqdm

    train = optimizer is not None
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    bar = tqdm(loader, desc=desc, leave=False)
    with torch.set_grad_enabled(train):  # context: restaura o estado do autograd ao sair
        for batch in bar:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)
            bar.set_postfix(loss=f"{total_loss / total:.4f}", acc=f"{correct / total:.4f}")
    return total_loss / total, correct / total


def main() -> None:
    p = argparse.ArgumentParser(description="Treina uma CNN de classificação de fornos.")
    p.add_argument("--root", default=".", help="pasta com train/ val/ test/")
    p.add_argument("--model", choices=["simple", "deep"], default="simple")
    p.add_argument("--task", choices=["type", "binary"], default="type")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4, help="regularização L2 (AdamW)")
    p.add_argument("--patience", type=int, default=5, help="épocas sem melhora da val_loss antes do early stopping")
    p.add_argument("--fraction", type=float, default=1.0, help="fração dos dados a usar")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-balance", action="store_true", help="desliga amostragem balanceada")
    p.add_argument("--out", default="outputs")
    p.add_argument("--weights", default="weights")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.weights, exist_ok=True)

    # Augmentation apenas no treino (val/test sem augmentation).
    train_ds = KilnTiles(args.root, "train", args.task, fraction=args.fraction,
                         transforms=TileAugment(train=True))
    val_ds = KilnTiles(args.root, "val", args.task, fraction=args.fraction)
    print(f"Classes: {train_ds.classes}")
    print(f"Treino: {len(train_ds)} tiles | distribuição: {train_ds.class_counts()}")
    print(f"Val:    {len(val_ds)} tiles  | distribuição: {val_ds.class_counts()}")

    sampler = None if args.no_balance else make_balanced_sampler(train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        shuffle=sampler is None, num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True
    )

    model = build_model(args.model, len(train_ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    # AdamW: regularização L2 (weight decay) desacoplada — combate overfitting.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Reduz o LR quando a val_loss estaciona.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2)

    history: dict[str, list[float]] = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0
    best_val_loss, bad_epochs = float("inf"), 0
    ckpt = os.path.join(args.weights, f"{args.model}_{args.task}.pt")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer,
                                    desc=f"Época {epoch:02d} treino")
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device,
                                    desc=f"Época {epoch:02d} val")
        scheduler.step(va_loss)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        print(
            f"Época {epoch:02d}/{args.epochs} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f}"
        )
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({"model": args.model, "task": args.task,
                        "classes": train_ds.classes, "state_dict": model.state_dict()}, ckpt)
        # Early stopping pela val_loss (sinal de overfitting).
        if va_loss < best_val_loss - 1e-4:
            best_val_loss, bad_epochs = va_loss, 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping na época {epoch} "
                      f"(val_loss não melhora há {args.patience} épocas)")
                break

    print(f"\nMelhor val accuracy: {best_acc:.4f} (meta do edital: 0.88)")
    print(f"Pesos salvos em: {ckpt}")

    with open(os.path.join(args.out, f"history_{args.model}_{args.task}.json"), "w") as f:
        json.dump(history, f, indent=2)
    _plot_curves(history, os.path.join(args.out, f"curves_{args.model}_{args.task}.png"))


def _plot_curves(history: dict[str, list[float]], path: str) -> None:
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, history["train_loss"], marker="o", label="treino")
    ax1.plot(epochs, history["val_loss"], marker="o", label="val")
    ax1.set_title("Loss por época"); ax1.set_xlabel("época"); ax1.legend()
    ax2.plot(epochs, history["train_acc"], marker="o", label="treino")
    ax2.plot(epochs, history["val_acc"], marker="o", label="val")
    ax2.axhline(0.88, color="gray", ls="--", label="meta 88%")
    ax2.set_title("Accuracy por época"); ax2.set_xlabel("época"); ax2.legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    print(f"Curvas salvas em: {path}")


if __name__ == "__main__":
    main()
