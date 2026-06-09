# kiln_cnn — Detecção de Fornos de Tijolo em Imagens de Satélite

Classificação de *tiles* de satélite (SentinelKilnDB) para detectar fornos de
tijolo / minas ilegais, usando **duas CNNs construídas do zero** (sem modelos
pré-treinados, conforme o edital da disciplina de ACV).

A abordagem segue o `idea.md`: em vez de detecção de objetos (YOLO/U-Net), o
problema é tratado como **classificação de tile** — muito mais viável de treinar
do zero. O pacote é inspirado nas convenções do [TorchGeo](../torchgeo) (camadas
*dataset → modelo → treino*, *sample* como `dict` `{"image", "label"}`).

**Integrantes:** 
- Lucas Rodrigues da Silva | RM: 98344
- Juan Pinheiro de França  | RM: 552202 
- Kaiky Alvaro de Miranda  | RM: 98118

## Estrutura

```
kiln_cnn/
  dataset.py     # KilnTiles: lê os tiles + rótulos YOLO -> rótulo de classe
  models.py      # SimpleCNN (CNN 1) e DeeperCNN (CNN 2) — do zero
  train.py       # treino com métricas por época + curvas
  evaluate.py    # accuracy, matriz de confusão, relatório, exemplos
  gradcam.py     # Grad-CAM (onde a rede olhou)
  app.py         # demo Streamlit ao vivo
```

## Dados

Os splits `train/`, `val/`, `test/` já estão na raiz do repositório, cada um com
`images/` (tiles 128×128 RGB `.png`) e `yolo_aa_labels/` (rótulos de detecção).
A conversão para classificação é automática:

| classe | origem |
|--------|--------|
| `none`   | tile **sem** arquivo de rótulo |
| `CFCBK`  | id YOLO `0` (forno circular) |
| `FCBK`   | id YOLO `1` |
| `Zigzag` | id YOLO `2` |

Tiles com vários fornos recebem o **tipo mais frequente**. Há também o modo
binário (`--task binary`): apenas `none` / `kiln`.

>  **Desbalanceamento:** `CFCBK` é raríssimo (~3% dos positivos). O treino usa
> amostragem balanceada por padrão (`WeightedRandomSampler`); ainda assim, vale
> justificar no relatório eventuais quedas de recall nessa classe.

## Como usar

```bash
pip install -r kiln_cnn/requirements.txt

# 1) Treino rápido (sanidade do pipeline, 10% dos dados)
python -m kiln_cnn.train --model simple --epochs 3 --fraction 0.1

# 2) CNN 1 (baseline) — treino completo
python -m kiln_cnn.train --model simple --epochs 20 --batch-size 128

# 3) CNN 2 (profunda, BatchNorm + Dropout) — para comparação
python -m kiln_cnn.train --model deep --epochs 20 --batch-size 128

# 4) Avaliação no teste (matriz de confusão + relatório + exemplos)
python -m kiln_cnn.evaluate --weights weights/deep_type.pt

# 5) Demo ao vivo com Grad-CAM
streamlit run kiln_cnn/app.py -- --weights weights/deep_type.pt
```

Saídas: pesos do melhor modelo em `weights/`; curvas de loss/accuracy, matriz de
confusão e grade de exemplos em `outputs/`.

## Mapa para o `idea.md`

| Passo do edital | Onde está |
|-----------------|-----------|
| 2. Dataset balanceado, splits treino/val/teste | `dataset.py` + `make_balanced_sampler` |
| 3. CNN 1 (simples) | `models.SimpleCNN` |
| 4. CNN 2 (profunda, dropout/batchnorm) | `models.DeeperCNN` |
| 5. Métricas (accuracy, loss/época, matriz de confusão, exemplos) | `train.py` + `evaluate.py` |
| 6. Meta de 88% | impressa e marcada nas curvas |
| 7. Demo + Grad-CAM | `app.py` + `gradcam.py` |
| 8. Entrega (GitHub, pesos, requirements, README) | este pacote |
