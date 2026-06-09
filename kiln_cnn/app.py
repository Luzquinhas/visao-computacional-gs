"""Demo Streamlit: classifica um tile novo ao vivo + Grad-CAM.

Uso:
    streamlit run kiln_cnn/app.py -- --weights weights/deep_type.pt

Faça upload de um tile (idealmente 128x128 RGB). O app mostra a classe predita,
as probabilidades e o mapa Grad-CAM indicando onde a rede olhou.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import streamlit as st
import torch
from PIL import Image

from kiln_cnn.gradcam import GradCAM, overlay_heatmap
from kiln_cnn.models import build_model


def parse_args() -> argparse.Namespace:
    # argumentos após o "--" do streamlit
    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="weights/deep_type.pt")
    return p.parse_args(argv)


@st.cache_resource
def load_model(weights_path: str):
    ckpt = torch.load(weights_path, map_location="cpu")
    model = build_model(ckpt["model"], len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["classes"]


def to_tensor(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB").resize((128, 128))
    arr = np.asarray(img).astype("float32") / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def main() -> None:
    args = parse_args()
    st.set_page_config(page_title="Detector de Fornos de Tijolo", layout="centered")
    st.title("🧱 Detector de Fornos de Tijolo (imagens de satélite)")
    st.caption("CNN treinada do zero — classificação de tiles + Grad-CAM")

    try:
        model, classes = load_model(args.weights)
    except FileNotFoundError:
        st.error(f"Pesos não encontrados: {args.weights}. Treine antes com kiln_cnn.train.")
        return

    file = st.file_uploader("Envie um tile de satélite", type=["png", "jpg", "jpeg"])
    if file is None:
        return

    image = Image.open(file)
    tensor = to_tensor(image)

    with torch.no_grad():
        probs = torch.softmax(model(tensor.unsqueeze(0)), dim=1)[0]
    pred = int(probs.argmax())

    cam, _ = GradCAM(model)(tensor, target_class=pred)
    overlay = overlay_heatmap(tensor, cam)

    col1, col2 = st.columns(2)
    col1.image(image, caption="Entrada", use_container_width=True)
    col2.image(overlay, caption="Grad-CAM (onde a rede olhou)", use_container_width=True)

    st.subheader(f"Predição: **{classes[pred]}**  ({probs[pred] * 100:.1f}%)")
    st.bar_chart({classes[i]: float(probs[i]) for i in range(len(classes))})


if __name__ == "__main__":
    main()
