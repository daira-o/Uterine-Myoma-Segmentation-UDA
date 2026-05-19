import os
import argparse
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser(description="Visualizador rápido de cortes procesados (NPY)")
    parser.add_argument("--path", type=str, default="data_ready_RM/train", 
                        help="Ruta al split que se desea visualizar (ej: data_ready_RM/train)")
    parser.add_argument("--num", type=int, default=3, 
                        help="Número de muestras aleatorias a visualizar")
    return parser.parse_args()

def visualize_samples(split_path: str, num_samples: int):
    split_dir = Path(split_path)
    img_dir = split_dir / "images"
    mask_dir = split_dir / "masks"

    if not img_dir.exists() or not mask_dir.exists():
        print(f"[ERROR] No se encontraron las carpetas 'images' o 'masks' en: {split_dir.resolve()}")
        return

    # Listar y emparejar archivos por ID
    img_files = sorted(list(img_dir.glob("*.npy")))
    if not img_files:
        print(f"[ERROR] No hay archivos .npy en {img_dir}")
        return

    # Selección aleatoria de muestras
    num_samples = min(num_samples, len(img_files))
    selected_imgs = random.sample(img_files, num_samples)

    print(f"Mostrando {num_samples} muestras de: {split_dir.name.upper()}")

    for img_path in selected_imgs:
        # El ID es idéntico para imagen y máscara
        file_name = img_path.name
        mask_path = mask_dir / file_name

        if not mask_path.exists():
            print(f"[WARN] Falta la máscara para la imagen: {file_name}. Saltando...")
            continue

        # Cargar datos npy
        img = np.load(img_path)
        mask = np.load(mask_path)

        # Configuración del plot (Matplotlib)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Muestra: {img_path.stem} | Dim: {img.shape}", fontsize=12, fontweight='bold')

        # 1. Imagen MRI Original (Procesada)
        axes[0].imshow(img, cmap="gray")
        axes[0].set_title("MRI (T2 Sagital)")
        axes[0].axis("off")

        # 2. Máscara de Segmentación Ground Truth
        axes[1].imshow(mask, cmap="inferno")
        axes[1].set_title(f"Máscara (Área: {int(np.sum(mask))} px)")
        axes[1].axis("off")

        # 3. Superposición (Overlay)
        axes[2].imshow(img, cmap="gray")
        # Enmascarar ceros para que el fondo del overlay sea transparente
        masked_overlay = np.ma.masked_where(mask == 0, mask)
        axes[2].imshow(masked_overlay, cmap="autumn", alpha=0.5)
        axes[2].set_title("Superposición (Overlay)")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    args = get_args()
    visualize_samples(args.path, args.num)