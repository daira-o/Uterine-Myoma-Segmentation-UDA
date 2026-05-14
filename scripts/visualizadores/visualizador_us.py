import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

def visualizador_US_triple(base_path, n_pacientes=3):
    """
    Visualizador adaptado a la nomenclatura de ultrasonido:
    IM_XXXX_..._left/right/mid.npy
    """
    path_img = os.path.join(base_path, "images_npy")
    
    # Obtenemos los archivos base del Anchor (asumiendo que terminan en _mid o similar)
    # Si tus archivos en 1_Anchor no tienen sufijo, usamos el ID IM_XXXX
    anchor_files = glob(os.path.join(path_img, "1_Anchor", "*.npy"))
    
    for i in range(min(n_pacientes, len(anchor_files))):
        anchor_p = anchor_files[i]
        filename = os.path.basename(anchor_p)
        
        # Extraer ID base (ej. IM_0001)
        base_id = "_".join(filename.split("_")[:2]) 
        
        # Buscar laterales con glob para manejar los sufijos F18, F38, etc.
        left_p = glob(os.path.join(path_img, "2_Lateral", f"{base_id}_*_left.npy"))
        right_p = glob(os.path.join(path_img, "2_Lateral", f"{base_id}_*_right.npy"))

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Dataset Ultrasonido: {base_id}", fontsize=15, fontweight='bold')

        vistas = [
            ("LEFT", left_p[0] if left_p else None),
            ("ANCHOR (Center)", anchor_p),
            ("RIGHT", right_p[0] if right_p else None)
        ]

        for idx, (label, p) in enumerate(vistas):
            if p and os.path.exists(p):
                data = np.load(p)
                # Si el US viene con canales extra, nos quedamos con el primero
                if data.ndim == 3: data = data[:,:,0]
                
                axes[idx].imshow(data, cmap='gray')
                axes[idx].set_title(label)
            else:
                axes[idx].text(0.5, 0.5, f"{label}\nNo encontrado", 
                               ha='center', va='center', color='red')
            axes[idx].axis('off')

        plt.tight_layout()
        plt.show()

# --- EJECUCIÓN ---
ruta_us = "./data_ready_US"
visualizador_US_triple(ruta_us)