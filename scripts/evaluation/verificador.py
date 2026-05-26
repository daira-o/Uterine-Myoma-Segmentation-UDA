import numpy as np

img = np.load("./data_ready_US/test/images/15cm_1.2.826.0.1.3680043.2.461.11522581.1739265486.npy")

print(f"Formato: {img.shape} | Tipo: {img.dtype}")
print(f"Rango de valores: Mín={img.min():.2f}, Máx={img.max():.2f}")
print(f"Porcentaje de fondo negro: {(img == 0).sum() / img.size * 100:.1f}%")