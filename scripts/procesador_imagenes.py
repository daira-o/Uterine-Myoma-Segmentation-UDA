import nibabel as nib
import numpy as np
import os
import cv2
from glob import glob
from tqdm import tqdm

def universal_processor(base_path, output_path, img_suffix='_t2', mask_suffix='_seg'):
    """
    base_path: Carpeta donde están las subcarpetas de pacientes.
    output_path: Carpeta de destino para los .npy.
    img_suffix: El final del nombre del archivo de imagen (ej: '_t2' o '_mri').
    mask_suffix: El final del nombre del archivo de máscara (ej: '_seg' o '_mask').
    """
    out_img = os.path.join(output_path, 'images_npy')
    out_mask = os.path.join(output_path, 'masks_npy')
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_mask, exist_ok=True)

    # Buscamos todas las subcarpetas de pacientes
    patient_folders = [f for f in glob(os.path.join(base_path, "*")) if os.path.isdir(f)]
    
    for folder in tqdm(patient_folders, desc="Procesando Dataset"):
        patient_id = os.path.basename(folder)
        
        # 1. Validación de existencia: ¿Ya procesamos a este paciente?
        # Si ya existe al menos un archivo con ese ID, pasamos al siguiente
        if glob(os.path.join(out_img, f"{patient_id}_*")):
            continue 

        # 2. Búsqueda de archivos
        img_nii = glob(os.path.join(folder, f"*{img_suffix}.nii*"))
        seg_nii = glob(os.path.join(folder, f"*{mask_suffix}.nii*"))

        if not img_nii or not seg_nii:
            continue 

        # 3. Carga con "Escudo" contra archivos corruptos (como el 037)
        try:
            data_img = nib.load(img_nii[0]).get_fdata()
            data_seg = nib.load(seg_nii[0]).get_fdata()
        except Exception as e:
            print(f"\n Error en {patient_id}: {e}. Salteando...")
            continue

        # ── Detectar el eje sagital via affine ────────────────────────────────
        import nibabel as nib_local
        affine = nib_local.load(img_nii[0]).affine
        codes  = nib_local.aff2axcodes(affine)
        # Sagital = direccion Left/Right
        sag_axis = next((i for i, c in enumerate(codes) if c in ("L", "R")), None)
        if sag_axis is None:
            # Fallback: eje con menos slices (tipicamente el sagital en pelvis)
            sag_axis = int(np.argmin(data_img.shape))

        n_sag = data_img.shape[sag_axis]

        # ── Extraer los N cortes sagitales reales ─────────────────────────
        for i in range(n_sag):
            if sag_axis == 0:
                img_slice = data_img[i, :, :]
                seg_slice = data_seg[i, :, :]
            elif sag_axis == 1:
                img_slice = data_img[:, i, :]
                seg_slice = data_seg[:, i, :]
            else:
                img_slice = data_img[:, :, i]
                seg_slice = data_seg[:, :, i]

            # 1. Rotar para poner vertical (90° horario)
            img_slice = np.rot90(img_slice, k=3)
            seg_slice = np.rot90(seg_slice, k=3)
            
            # 2. Voltear horizontalmente para cumplir regla Anterior-Izquierda / Posterior-Derecha
            # np.fliplr invierte las columnas (espejo horizontal)
            img_slice = np.fliplr(img_slice)
            seg_slice = np.fliplr(seg_slice)

            # Filtro: solo cortes con mioma (vital para no sesgar la red)
            if np.sum(seg_slice) > 0:
                # Normalización Min-Max
                diff = np.max(img_slice) - np.min(img_slice)
                img_norm = (img_slice - np.min(img_slice)) / (diff if diff != 0 else 1)

                # Resize y Binarización
                img_res = cv2.resize(img_norm, (256, 256), interpolation=cv2.INTER_AREA)
                seg_res = cv2.resize(seg_slice, (256, 256), interpolation=cv2.INTER_NEAREST)
                seg_res = (seg_res > 0).astype(np.float32)

                # Guardar — formato: {patient_id}_sag_{i} donde i es el indice sagital real
                file_id = f"{patient_id}_sag_{i}"
                np.save(os.path.join(out_img, f"{file_id}.npy"), img_res)
                np.save(os.path.join(out_mask, f"{file_id}.npy"), seg_res)

# --- CONFIGURACIÓN: CAMBIA ESTO SEGÚN EL DATASET ---
if __name__ == "__main__":
    # PARA EL UMD 
   universal_processor(
        base_path="C:/Users/Daira/Documents/Uterine-Myoma-Segmentation-UDA/data/UMD", 
        output_path="C:/Users/Daira/Documents/Uterine-Myoma-Segmentation-UDA/data_ready_RM",
        img_suffix="_t2", 
        mask_suffix="_seg"
   )
    # CUANDO TENGAS EL DE ULTRASONIDO (EJEMPLO):
    # universal_processor(
    #     base_path="./data/Ultrasonido", 
    #     output_path="./data_ready_US",
    #     img_suffix="_us", 
    #     mask_suffix="_mask"
    # )