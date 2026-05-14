# Segmentación de Miomas Uterinos

Attention U-Net para segmentación de miomas en RM sagital T2.

## Estructura

```
proyecto/
├── .env                        ← rutas locales (NO subir a git)
├── .gitignore
├── config.py                   ← configuración central (lee .env)
├── models/
│   └── attention_unet.py       ← arquitectura + métricas
├── scripts/
│   ├── procesador_imagenes.py  ← NIfTI → .npy listos para entrenar
│   ├── train.py                ← entrenamiento con patient-level split
│   ├── visualizar_modelo.py    ← dashboard Streamlit
│   └── visualizadores/         ← utilidades interactivas de auditoría
└── logs/
    └── training_metrics.csv    ← métricas por época (auto-generado)
```

## Setup

```bash
pip install torch numpy scikit-learn scipy scikit-image nibabel \
            streamlit matplotlib python-dotenv Pillow opencv-python
```

Crear el archivo `.env` en la raíz del proyecto:
```
DATA_PATH=C:/ruta/a/data_ready_RM
NIFTI_ROOT=C:/ruta/a/data/UMD
NIFTI_IMG_SUFFIX=_t2
US_DATA_PATH=C:/ruta/a/data/Ultrasound
US_OUTPUT_PATH=C:/ruta/a/data_ready_US
MODEL_PATH=best_model_sagital.pth
LOGS_PATH=logs
```

## Uso

```bash
# 1. Procesar NIfTI originales a .npy
python scripts/procesador_imagenes.py

# 2. Entrenar
python scripts/train.py

# 3. Visualizar
streamlit run scripts/visualizar_modelo.py

# Auditoría rápida MRI mid vs US F28
python scripts/visualizadores/comparar_mid_random.py
```

## Métricas guardadas

Cada run de entrenamiento agrega filas al archivo `logs/training_metrics.csv`:

| columna | descripción |
|---|---|
| run_id | timestamp del run (YYYYMMDD_HHMMSS) |
| epoch | número de época |
| avg_loss | pérdida promedio de entrenamiento |
| dice | Dice coefficient en validación |
| hd95 | Hausdorff 95% en píxeles |
| obj_precision | precisión a nivel de instancia |
| hd95_inf_batches | batches con predicción vacía |
| is_best | 1 si fue el mejor Dice hasta ese momento |
| timestamp | fecha y hora exacta |
