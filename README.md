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
│   ├── data_preparation/       ← pipelines MRI/US, curación y splits
│   ├── training/               ← entrenamientos source, target y baselines
│   ├── inference/              ← inferencia de producción
│   ├── evaluation/             ← revisión de dataset y comparación de modelos
│   ├── visualization/          ← dashboards y visualizadores
│   └── utils/                  ← utilidades operativas pequeñas
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
python scripts/data_preparation/mri_pipeline.py

# 2. Entrenar
python scripts/training/train_source.py

# 3. Visualizar
streamlit run scripts/visualization/visualizar_modelo.py

# Visualizar ultrasonido procesado
python scripts/visualization/visualizador_us.py
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
