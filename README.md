# Segmentacion de Miomas Uterinos con Adaptacion de Dominio

Proyecto para segmentar miomas uterinos combinando RM sagital T2 como dominio fuente y ultrasonido como dominio objetivo. El flujo actual cubre preparacion de datos, entrenamiento source, adaptacion target con DANN/supervision debil, evaluacion de checkpoints, inferencia y visualizacion.

## Estructura

```text
Uterine-Myoma-Segmentation-UDA/
+-- config.py
+-- models/
|   +-- attention_unet.py
|   +-- attention_unet_dann.py
|   +-- dann_unet.py
|   +-- domain_discriminator.py
|   +-- grl.py
+-- scripts/
|   +-- data_preparation/
|   +-- training/
|   +-- inference/
|   +-- evaluation/
|   +-- visualization/
|   +-- utils/
+-- docs/
|   +-- dann_implementation.md
+-- logs/
+-- requirements.txt
```

## Configuracion

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Crear un `.env` en la raiz del proyecto. Las rutas pueden ser absolutas o relativas al repo.

```env
DATA_PATH=data_ready_RM
MRI_DATA_PATH=data/UMD
MRI_OUTPUT_PATH=data_ready_RM
NIFTI_ROOT=data/UMD
NIFTI_IMG_SUFFIX=_t2
NIFTI_MASK_SUFFIX=_seg

US_DATA_PATH=data/Ultrasound
US_OUTPUT_PATH=data_ready_US
US_READY_PATH=data_ready_US

MODEL_PATH=best_model_sagital.pth
LOGS_PATH=logs
OUTPUTS_PATH=outputs

BATCH_SIZE=8
EPOCHS=30
LR=1e-4
USE_DANN=0
LAMBDA_DOMAIN=0.0

WEAK_LOSS_TYPE=soft_bbox
WEAK_BBOX_MARGIN_PX=4
WEAK_OUTSIDE_WEIGHT=1.0
WEAK_INSIDE_WEIGHT=2.0
WEAK_AREA_WEIGHT=0.10
US_METRIC_THRESHOLDS=0.30,0.35,0.40,0.50
```

## Flujo Principal

Preparar RM desde NIfTI a `.npy`:

```bash
python scripts/data_preparation/mri_pipeline.py
```

Preparar ultrasonido y anotaciones:

```bash
python scripts/data_preparation/us_pipeline.py
python scripts/data_preparation/organize_us_annotations.py --dry-run
python scripts/data_preparation/organize_us_annotations.py
python scripts/data_preparation/build_us_splits_from_clean.py
```

Entrenar el modelo source sobre RM:

```bash
python scripts/training/train_source.py
```

Adaptar a US con entrenamiento target:

```bash
python scripts/training/train_target.py
```

Comparar checkpoints sobre US:

```bash
python scripts/evaluation/compare_checkpoints_us.py --help
```

Ejecutar inferencia de produccion:

```bash
python scripts/inference/infer_us_production.py --help
```

## Visualizacion

Dashboard para RM:

```bash
streamlit run scripts/visualization/visualizar_modelo.py
```

Dashboard para US:

```bash
streamlit run scripts/visualization/visualizar_us_modelo.py
```

Visualizadores rapidos con Matplotlib:

```bash
python scripts/visualization/visualizador_mri.py
python scripts/visualization/visualizador_us.py
python scripts/visualization/visualizar_epocas_target.py
```

## Modelos

- `AttentionUNet`: arquitectura base para segmentacion binaria.
- `DANNUNet` / `AttentionUNetDANN`: variantes con adaptacion de dominio.
- `GradientReversalLayer`: invierte gradientes para entrenamiento adversarial.
- `DomainDiscriminator`: predice dominio MRI/US desde features del encoder.

## Metricas y Logs

Los entrenamientos escriben metricas y checkpoints en `logs/`. Las metricas principales son:

- Dice coefficient.
- HD95.
- Object Precision.
- Perdidas de segmentacion, dominio y supervision debil.
- Metricas US por umbral cuando corresponde.

Los artefactos generados en `logs/`, `outputs/`, checkpoints y previews de debug no deberian versionarse salvo que se necesiten para una entrega especifica.
