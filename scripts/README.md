# Scripts

Los scripts estan agrupados por etapa del flujo de trabajo. La ruta raiz y las variables de entorno se resuelven desde `config.py`, asi que conviene ejecutar los comandos desde la raiz del repo.

## Preparacion de Datos

- `data_preparation/mri_pipeline.py`: convierte NIfTI de RM a tiles `.npy`, separando train/val/test por paciente.
- `data_preparation/us_pipeline.py`: prepara imagenes de ultrasonido para el dominio target.
- `data_preparation/organize_us_annotations.py`: alinea XML/anotaciones con imagenes US; usar primero con `--dry-run`.
- `data_preparation/build_us_splits_from_clean.py`: genera splits curados de US.
- `data_preparation/procesador_imagenes_us.py`: normalizacion fisica de US a 0.8 mm/px.
- `data_preparation/us_nosano_processor.py`: procesador legacy para el dataset US No Sano.

Comandos habituales:

```bash
python scripts/data_preparation/mri_pipeline.py
python scripts/data_preparation/us_pipeline.py
python scripts/data_preparation/organize_us_annotations.py --dry-run
python scripts/data_preparation/build_us_splits_from_clean.py --help
```

## Entrenamiento

- `training/train_source.py`: entrena Attention U-Net sobre RM.
- `training/train_target.py`: ajusta el modelo a US con supervision debil; opcionalmente permite DANN si se activa en la configuracion.

```bash
python scripts/training/train_source.py
python scripts/training/train_target.py
```

## Inferencia

- `inference/infer_us_production.py`: inferencia de produccion para US.
- `inference/postprocessing.py`: utilidades de postprocesamiento.

```bash
python scripts/inference/infer_us_production.py --help
```

## Evaluacion

- `evaluation/compare_checkpoints_us.py`: compara checkpoints sobre US.
- `evaluation/review_us_dataset.py`: curador visual para revisar muestras US.
- `evaluation/verificador.py`: utilidad puntual para inspeccionar arrays.

```bash
python scripts/evaluation/compare_checkpoints_us.py --help
python scripts/evaluation/review_us_dataset.py --help
```

## Visualizacion

- `visualization/visualizar_modelo.py`: dashboard Streamlit para RM/modelo source.
- `visualization/visualizar_us_modelo.py`: dashboard Streamlit para inferencia y revision del modelo ajustado a US.
- `visualization/visualizador_mri.py`: comparacion rapida entre NIfTI original y tile procesado.
- `visualization/visualizador_us.py`: inspeccion rapida de US procesado.
- `visualization/visualizar_epocas_target.py`: revisa la evolucion visual del entrenamiento target.

```bash
streamlit run scripts/visualization/visualizar_modelo.py
streamlit run scripts/visualization/visualizar_us_modelo.py
python scripts/visualization/visualizador_mri.py
python scripts/visualization/visualizador_us.py
```

## Utilidades

- `utils/chequear_gpu.py`: chequeo rapido de CUDA/GPU.

```bash
python scripts/utils/chequear_gpu.py
```
