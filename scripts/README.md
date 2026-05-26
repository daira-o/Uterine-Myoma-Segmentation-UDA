# Scripts

Entry points are grouped by data science workflow stage:

- `data_preparation/`: MRI/US preprocessing, annotation cleanup and split creation.
- `training/`: source MRI training, target adaptation and non-DANN baselines.
- `inference/`: production-style prediction scripts.
- `evaluation/`: dataset review and checkpoint comparison utilities.
- `visualization/`: Streamlit dashboards and Matplotlib viewers.
- `utils/`: small operational helpers.

Common commands:

```bash
python scripts/data_preparation/mri_pipeline.py
python scripts/data_preparation/us_pipeline.py
python scripts/training/train_source.py
python scripts/training/train_target.py
python scripts/inference/infer_us_production.py --input data_ready_US/test/images
streamlit run scripts/visualization/visualizar_modelo.py
streamlit run scripts/visualization/visualizar_us_modelo.py
```
