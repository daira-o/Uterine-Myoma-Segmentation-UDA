# Documentacion de la implementacion DANN MRI -> US

Este documento describe la integracion de Domain-Adversarial Neural Network
(DANN) sobre la Attention U-Net del proyecto para adaptar segmentacion de miomas
desde resonancia magnetica (MRI/RM) hacia ecografia (US).

## Objetivo

La Fase 1 entrena una `AttentionUNet` con MRI y mascaras. La Fase 2 reutiliza
esos pesos y agrega adaptacion de dominio adversarial:

```text
MRI -> Encoder -> Bottleneck
                 |-> Decoder -> seg_logits
                 |-> GRL -> DomainDiscriminator -> domain_logits MRI=0

US  -> Encoder -> Bottleneck
                 |-> GRL -> DomainDiscriminator -> domain_logits US=1
```

La idea es que el discriminator aprenda a distinguir MRI de US, mientras que el
encoder, por efecto de la Gradient Reversal Layer (GRL), aprende features mas
invariantes al dominio.

## Archivos agregados o modificados

### `models/attention_unet.py`

Este archivo queda como la Attention U-Net original de Fase 1.

No se usa para exponer features en DANN. Se conserva como backup limpio y sigue
siendo compatible con:

```python
from models.attention_unet import AttentionUNet

model = AttentionUNet(in_channels=1, num_classes=1)
logits = model(images)
```

### `models/attention_unet_dann.py`

Contiene la clase `AttentionUNetDANN`, que hereda de `AttentionUNet`.

No duplica la arquitectura completa: reutiliza los modulos ya definidos por la
U-Net original (`enc1`, `enc2`, `enc3`, `enc4`, `bottleneck`, decoder, attention
gates y `output_conv`).

Agrega estos metodos:

```python
features = model.encode(x)
seg_logits = model.decode(features)
bottleneck = model.extract_features(x)
seg_logits, features = model(x, return_features=True)
```

Las features devueltas tienen esta estructura:

```python
{
    "enc1": e1,
    "enc2": e2,
    "enc3": e3,
    "enc4": e4,
    "bottleneck": b,
}
```

El bottleneck es la feature principal que se conecta al discriminator.

### `models/dann_unet.py`

Contiene la clase principal de Fase 2:

```python
from models.dann_unet import DANNUNet

model = DANNUNet(in_channels=1, num_classes=1)
```

Internamente tiene:

```text
DANNUNet
|-- segmenter: AttentionUNetDANN
|-- grl: GradientReversalLayer
|-- domain_discriminator: DomainDiscriminator
```

Forward para MRI:

```python
seg_logits, domain_logits = model(x_mri, alpha=alpha)
```

Forward para US:

```python
_, domain_logits_us = model(
    x_us,
    alpha=alpha,
    return_segmentation=False,
)
```

Cuando `return_segmentation=False`, no se ejecuta el decoder. Esto reduce costo
en US, donde no hay mascaras y no se calcula perdida de segmentacion.

### `models/grl.py`

Implementa la Gradient Reversal Layer.

En forward:

```text
GRL(x) = x
```

En backward:

```text
dL/dx = -lambda * grad
```

Esto hace que:

- El `DomainDiscriminator` minimice la perdida de dominio.
- El encoder reciba el gradiente invertido y aprenda a confundir al discriminator.

Tambien contiene el schedule:

```python
alpha = 2 / (1 + exp(-10 * p)) - 1
```

donde:

```python
p = current_step / total_steps
```

### `models/domain_discriminator.py`

Implementa el discriminator de dominio.

Entrada recomendada:

```text
bottleneck features [B, C, H, W]
```

Salida:

```text
domain_logits [B, 2]
```

Las etiquetas usadas son:

```python
MRI_DOMAIN = 0
US_DOMAIN = 1
```

La loss correcta es:

```python
torch.nn.functional.cross_entropy(domain_logits, domain_labels)
```

No se aplica softmax antes de la loss.

### `scripts/train_target.py`

Entrena la Fase 2 completa.

Hace:

- Carga MRI train/val igual que `scripts/train_source.py`.
- Carga US desde `data_ready_US/train/images`.
- Carga pesos preentrenados de Fase 1 dentro de `DANNUNet.segmenter`.
- Entrena con batches simultaneos MRI + US.
- Calcula segmentation loss solo para MRI.
- Calcula domain loss para MRI y US.
- Valida sobre MRI val.
- Guarda `best_model_dann.pth` y `last_model_dann.pth`.
- Loggea metricas en CSV.

## Datos esperados

MRI:

```text
data_ready_RM/
|-- train/
|   |-- images/*.npy
|   |-- masks/*.npy
|-- val/
|   |-- images/*.npy
|   |-- masks/*.npy
|-- test/
|   |-- images/*.npy
|   |-- masks/*.npy
```

US:

```text
data_ready_US/
|-- train/
|   |-- images/*.npy
|-- val/
|   |-- images/*.npy
|-- test/
|   |-- images/*.npy
```

US no necesita mascaras para DANN.

## Carga de pesos de Fase 1

El checkpoint de Fase 1 se toma desde:

```python
CONFIG["model_path"]
```

En tu proyecto apunta a:

```text
best_model_sagital.pth
```

La carga ocurre en:

```python
model.load_pretrained_segmenter(checkpoint_path, strict=False)
```

Esto carga pesos solamente en:

```text
DANNUNet.segmenter
```

Pesos cargados desde Fase 1:

- Encoder de la U-Net.
- Bottleneck.
- Decoder.
- Attention gates.
- Capa final de segmentacion.

Pesos inicializados desde cero:

- `DomainDiscriminator`.
- Capas internas propias del discriminator.

La GRL no tiene pesos entrenables.

## Loss total

Por cada step:

```python
seg_loss = bce_dice_loss(seg_logits_mri, masks_mri)

domain_loss_mri = cross_entropy(domain_logits_mri, labels_mri_0)
domain_loss_us = cross_entropy(domain_logits_us, labels_us_1)

domain_loss = (domain_loss_mri + domain_loss_us) / 2

total_loss = seg_loss + lambda_domain * domain_loss
```

Por defecto, si no se agrega otra cosa en `CONFIG`, se usa:

```python
lambda_domain = 0.1
```

## Que modulos reciben gradientes

### Batch MRI

MRI tiene imagen y mascara.

Reciben gradientes por segmentation loss:

- Encoder.
- Bottleneck.
- Decoder.
- Attention gates.
- `output_conv`.

Reciben gradientes por domain loss:

- `DomainDiscriminator`, gradiente normal.
- Encoder + bottleneck, gradiente invertido por GRL.

### Batch US

US tiene imagen sin mascara.

No se calcula segmentation loss.

Reciben gradientes por domain loss:

- `DomainDiscriminator`, gradiente normal.
- Encoder + bottleneck, gradiente invertido por GRL.

El decoder no se ejecuta para US cuando:

```python
return_segmentation=False
```

Por eso no recibe gradientes desde US.

## Dataloaders y cuidado de temperatura

El trainer usa:

```python
num_workers = CONFIG.get("num_workers", 0)
```

Si no se define `num_workers`, queda en `0`. Esto es mas conservador para CPU,
RAM y temperatura, especialmente en Windows.

En entrenamiento se usa:

```python
us_iter = cycle(us_loader)
```

La epoca se define por el loader MRI. Si US tiene menos batches, se recicla. Esto
evita quedarse sin US y mantiene un batch MRI + un batch US en cada step.

Importante: `cycle(us_loader)` no carga todo el dataset en memoria. Va pidiendo
batches al `DataLoader`.

Para una computadora sensible a temperatura, empezar con:

```python
"batch_size": 2,
"num_workers": 0,
"epochs": 3,
"lambda_domain": 0.05,
```

Luego subir a `batch_size=4` si la temperatura y la memoria lo permiten.

## Checkpoints

Se guardan en:

```text
logs/checkpoints_dann/
|-- best_model_dann.pth
|-- last_model_dann.pth
```

`best_model_dann.pth` se actualiza cuando mejora el Dice de validacion.

`last_model_dann.pth` se actualiza al final de cada epoca.

Si se detiene el entrenamiento con `Ctrl+C`:

- Si ya termino al menos una epoca, queda guardado el mejor modelo anterior.
- Si se corta en medio de una epoca, esa epoca parcial no se guarda.

Cada checkpoint contiene:

```python
{
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "segmenter_state_dict": model.segmenter.state_dict(),
    "domain_discriminator_state_dict": model.domain_discriminator.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "best_dice": best_dice,
    "metrics": val_metrics,
    "alpha": alpha,
    "lambda_domain": lambda_domain,
    "config": CONFIG,
}
```

## Logs

Durante entrenamiento se imprimen lineas como:

```text
Epoca 01/30 | total 0.1209 | seg 0.0244 | dom 0.9650 |
val_loss 0.0953 | Dice 0.8824 | HD95 8.05px | dom_acc 0.511 | 143.1s * BEST
```

Interpretacion:

- `total`: loss total usada para backward.
- `seg`: loss de segmentacion en MRI.
- `dom`: perdida de dominio promedio MRI/US.
- `val_loss`: loss de segmentacion sobre MRI val.
- `Dice`: Dice en MRI val.
- `HD95`: distancia Hausdorff 95 en validacion.
- `dom_acc`: accuracy del discriminator.
- `* BEST`: se guardo nuevo mejor checkpoint.

En DANN, un `dom_acc` cerca de `0.5` puede ser una buena senal: significa que el
discriminator tiene dificultades para separar MRI de US. Si `dom_acc` sube mucho,
el discriminator esta distinguiendo dominios con facilidad.

## Ejecucion

Desde la raiz del proyecto:

```powershell
python scripts/train_target.py
```

Si se usa el Python local configurado en este entorno:

```powershell
& "C:\Users\Daira\AppData\Local\Python\pythoncore-3.12-64\python.exe" scripts\train_target.py
```

## Fase 3: inferencia de produccion

La Fase 3 usa solo el segmentador adaptado. En produccion no se necesita:

- `GradientReversalLayer`.
- `DomainDiscriminator`.
- Loss de dominio.
- Etiquetas MRI/US.

El flujo final es:

```text
US 256x256 a 0.8 mm/px
    -> AttentionUNet encoder adaptado
    -> bottleneck alineado
    -> AttentionUNet decoder
    -> mascara final
```

El script implementado es:

```text
scripts/infer_us_production.py
```

Este script carga el checkpoint DANN, extrae solo:

```python
segmenter_state_dict
```

y lo coloca dentro de una `AttentionUNet` normal. De esta manera, GRL y
discriminator quedan descartados por completo.

Ejemplo con una carpeta US:

```powershell
python scripts/infer_us_production.py --input data_ready_US/test/images
```

Ejemplo con una sola imagen:

```powershell
python scripts/infer_us_production.py --input data_ready_US/test/images/imagen.npy
```

Usando checkpoint explicito:

```powershell
python scripts/infer_us_production.py ^
  --input data_ready_US/test/images ^
  --checkpoint logs/checkpoints_dann/best_model_dann.pth ^
  --threshold 0.5 ^
  --save-prob
```

Salidas por defecto:

```text
outputs/phase3_inference/
|-- masks/
|   |-- *_mask.npy
|-- probabilities/
|   |-- *_prob.npy      # solo si se usa --save-prob
|-- *_mask.png
|-- *_prob.png
|-- *_overlay.png
```

Requisitos de entrada:

- Imagen US ya estandarizada a `256x256`.
- Intensidades normalizadas o normalizables a `[0, 1]`.
- Resolucion esperada: `0.8 mm/px`, igual que la salida de `us_pipeline.py`.

Si el input no tiene shape `256x256`, el script falla de forma explicita para
evitar inferir sobre una imagen no estandarizada.

### Visualizador US de Fase 3

Tambien se agrego un dashboard Streamlit para inspeccionar predicciones sobre
ultrasonido:

```text
scripts/visualizar_us_modelo.py
```

Ejecucion:

```powershell
streamlit run scripts/visualizar_us_modelo.py
```

O con el Python local:

```powershell
& "C:\Users\Daira\AppData\Local\Python\pythoncore-3.12-64\python.exe" -m streamlit run scripts\visualizar_us_modelo.py
```

El visualizador:

- Carga `best_model_dann.pth`.
- Extrae solo el segmentador adaptado.
- Lista imagenes desde `data_ready_US`.
- Muestra US, overlay, mapa de probabilidad y mascara binaria.
- Calcula area en pixeles y area aproximada en mm2 usando `0.8 mm/px`.
- Permite guardar mascara, probabilidad y overlay.

## Ejemplo minimo de forward

```python
import torch
from models.dann_unet import DANNUNet

model = DANNUNet(in_channels=1, num_classes=1)

x_mri = torch.randn(2, 1, 256, 256)
x_us = torch.randn(2, 1, 256, 256)

alpha = 0.5

# MRI: segmentacion + dominio
seg_logits, domain_logits_mri = model(x_mri, alpha=alpha)
print(seg_logits.shape)          # [2, 1, 256, 256]
print(domain_logits_mri.shape)   # [2, 2]

# US: solo dominio, sin decoder
seg_us, domain_logits_us = model(
    x_us,
    alpha=alpha,
    return_segmentation=False,
)
print(seg_us)                    # None
print(domain_logits_us.shape)    # [2, 2]
```

## Checklist de verificacion

Antes de entrenar:

- `data_ready_RM/train/images` existe.
- `data_ready_RM/train/masks` existe.
- `data_ready_RM/val/images` existe.
- `data_ready_RM/val/masks` existe.
- `data_ready_US/train/images` existe.
- `best_model_sagital.pth` existe o se acepta entrenar desde cero.

Durante entrenamiento:

- `seg` baja o se mantiene razonable.
- `Dice` valida no cae bruscamente.
- `dom_acc` no se queda fijo en 1.0 durante muchas epocas.
- La temperatura de CPU/GPU es estable.
- Se genera `logs/checkpoints_dann/best_model_dann.pth`.

Despues de entrenar:

- Usar `best_model_dann.pth` para evaluacion o inferencia.
- Revisar `logs/target_training_metrics.csv`.
- Comparar Dice/HD95 contra Fase 1.
- Visualizar predicciones MRI y, si corresponde, salidas sobre US.
