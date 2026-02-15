# LSE TCN MVP (skeleton-only, realtime)

MVP funcional para reconocimiento de signos aislados de LSE en tiempo real, usando:
- **MediaPipe Holistic** para keypoints de manos + pose superior.
- **TCN ligera residual** para clasificación temporal.
- **Privacidad por diseño**: el modelo consume solo coordenadas normalizadas (`[T, 144]`), nunca RGB.

## Estructura

```text
lse_tcn/
  data/
  lse_tcn/
    models/tcn.py
    data/adapters_swl_lse.py
    data/adapters_sign4all.py
    data/preprocess.py
    data/augment.py
    train.py
    eval.py
    export_onnx.py
    realtime_demo.py
    utils/config.py
  configs/social50.yaml
  tests/
  requirements.txt
```

## Instalación rápida (Linux/macOS)

```bash
cd lse_tcn
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Guía **paso a paso en Windows** (nivel principiante)

> Esta sección está pensada para seguirla literalmente en Windows 10/11, usando **PowerShell**.

## 1) Instalar Python

1. Descarga Python 3.10+ desde: https://www.python.org/downloads/windows/
2. Al instalar, marca la casilla: **Add Python to PATH**.
3. Abre PowerShell y verifica:

```powershell
python --version
```

Debes ver algo como `Python 3.11.x` o similar.

## 2) Crear carpeta del proyecto

Ejemplo de ruta (puedes copiar/pegar):

```powershell
mkdir C:\Users\TU_USUARIO\proyectos\lse_tcn_mvp
cd C:\Users\TU_USUARIO\proyectos\lse_tcn_mvp
```

Copia aquí el contenido de este repo (incluyendo la carpeta `lse_tcn`).

## 3) Entrar al proyecto y crear entorno virtual

```powershell
cd C:\Users\TU_USUARIO\proyectos\lse_tcn_mvp\lse_tcn
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Si PowerShell bloquea scripts, ejecuta una sola vez:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

y vuelve a activar:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 4) Instalar dependencias

```powershell
pip install -U pip
pip install -r requirements.txt
```

## 5) Preparar carpetas de datos (ya con rutas listas)

Crea exactamente estas rutas:

```powershell
mkdir C:\Users\TU_USUARIO\proyectos\lse_tcn_mvp\lse_tcn\data\raw\swl_lse
mkdir C:\Users\TU_USUARIO\proyectos\lse_tcn_mvp\lse_tcn\data\raw\sign4all
mkdir C:\Users\TU_USUARIO\proyectos\lse_tcn_mvp\lse_tcn\data\processed
mkdir C:\Users\TU_USUARIO\proyectos\lse_tcn_mvp\lse_tcn\outputs
```

Y coloca tus datos así:
- SWL-LSE: `*.pkl` en `data\raw\swl_lse\`
- Sign4all: `*.h5` en `data\raw\sign4all\`

## 6) Formato esperado de cada muestra (muy importante)

### SWL-LSE (`.pkl`)
Cada archivo debería incluir como mínimo:
- `frames`: array `float32[T, 144]`
- `label`: string (ej. `"hola"`)
- `signer_id` (opcional)

### Sign4all (`.h5`)
Cada archivo debería incluir:
- dataset `frames`: `float32[T, 144]`
- atributo `label`
- atributo `signer_id` (opcional)

## 7) Revisar labels del config

Abre `configs\social50.yaml` y confirma que los nombres de `labels:` coinciden con los de tu dataset.

Si tu dataset usa nombres distintos (por ejemplo `adios_chao` en vez de `adios`), cámbialos en YAML para que coincidan.

## 8) Comando de entrenamiento (copiar/pegar)

Con el entorno virtual activo y en la carpeta `lse_tcn`:

```powershell
python -m lse_tcn.train --config configs/social50.yaml --epochs 25 --batch 32 --lr 0.001 --device cpu --finetune
```

### ¿Qué hace este comando?
- Lee datos desde `data\raw\swl_lse` y/o `data\raw\sign4all`.
- Filtra por labels del Social50.
- Entrena el TCN.
- Guarda caché en: `data\processed\social50_dataset.pt`
- Guarda mejor modelo en: `outputs\best_model.pt`

## 9) Si tienes GPU NVIDIA (opcional)

Prueba primero CPU. Si ya tienes PyTorch con CUDA instalado correctamente:

```powershell
python -m lse_tcn.train --config configs/social50.yaml --epochs 25 --batch 32 --lr 0.001 --device cuda --finetune
```

Si falla `cuda`, vuelve a `--device cpu`.

## 10) Evaluar modelo entrenado

```powershell
python -m lse_tcn.eval --model outputs/best_model.pt --dataset data/processed/social50_dataset.pt --device cpu
```

Con LOSO (si hay `signer_id`):

```powershell
python -m lse_tcn.eval --model outputs/best_model.pt --dataset data/processed/social50_dataset.pt --loso --device cpu
```

## 11) Ejecutar demo en tiempo real con webcam

```powershell
python -m lse_tcn.realtime_demo --model outputs/best_model.pt --webcam-index 0 --window-size 60 --infer-stride 5 --movement-threshold 0.015 --smoothing-window 5 --confidence-threshold 0.75 --device cpu
```

- Pulsa `q` para salir.
- Verás overlay con label, confianza, FPS y latencia.
- Si la confianza no supera 0.75, responde `NO_ENTIENDO`.

## 12) Exportar a ONNX (opcional)

```powershell
python -m lse_tcn.export_onnx --model outputs/best_model.pt --output outputs/lse_tcn.onnx
```

## 13) Errores típicos y solución rápida

- **`No samples found`**
  - No hay `.pkl/.h5` en las rutas esperadas o labels no coinciden con el YAML.

- **`ModuleNotFoundError`**
  - Entorno virtual no activado. Ejecuta `\.venv\Scripts\Activate.ps1`.

- **`Cannot open webcam`**
  - Cámara ocupada por otra app o índice incorrecto. Prueba `--webcam-index 1`.

- **Entrena pero no aprende bien**
  - Revisa calidad de keypoints, balance de clases y consistencia de labels.

---

## Entrenamiento (resumen general)

### Fine-tuning Social50
```bash
python -m lse_tcn.train --config configs/social50.yaml --epochs 25 --batch 32 --lr 1e-3 --device cpu --finetune
```

### Pretrain (si tienes SWL-LSE completo)
Usa un config alternativo (p.ej. `configs/swl300.yaml`) con 300 clases:
```bash
python -m lse_tcn.train --config configs/swl300.yaml --pretrain --device cuda
```

El script guarda:
- caché de samples en `data/processed/social50_dataset.pt`
- mejor checkpoint en `outputs/best_model.pt`

## Evaluación

```bash
python -m lse_tcn.eval --model outputs/best_model.pt --dataset data/processed/social50_dataset.pt --device cpu
```

Modo LOSO (si `signer_id` está disponible):
```bash
python -m lse_tcn.eval --model outputs/best_model.pt --dataset data/processed/social50_dataset.pt --loso
```

Métricas:
- accuracy
- macro-F1
- matriz de confusión (`outputs/confusion_matrix.png`)

## Demo realtime

```bash
python -m lse_tcn.realtime_demo \
  --model outputs/best_model.pt \
  --webcam-index 0 \
  --window-size 60 \
  --infer-stride 5 \
  --movement-threshold 0.015 \
  --smoothing-window 5 \
  --confidence-threshold 0.75 \
  --device cpu
```

Incluye:
- buffer circular + sliding window
- inferencia cada `K` frames
- **activity gating** por magnitud de movimiento de manos
- smoothing temporal por media móvil de probabilidades
- fallback a `NO_ENTIENDO` si confianza < umbral
- overlay: etiqueta, confianza, FPS y latencia

## Export ONNX

```bash
python -m lse_tcn.export_onnx --model outputs/best_model.pt --output outputs/lse_tcn.onnx
```

## Notas de diseño

- Features/frame: 48 puntos (21 mano izq + 21 mano der + 6 pose superior), xyz => 144.
- Normalización geométrica por frame:
  - pivot = punto medio entre hombros
  - traslación por pivot
  - escala por ancho de hombros
  - rotación opcional para alinear hombros horizontalmente (`--rotate-align`)
- Clase `GARBAGE_SILENCIO` incluida en `social50.yaml` para entrenamiento explícito cuando haya datos.

## Tests smoke

```bash
pytest -q
```

Incluye tests para:
- normalización (pivot/escala)
- forward del modelo TCN
- lógica de buffer/streaming sin webcam (datos sintéticos)
