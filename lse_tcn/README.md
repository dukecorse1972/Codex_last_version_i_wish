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

## Instalación

```bash
cd lse_tcn
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Preparar datos

Formato interno esperado por muestra:

```python
sample = {
  "frames": np.ndarray[T, 144],
  "label": int,
  "label_name": str,
  "signer_id": str,
  "source": "swl_lse" | "sign4all"
}
```

### SWL-LSE adapter (`.pkl`)
Cada `.pkl` debe tener al menos:
- `frames`: `float32[T, 144]`
- `label`
- opcional `signer_id`

### Sign4all adapter (`.h5`)
Cada `.h5` debe tener:
- dataset `frames`: `float32[T, 144]`
- atributo `label`
- opcional atributo `signer_id`

Ajusta rutas en `configs/social50.yaml`.

## Entrenamiento

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
