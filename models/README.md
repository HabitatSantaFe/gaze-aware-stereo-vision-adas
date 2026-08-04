# Modelos externos

Esta carpeta se publica vacía. Los pesos no forman parte del repositorio.

Archivos esperados por la configuración predeterminada:

- `resnet101_v2.pt`: checkpoint de mirada entrenado con ETH-XGaze. **No puede
  redistribuirse** Los
  términos de ETH-XGaze prohíben redistribuir parámetros originales o
  convertidos y restringen su uso a investigación académica no comercial. Cada
  usuario debe tener autorización directa y preparar su propio checkpoint.
- `yolov8n-seg.pt`: modelo de segmentación compatible con Ultralytics. Debe
  obtenerse de su fuente oficial bajo AGPL-3.0 o una licencia comercial.
- `face-detection-retail-0004.blob`: detector facial de Open Model Zoo,
  distribuido separadamente bajo Apache-2.0.
- `facial-landmarks-35-adas-0002.blob`: localizador de Open Model Zoo bajo
  Apache-2.0, con entrada 60 x 60 y salida de 70 coordenadas normalizadas.

Se pueden indicar otras rutas mediante `--gaze-model`, `--yolo-model`,
`--face-detector-model` y `--landmark-model`. Un modelo YOLO alternativo debe conservar una salida
de segmentación compatible; un detector que solo produzca cajas no es un
reemplazo directo.
