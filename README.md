# Real-time ADAS gaze and planar-distance estimation

> **Alcance de esta carpeta:** código fuente y calibración numérica. No contiene
> el manuscrito, pesos, imágenes, videos ni datos de participantes. Consulta
> `LEGAL_AUDIT.md` y `LEGAL_CHECKLIST.md` antes de añadir otros archivos.

Implementación asociada al paper para estimación de mirada, detección ADAS, profundidad, TTC y métricas de rendimiento con dos cámaras OAK-D.

## Privacidad

La ejecución predeterminada no debe escribir videos, imágenes, frames de datasets ni reportes. Las métricas de la interfaz se mantienen en memoria y los identificadores físicos de las cámaras no están incluidos en el código.

Las opciones de grabación pueden almacenar imágenes del conductor. Úsalas solamente con consentimiento informado y una política de retención adecuada.

## Instalación

Las dependencias de Python 3.11 están fijadas en `requirements.txt`. La ejecución
integral requiere dos cámaras OAK-D y los pesos descritos en el paper, que no se
incluyen en esta carpeta.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Los pesos (`.pt`) y modelos DepthAI (`.blob`) no se versionan. Consulta
`models/README.md`, obtén cada archivo de una fuente autorizada y colócalo en
la ruta indicada o utiliza los argumentos de línea de comandos.

## Ejecución

```bash
python adas_gaze_realtime.py
```

La rama interior usa RGB-D: detecta el rostro, estima 35 landmarks, alinea el
recorte y calcula el origen 3D del rayo de mirada con la profundidad estéreo de
la OAK-D Pro. Si no hay landmarks o profundidad facial válida, ese frame no se
clasifica como procesado.

El ejecutable usa por defecto ResNet-101 y la calibración extrínseca
`extrinsics_pro_to_lr_no_mirror.json`.

Para fijar el orden de cámaras sin publicar sus identificadores:

```bash
python adas_gaze_realtime.py --lr-device-id MXID_1 --pro-device-id MXID_2
```

Las salidas opcionales se habilitan con `--save-reports`, `--record-video`, `--save-captures` y `--save-dataset-frames`, y se escriben en `outputs/`.

## Sincronización temporal

Las dos OAK-D se emparejan por software usando `getTimestamp()`, cuyo reloj de
dispositivo es alineado por DepthAI al reloj monotónico del host. El programa
mantiene buffers independientes y selecciona el par LR/Pro con la menor
diferencia absoluta. Por defecto sólo acepta pares con un desfase máximo de
20 ms, descarta el mensaje más antiguo cuando no encuentra una pareja válida y
no reutiliza ni duplica frames.

La profundidad LR se empareja con el timestamp RGB LR; la profundidad y la
detección facial Pro se emparejan con el timestamp RGB Pro. Un auxiliar que no
llega dentro de 100 ms o excede el umbral no se sustituye por datos antiguos.
Los umbrales se configuran con `--sync-tolerance-ms`, `--sync-buffer-size` y
`--aux-sync-wait-ms`. El reporte JSON incluye desfase medio, máximo y p95,
mensajes descartados, huecos de secuencia y latencia desde captura hasta
decisión. No se aplica extrapolación temporal de mirada u objetos.

La formulación destinada a la sección metodológica está en
`docs/TEMPORAL_SYNCHRONIZATION.md`.

## Pesos

Este repositorio no redistribuye pesos mediante Git, Releases, LFS u otro
servicio. Solo conserva las rutas esperadas y no descarga pesos silenciosamente.

## Licencia y cita

El código se publica bajo GNU AGPL-3.0 para ser compatible con la dependencia
Ultralytics. Consulta `LICENSE` y `THIRD_PARTY_NOTICES.md`. La intención del
proyecto es la investigación académica, pero la AGPL-3.0 no permite imponer una
restricción adicional de "solo uso académico".

La licencia del código no elimina las restricciones de los modelos. En
particular, el sistema completo con un checkpoint derivado de ETH-XGaze queda
limitado a investigación académica no comercial y dicho checkpoint no puede
redistribuirse.


