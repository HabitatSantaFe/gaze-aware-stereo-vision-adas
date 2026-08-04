# Dependencias y modelos externos

Este repositorio contiene código propio, pero no concede derechos sobre modelos,
datasets ni bibliotecas de terceros.

- Ultralytics se utiliza para ejecutar YOLOv8n-Seg y se distribuye bajo
  AGPL-3.0 o licencia empresarial: <https://www.ultralytics.com/license>.
- PyTorch y TorchVision conservan sus licencias BSD respectivas.
- DepthAI conserva la licencia MIT de Luxonis.
- OpenCV conserva su licencia Apache-2.0.
- Open Model Zoo y sus modelos Intel se ofrecen bajo Apache-2.0. Los blobs se
  obtienen separadamente de la fuente oficial y conservan sus avisos.
- Los checkpoints entrenados con ETH-XGaze no se redistribuyen. Sus términos
  prohíben distribuir parámetros originales o convertidos y limitan el uso a
  investigación académica no comercial:
  <https://xgaze.ait.ethz.ch/>.

La ausencia de pesos es deliberada. Cada usuario debe obtenerlos legalmente y
proporcionar sus rutas mediante los argumentos del programa.
