# Sincronización temporal de los dos flujos

Las cámaras OAK-D LR y OAK-D Pro operan como dispositivos independientes y no
utilizan una señal de sincronización física. La asociación se realiza por
software con los timestamps de adquisición retornados por `getTimestamp()`.
DepthAI alinea continuamente esos timestamps al reloj monotónico del host; no
se utiliza `getTimestampDevice()`, cuyo origen es local a cada dispositivo.

Para cada stream RGB se mantiene un buffer FIFO acotado de ocho mensajes. Entre
los mensajes disponibles se selecciona el par LR/Pro que minimiza

```text
Δt = |t_LR - t_Pro|.
```

El par se acepta únicamente cuando `Δt <= 20 ms`. Si ningún par satisface ese
umbral, se elimina el mensaje más antiguo y se repite la búsqueda. Cada mensaje
aceptado se consume una sola vez, por lo que no existe duplicación ni
reutilización entre pares. Los huecos observados en los números de secuencia y
los descartes efectuados por el emparejador se contabilizan por stream.

Los mensajes auxiliares se asocian al RGB del mismo dispositivo mediante el
mismo criterio de vecino temporal más cercano. La profundidad frontal se
compara con `t_LR`; la profundidad interior y la detección facial se comparan
con `t_Pro`. El host usa una ventana común de espera de hasta 100 ms para los
auxiliares del par. Si el mensaje
más cercano excede 20 ms, ese dato se considera ausente y no se conserva una
medición de un frame anterior.

La inferencia de mirada se ejecuta sobre el frame Pro ya emparejado y su punto
se evalúa sobre el frame LR del mismo par. La latencia desde el timestamp de
captura más reciente del par hasta la decisión ADAS se mide con el mismo reloj
monotónico y se reporta por separado. No se extrapolan la mirada ni la posición
de los objetos durante esa latencia.

El reporte generado con `--save-reports` conserva el umbral configurado, el
número de pares aceptados, mensajes recibidos, descartes, huecos de secuencia,
disponibilidad de auxiliares y los valores medio, máximo y percentil 95 tanto
del desfase LR/Pro como de la latencia captura-decisión.
