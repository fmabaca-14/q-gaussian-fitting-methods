# Piloto sintético centrado (2026-09-23)

Ejecutar desde la raíz del repositorio:

```bash
python -m scripts.run_pilot --repetitions 100 --output results/pilot.csv
```

Requiere Python, NumPy y SciPy. El archivo `pilot.json` conserva los valores de configuración y las versiones utilizadas. `pilot.csv` tiene una fila por muestra y método; `success`, `at_bound` y `at_start` deben leerse por separado. Las 100 muestras de cada escenario son independientes, y los cuatro métodos comparten cada muestra.

| q | N | Mediana de bins ocupados (de 50) | Diagnóstico principal |
|---:|---:|---:|---|
| 1.4 | 100 | 30.5 | q-log: 83/100 soluciones en límites; CDF: 17/100 |
| 1.4 | 2000 | 35.5 | Sin fallos o soluciones en límites |
| 1.9 | 100 | 16 | PDF: 5/100 en límites; q-log: 15/100 |
| 1.9 | 2000 | 15 | PDF: 35/100 en límites; q-log: 3 fallos |
| 2.5 | 100 | 3.5 | PDF: 57/100 permanece en inicio; q-log: 19 fallos |
| 2.5 | 2000 | 4 | PDF: 99/100 permanece en inicio; q-log: 84 fallos |

Los cuatro ajustes acumularon aproximadamente 41 segundos de tiempo medido dentro de las llamadas al optimizador (2400 ajustes en esta máquina). La PDF y la CDF pasaron la verificación numérica contra Student-t en los tres escenarios. A pesar de que el optimizador de PDF indica éxito en los 100 casos con q=2.5 y N=2000, 99 estimaciones quedan exactamente en el punto inicial (q=1.6, b=2): **éxito del optimizador no equivale a estimación válida**. Para q=2.5 la construcción del histograma según el rango observado deja típicamente solo 3 a 4 bins ocupados. El q-log también sufre desbordamiento/infraflujo en la cola, además de su dependencia del histograma.

Este ensayo es un **diagnóstico de implementación**, no una comparación publicable de precisión: el esquema de bins y la función objetivo del q-log se deben definir mejor antes del estudio final. No interpretar sus RMSE condicionales a éxito como una clasificación de los métodos. Mantener el archivo CSV como línea de base al cambiar estas reglas.
