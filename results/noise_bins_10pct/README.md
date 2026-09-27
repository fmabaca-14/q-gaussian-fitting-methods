# Sensibilidad a bins con ruido del 10 %

Ejecutar desde la raíz del repositorio:

```bash
python -m scripts.noise_bins_sensitivity
```

Se usan las mismas 10 muestras latentes y las mismas realizaciones del ruido del
experimento `results/noise_10pct`: q verdadero = 1.5 o 1.9; b verdadero = 2 o 5;
N = 2000; mu = 0. El ruido es gaussiano iid con desviación estándar igual a
`0.1 / sqrt(b*(3-q))`. El 10 % es relativo a la escala t equivalente, no a
la desviación estándar de la q-Gaussiana. Para cada muestra limpia y ruidosa
se ajustan PDF y qlog con 10, 20, ..., 200 bins; MLE y CDF se evalúan una
vez por muestra y condición como referencias independientes de bins.

`detail.csv` contiene 3200 ajustes con histogramas. `summary.csv` contiene
320 resúmenes (media, sesgo, desvío, RMSE, fallos) y `analysis.csv` resume
10, 50, 100 y 200 bins y el rango de las medias a lo largo del barrido.
`paired_changes.csv` registra el efecto del ruido para cada cantidad de bins.
`references_detail.csv` y `references_summary.csv` guardan las referencias.
Las figuras `q_vs_bins.png` y `b_vs_bins.png` muestran las medias; no muestran
incertidumbre, que está tabulada en `summary.csv`.

No hubo fallos ni soluciones en el límite (3200/3200 ajustes válidos con bins;
160/160 referencias válidas). A 50 bins, los parámetros del PDF y qlog
coinciden con el ensayo previo con ruido del 10 %.

El PDF varía relativamente poco entre 10 y 200 bins. En cambio, qlog tiene
un fuerte error sistemático a 10 bins: la media de q estimado va de 1.96 a
2.40 según el escenario, aun sin ruido. Entre aproximadamente 40 y 100 bins
se aproxima mucho más a los valores generadores, y después q y, especialmente,
b pueden volver a desviarse. El ruido del 10 % desplaza las curvas mucho menos
que la elección de bins. Este resultado concierne a esta construcción concreta
de histograma, esta parametrización y diez muestras; no identifica un número
universal de bins óptimo ni explica por sí solo las discrepancias reales.
