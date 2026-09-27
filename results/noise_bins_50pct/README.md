# Sensibilidad a bins con ruido del 50 %

Reproducir desde la raíz:

```bash
python -m scripts.noise_bins_sensitivity --noise-fraction 0.5 --output-dir results/noise_bins_50pct
```

Se reutilizan las muestras latentes y los ruidos normales estandarizados
del análisis de 10 %, multiplicando la desviación estándar del ruido por
cinco: `sigma = 0.5 / sqrt(b*(3-q))`. Hay diez muestras por combinación de
q = 1.5/1.9 y b = 2/5, N = 2000, mu = 0. PDF y qlog se ajustan con
10, 20, ..., 200 bins a cada muestra limpia y ruidosa. MLE y CDF se
ajustan una vez por muestra y condición como referencias sin bins.

`detail.csv` y `summary.csv` guardan los ajustes y sus métricas por
escenario, condición, método y bins; `paired_changes.csv` da las diferencias
entre versiones ruidosa y limpia. `analysis.csv` resume 10/50/100/200 bins
y la amplitud de la media a través de los 20 números de bins. Las referencias
están en `references_detail.csv` y `references_summary.csv` y las figuras
en `q_vs_bins.png` y `b_vs_bins.png`.

Hubo un fallo entre 3200 ajustes histogramados: qlog, q=1.9, b=5,
repetición 3, condición ruidosa, 10 bins. La media en ese punto usa nueve
ajustes válidos, mientras el resto usa diez. No hubo parámetros al límite;
las 160 estimaciones MLE/CDF fueron válidas. Los resultados a 50 bins
reproducen `results/noise_50pct`.

El ruido del 50 % sesga hacia abajo q y b para PDF y las referencias, y
amplifica la dependencia de PDF respecto a los primeros bins: para q=1.5,
la diferencia entre la media con 10 bins y con 200 bins es ~0.055 en q,
frente a ~0.015–0.02 con ruido del 10 %. Desde ~40 bins PDF cambia poco,
pero converge a parámetros efectivos sesgados, no a los verdaderos.
qlog muestra en 10 bins q ~1.92–2.36 (muy por encima del valor verdadero)
igual que antes; en bins intermedios sus q pueden acercarse accidentalmente
al valor real mientras b sigue subestimado. La sensibilidad de qlog a bins
ya existe en las muestras sin ruido.

El ruido es gaussiano aditivo iid; los datos observados corresponden a una
convolución y no a una q-Gaussiana exacta. Que un método recupere q por
cancelación de sesgos no demuestra que modele correctamente el ruido.
