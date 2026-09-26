# Ruido aditivo del 50 %: experimento exploratorio

Desde la raíz del repositorio:

```bash
python -m scripts.noise_sensitivity --noise-fraction 0.5 --output-dir results/noise_50pct
```

Se generan 10 muestras independientes por escenario, con q = 1.5 o 1.9,
b = 2 o 5, N = 2000, mu = 0 y 50 bins. En cada repetición se ajusta la misma
muestra latente antes y después de sumarle ruido gaussiano independiente
N(0, sigma^2), donde sigma = 0.5 / sqrt(b * (3 - q)). La semilla y las
muestras latentes coinciden con las del experimento del 10 %; también se
conservan las realizaciones normales estandarizadas del ruido. El 50 %
se refiere a la escala t equivalente, no a la desviación estándar de
la q-Gaussiana (infinita para q = 1.9).

`detail.csv` tiene un ajuste por repetición, condición y método; `summary.csv`
contiene medias, sesgos, desvíos, RMSE y errores Monte Carlo; `paired_changes.csv`
compara cada ajuste con su versión sin ruido; `disagreement.csv` mide el rango
de las cuatro estimaciones de cada repetición; `disagreement_change.csv` incluye
el cambio pareado y un intervalo t exploratorio. `comparison_10_50.csv` reúne
la separación entre métodos de los ensayos del 10 % y 50 %.

Con ruido del 50 %, los promedios de q y b bajan respecto de los parámetros
generadores. La separación media entre métodos en q sube de aproximadamente
0.04–0.06 (sin ruido) a 0.08–0.10. La separación en b dividida por b verdadero
baja, pero esto no significa mejor recuperación: las estimaciones de b de
todos los métodos se alejan hacia abajo del valor verdadero. Los cuatro
intervalos exploratorios del cambio pareado del rango de q excluyen cero;
solo hay diez repeticiones por escenario, por lo que no constituyen una prueba
definitiva sobre el mecanismo de las discrepancias de datos reales.

El ajuste supone una q-Gaussiana pura aunque los datos con ruido son una
convolución q-Gaussiana–Gaussiana. Sus parámetros ajustados pueden entenderse
como parámetros efectivos de un modelo mal especificado. Para contrastar la
hipótesis sobre datos reales habrá que justificar una amplitud y una forma
plausible del ruido instrumental, o comparar mecanismos alternativos.
