# Prueba exploratoria: ruido gaussiano aditivo del 10 %

Ejecutar desde la raíz: `python -m scripts.noise_sensitivity`.
Los escenarios son q=1.5/1.9, b=2/5, N=2000, mu=0 y 10 réplicas.
La condición ruidosa comparte la muestra latente con la limpia y suma
ruido independiente por observación: epsilon ~ N(0, sigma_epsilon**2),
con sigma_epsilon = 0.10 / sqrt(b*(3-q)). Esta escala es la de la
Student-t equivalente, **no** la desviación estándar de X (para q=1.9,
la varianza de X no existe). Ambos conjuntos se ajustan con PDF, MLE,
q-log y CDF; PDF y q-log usan 50 bins.

Archivos: `detail.csv` (320 ajustes), `summary.csv` (32 resúmenes de
parámetros, sesgo, RMSE, dispersión y error Monte Carlo),
`paired_changes.csv` (16 cambios ruidosa menos limpia por método),
`disagreement.csv` (rango máximo-mínimo de las estimaciones de los
cuatro métodos en cada escenario), `disagreement_change.csv` (cambio
pareado en ese rango, con intervalos t exploratorios),
`disagreement.png` y `config.json`.

Ningún ajuste falló ni terminó exactamente en un límite. La media del
rango entre métodos de q cambió entre -0.0014 y +0.0026; los intervalos
t del cambio pareado incluyen cero en los cuatro escenarios. Los
rangos relativos de b tampoco crecieron sistemáticamente. El ruido
desplazó levemente b hacia abajo, pero no generó la gran discrepancia
entre métodos observada en algunos datos reales.

Este resultado evalúa **solo** ruido aditivo gaussiano iid de esta
amplitud y diez repeticiones. No descarta ruido dependiente de la señal,
errores correlacionados, mezcla de regímenes, no estacionariedad o
desviaciones respecto a una q-Gaussiana única. Tampoco permite rechazar
la hipótesis general de que errores de medición contribuyen a las
discrepancias reales; para eso hará falta una familia de mecanismos
y amplitudes plausibles, con más repeticiones.
