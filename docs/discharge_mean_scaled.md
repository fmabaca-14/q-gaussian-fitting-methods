# Caudal: incrementos escalados por el promedio

Desde la raíz del repositorio, con los datos locales originales:

```bash
python -m scripts.analyze_discharge_mean_scaled \
  --discharge "data/raw/daily-data-mean.xls" \
  --bins 100
```

Produce `results/discharge_mean_scaled/` con:

- `metadata.json`: promedio de caudal, período, cantidad de muestras y huecos, autocorrelaciones de incrementos y valores absolutos (retardos 1, 5, 10, 30 días).
- `body_and_fit_summary.csv`: parámetros `b,q,mu`, distancia cuadrática integrada **respecto de la CDF empírica** (`dic_dfn`), KS, R² de la PDF de 100 bins, chi² descriptivo y log-verosimilitud promedio, además del error absoluto medio de excedencia en puntos porcentuales.
- `tail_threshold_errors.csv`: umbrales empíricos 1, 5, 10, 90, 95 y 99 %, probabilidades observadas y previstas y diferencia firmada en puntos porcentuales.
- `discharge_body.png`: PDF empírica y cuatro ajustes en escala lineal.
- `discharge_tails.png`: PDF de ambas colas y excedencias acumuladas en escala logarítmica.

Se ordena por fecha; `mean(Q)` incluye **todas** las observaciones válidas de caudal antes de formar pares y se omiten los saltos entre fechas no consecutivas. Las comparaciones usan `mu` libre. Directo y q-log emplean 100 bins; MLE y CDF no emplean bins en el ajuste, pero sus métricas de PDF se evalúan sobre los mismos 100 bins.

Las métricas describen ajustes a estos datos; KS aquí es una **distancia**, no un p-valor calculado bajo iid. Las autocorrelaciones descriptivas no sustituyen intervalos de confianza basados en bloques. `dic_dfn` integra con respecto a la distribución empírica, no con respecto a `dx`.

Los resultados nuevos se guardan en un directorio separado para no sobrescribir los del incremento simétrico acotado.
