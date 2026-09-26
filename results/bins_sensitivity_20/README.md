# Sensibilidad a bins: 20 corridas Monte Carlo

Comando desde la raiz del repositorio:

```powershell
python -m scripts.bins_sensitivity --repetitions 20 --bins-start 10 --bins-stop 200 --bins-step 10 --output-dir results/bins_sensitivity_20
```

Escenarios: q=1.5/1.9, b=2/5, N=1000/5000, mu=0. Hay 20 muestras por
escenario y cada muestra se reutiliza para 20 numeros de bins y los dos
metodos con histograma (PDF y q-log). Total: 6400 ajustes. La semilla de
una muestra no depende del numero de bins; las primeras diez muestras de
cada escenario coinciden con la primera corrida (semilla 20260926).

Archivos:

- `detail.csv`: una fila por muestra, numero de bins y metodo (6400 filas).
- `summary.csv`: bias, RMSE, dispersion y error Monte Carlo de ambos
  parametros, fallos, limites, bins ocupados y tiempo (320 filas).
- `q_vs_bins.png`: media de q y banda de una desviacion estandar entre
  muestras; linea discontinua para q verdadero. La banda no es un intervalo
  de confianza de la media.
- `config.json`: factores, grilla q-log, semilla y versiones.

El PDF evalua las probabilidades integradas de todos los bins. El q-log
regresa la densidad estimada de bins ocupados en funcion de x**2 para cada
q candidato; los conteos de colas estan incluidos, pero los bins mas
anchos aproximan su densidad en el centro. Con diez bins hay ocho bins
centrales y uno por cola. No se trunca ningun valor extremo.

Durante la revision se detectaron minimos locales falsos en algunos
ajustes PDF que el optimizador habia marcado como convergentes. La
implementacion ahora compara el resultado con un conjunto prefijado de
puntos iniciales y reinicia cuando uno tiene menor deviance. Se
reestimaron todos los ajustes PDF sobre las mismas muestras; el q-log
no cambio. Las filas finales ya incorporan esa correccion.

Hallazgos exploratorios: no hubo fallos ni soluciones en limites. PDF
mostro poca sensibilidad de q a los bins; q-log sobreestimo q
considerablemente con 10-20 bins y se aproximo al valor verdadero
alrededor de 50-100. Con mas bins, q-log no mejoro necesariamente:
el sesgo de b se hizo mas negativo en varios escenarios. Elegir bins
solo por el menor RMSE observado en 20 muestras sobreajustaria esta
simulacion; el estudio final debe fijar una regla o validar la seleccion
en nuevas repeticiones independientes.
