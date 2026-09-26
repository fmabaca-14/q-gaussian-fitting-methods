# Sensibilidad a los bins: experimento exploratorio

**Nota:** esta primera corrida con 10 muestras conserva la version anterior
del optimizador PDF, que en algunas combinaciones de bins acepto minimos
locales espurios. Para comparar tendencias utilizar preferentemente la
corrida corregida `../bins_sensitivity_20/`.

Ejecutar `python -m scripts.bins_sensitivity` desde la raiz del repositorio.
Por cada escenario (q=1.5/1.9; b=2/5; N=1000/5000) se generaron diez
muestras independientes. Cada muestra se reutilizo para los ocho valores
de bins (20, 30, 40, 50, 70, 100, 150, 200) y ambos metodos (PDF y q-log).
Los 1280 ajustes se guardan en `detail.csv`; las 128 combinaciones de
escenario, bins y metodo se resumen en `summary.csv`.

La figura muestra la media de q estimado y una banda de una desviacion
estandar **entre las 10 muestras**, junto al q verdadero. La banda no es
un intervalo de confianza de la media. `summary.csv` proporciona error
Monte Carlo del bias y RMSE, dispersion, b estimado, fallos, bins ocupados
y tiempo. Diez repeticiones alcanzan para diagnosticar tendencias, no para
seleccionar un numero optimo de bins con precision.

La construccion de bins es la misma del piloto previo: 80 % cubre el
intervalo central fijado por el percentil 90 de |x| y 20 % cubre las colas
hasta el maximo observado. El numero de bins cambia tambien el ancho de
los bins centrales y la resolucion de colas. Ninguna observacion se elimina.
El PDF integra la probabilidad de cada bin; el q-log transforma la densidad
estimada en el centro de los bins ocupados. Por tanto, esta figura mide
sensibilidad conjunta a esa regla de bordes y al numero de bins.

En esta corrida no hubo fallos ni soluciones en limites. El PDF fue
comparativamente estable en q entre 20 y 200 bins. El q-log sobreestimo
q con 20 bins, mejoro alrededor de 50-100 y en algunos escenarios volvio
a aumentar dispersión y sesgo de b hacia 200 bins. No elegir el minimo
RMSE observado entre ocho valores basandose solo en diez muestras.
