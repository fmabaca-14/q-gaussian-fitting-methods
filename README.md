# q-gaussian-fitting-methods
Comparison of q-Gaussian parameter estimation methods in complex systems

## Comparación gaussiana y q-Gaussiana

`src/gaussian.py` ofrece `fit_gaussian_cdf(x)` con mu libre y sigma positiva.
La comparación usa ajuste CDF para ambos modelos, con el mismo objetivo de
error cuadrático frente a la CDF empírica de puntos medios. Para generar la
figura con los tres datasets desde PowerShell:

```powershell
$data = "data/raw"
python -m scripts.compare_gaussian_qgaussian --solar "$data/1980-2024 np.txt" --bitcoin "$data/Bitcoin_15_10_2013-14_12_2013_historical_data_coinmarketcap.xls" --discharge "$data/daily-data-mean.xls" --bins 100
```

Escribe `results/gaussian_comparison/gaussian_vs_qgaussian.png` y `fits.csv`.
La figura muestra densidades en escala logarítmica y las únicas estadísticas
anotadas son R² y q. Ambos R² usan todos los centros del mismo histograma de
100 bins (o el número indicado por `--bins`), aunque ambos modelos se ajustan
sin bins. Se muestra todo el rango observado. Un R² mayor describe ese
histograma; no demuestra que CDF sea el mejor estimador ni constituye un test
de selección de modelos.

Por defecto se conserva el incremento simétrico del caudal utilizado por
`load_increments`. Agregá `--discharge-transform difference` para usar
Q[t+1]-Q[t], o `--discharge-transform mean-scaled` para dividir esa diferencia
por el caudal medio. Se omiten pares que cruzan días faltantes. El CSV registra
la transformación elegida; evitá comparar resultados con tratamientos distintos.

## Series reales

`src/real_data.py` lee los tres formatos originales y devuelve tablas ordenadas
cronológicamente con índice temporal. La carga y la visualización no calculan
incrementos ni alteran unidades. Los datos originales se colocan localmente en
`data/raw/` (ignorados por Git); los archivos `.xls` adjuntos son **texto CSV**,
no libros de Excel. Desde la raíz del repositorio, con `pandas`, `numpy`,
`scipy` y `matplotlib` instalados:

```bash
python -m scripts.plot_real_data \
  --solar "data/raw/1980-2024 np.txt" \
  --bitcoin "data/raw/Bitcoin_15_10_2013-14_12_2013_historical_data_coinmarketcap.xls" \
  --discharge "data/raw/daily-data-mean.xls"
```

Agregá `--output results/real_series.png` para guardar la figura. En Jupyter:

```python
from src.real_data import load_solar_wind, load_bitcoin, load_discharge, plot_series

solar = load_solar_wind("data/raw/1980-2024 np.txt")
bitcoin = load_bitcoin("data/raw/Bitcoin_15_10_2013-14_12_2013_historical_data_coinmarketcap.xls")
discharge = load_discharge("data/raw/daily-data-mean.xls")
fig, axes = plot_series(solar, bitcoin, discharge)
```

Ejecutá el notebook desde la raíz del repositorio para que `src` esté en el
camino de importación. El viento solar conserva `np` y `ErrNp`, eliminando
`np=999.9` y `ErrNp=0/999.9`; Bitcoin usa `timeClose` en UTC; el caudal usa
fechas diarias y conserva `unit_of_measure`. Los archivos aquí proporcionados
abarcan 1980–2024, 2012–2026 y 2016–2026 respectivamente; el nombre del
archivo de Bitcoin no describe su intervalo real. Si se usan para inferencia,
definir después los incrementos, lagunas y posibles dependencias temporales.

## Piloto sintético centrado

Desde la raíz del repositorio, en el terminal de VS Code con el entorno
Python activado y `numpy`/`scipy` instalados:

```powershell
python -m scripts.run_pilot --repetitions 100
```

Se generan `results/pilot_v2_detail.csv` (2 400 ajustes),
`results/pilot_v2_summary.csv` (24 filas, una por escenario y método)
y `results/pilot_v2.json` (parámetros y versiones). Para una prueba rápida,
usar `--repetitions 2`. El código del primer piloto sigue disponible en el
historial Git; sus archivos `results/pilot.csv` y `results/pilot.json` se
conservan como referencia. Véase `results/README.md` para limitaciones.

## Sensibilidad a la cantidad de bins

```powershell
python -m scripts.bins_sensitivity
```

Genera 10 muestras por escenario con q=1.5/1.9, b=2/5 y N=1000/5000.
Cada muestra se ajusta con 20, 30, 40, 50, 70, 100, 150 y 200 bins,
compartiendo los datos entre PDF y q-log. Los resultados estan en
`results/bins_sensitivity/`: `detail.csv`, `summary.csv`, `config.json`
y la figura `q_vs_bins.png`. Consultar el README de esa carpeta para
interpretar la figura y los limites de este experimento exploratorio.

Para la extension con 20 corridas y 10, 20, ..., 200 bins:

```powershell
python -m scripts.bins_sensitivity --repetitions 20 --bins-start 10 --bins-stop 200 --bins-step 10 --output-dir results/bins_sensitivity_20
```

Esta corrida se guarda aparte en `results/bins_sensitivity_20/`.

## Ajustes empíricos con ubicación libre

Los análisis siguientes están separados del piloto sintético centrado de
`src/estimators.py`. Usan `src/real_data.py` para leer los archivos y
`src/distributions.py` para PDF/CDF normalizadas. En `src/empirical.py`,
las variables ajustadas son, sin recentrado ni cambio de escala:

- Viento solar: `2*(np[t+1]-np[t])/(np[t+1]+np[t])`, entre horas consecutivas.
- Bitcoin: `log(close[t+1]/close[t])`, entre días consecutivos.
- Caudal: `2*(Q[t+1]-Q[t])/(Q[t+1]+Q[t])`, entre días consecutivos.

Se excluyen los pares que cruzan huecos de tiempo después de la limpieza de
datos. Los parámetros `b`, `q` y `mu` son libres, con `b>0` y `1<q<3`.
El ajuste directo reproduce el histograma de **ancho uniforme** en todo el
rango, omite bins vacíos y pondera las densidades con
`sqrt(count)/(N*width)`. MLE optimiza la log-verosimilitud individual; CDF
minimiza el error cuadrático frente a la CDF empírica. Q-log recorre q con
paso 0.01 y ajusta `ln_q(density)` frente a `1, x, x²`. Para cada q minimiza
el error ponderado J_reg de esa regresión, selecciona el menor J_reg de toda
la grilla y sólo entonces recupera b de la pendiente y la normalización;
mu se obtiene del vértice. No calcula conteos esperados para elegir q.
Los bins amplios en colas
hacen que la aproximación en centros de bin del q-log sea especialmente
frágil. Estos cuatro métodos optimizan **objetivos diferentes**.

**Cambio de criterio (03/10/2026):** el q-log empírico ahora selecciona por
J_reg, usando los errores Poisson propagados ya presentes. No se ha modificado
la ubicación libre ni la ponderación. `objective` guarda ese J_reg; ya no
contiene el score de Pearson. Los scripts existentes que llaman al q-log
usan el nuevo criterio y sus resultados deben regenerarse en otra carpeta
para compararlos con las salidas anteriores. Para reproducir la selección
anterior de conteos se puede especificar:

```python
legacy = fit_histogram(x, bins=100, method="qlog", qlog_selection="pearson_counts")
current = fit_histogram(x, bins=100, method="qlog", qlog_selection="regression")
```

El criterio compara errores de regresiones cuya transformación y varianza
cambian con q; no es una verosimilitud común. El intercepto sigue libre y
puede no ser compatible con el de la PDF normalizada reconstruida.

Desde la raíz del repositorio, con los tres archivos en `data/raw/`:

```powershell
$data = "data/raw"
python -m scripts.compare_real_methods --solar "$data/1980-2024 np.txt" --bitcoin "$data/Bitcoin_15_10_2013-14_12_2013_historical_data_coinmarketcap.xls" --discharge "$data/daily-data-mean.xls" --bins 50
python -m scripts.real_bins_sensitivity --solar "$data/1980-2024 np.txt" --bitcoin "$data/Bitcoin_15_10_2013-14_12_2013_historical_data_coinmarketcap.xls" --discharge "$data/daily-data-mean.xls" --bins-start 10 --bins-stop 200 --bins-step 5
```

El primer comando escribe `results/empirical_50/fits.csv` y `fits.png`.
El segundo escribe `results/empirical_bins/detail.csv`, `summary.csv`,
`references.csv` y `q_vs_bins.png`. `detail.csv` incluye `b`, `q`, `mu`,
bins ocupados, estado y objetivo para cada combinación. MLE y CDF se
calculan una vez por dataset y figuran como referencias independientes de
los bins. Los directorios de salida se pueden cambiar con `--output-dir`.

El gráfico de densidades muestra solo los percentiles 0.1 a 99.9 en el eje
horizontal, pero los ajustes usan **todos** los incrementos y el histograma
completo. `r2_hist`, `chi2_hist` y `cdf_max_abs` son diagnósticos descriptivos:
los errores Poisson y las comparaciones no corrigen dependencia temporal.
La desviación de q a través de distintos números de bins tampoco es un
intervalo de confianza. La variable simétrica de caudal pertenece a `(-2,2)`;
la q-Gaussiana normalizada asigna probabilidad fuera de ese soporte.

Para generar dos figuras individuales por dataset usando 100 bins para el
histograma, PDF directo y q-log (MLE/CDF siguen sin bins):

```powershell
python -m scripts.plot_empirical_fits --solar "$data/1980-2024 np.txt" --bitcoin "$data/Bitcoin_15_10_2013-14_12_2013_historical_data_coinmarketcap.xls" --discharge "$data/daily-data-mean.xls" --bins 100
```

`results/empirical_figures/` contendrá `solar_body.png`, `solar_tails.png`,
`bitcoin_body.png`, `bitcoin_tails.png`, `discharge_body.png` y
`discharge_tails.png`. La figura del cuerpo usa escala lineal y muestra el
intervalo de los percentiles 1–99 para apreciar el pico. La figura de las
colas usa escala logarítmica desde cada umbral empírico unilateral del 10 %
hasta el extremo observado: arriba muestra la misma PDF de 100 bins, abajo
la probabilidad de excedencia empírica y modelada. No se reajusta ni
renormaliza la distribución al graficar la cola. Las barras de error de la
PDF son Poisson descriptivas y no incorporan dependencia temporal.

## Evaluación descriptiva del cuerpo y las colas

`src/empirical_scores.py` implementa métricas comunes para los parámetros
producidos por los cuatro ajustes empíricos. Desde la raíz del proyecto:

```powershell
$data = "data/raw"
python -m scripts.evaluate_real_methods --solar "$data/1980-2024 np.txt" --bitcoin "$data/Bitcoin_15_10_2013-14_12_2013_historical_data_coinmarketcap.xls" --discharge "$data/daily-data-mean.xls"
```

En `results/empirical_scores/` se generan `global_evaluation.csv` (18 filas),
`tail_threshold_errors.csv` (6 umbrales por ajuste), `tail_summary.csv`,
`exceedance_curves.csv`, `exceedance_curves.png` y `config.json`. PDF y q-log
se ajustan con 50 y 100 bins; MLE y CDF, una vez por dataset. Todos se evalúan
sobre **la misma muestra** y con un histograma común de 100 bins para R² y χ².

La distancia integrada reportada es `dic_dfn = mean((F_model(x_i)-F_n(x_i))²)`:
integra respecto de la distribución empírica, **no respecto de dx**. Para
valores repetidos se usa la CDF empírica continua por la derecha. `ks` toma
el máximo de las diferencias a ambos lados de sus saltos. Las seis cotas de
cola se fijan por dataset en los percentiles 1, 5, 10, 90, 95 y 99;
`error_pp = 100*(probabilidad_modelo-probabilidad_observada)`, con `X<=u`
a la izquierda y `X>u` a la derecha. `tail_mae_pp` promedia sus seis errores
absolutos. Las curvas muestran la probabilidad de excedencia derecha.

Estas comparaciones son **dentro de la misma muestra** usada para estimar
parámetros: la distancia CDF puede favorecer al ajuste CDF. Los umbrales son
cuantiles de esa misma muestra; las dependencias temporales y posibles cambios
de régimen siguen requiriendo análisis adicional. `r2_hist_100` depende de la
elección del histograma y no tiene unidades probabilísticas.

## Variante q-log centrada con selección por error ponderado

**Estado histórico del análisis empírico (30/09/2026):** se mantuvo el q-log original
de `src/empirical.py`, con `mu` libre y selección por Pearson, como método
principal de la comparación. Esa selección ahora se puede reproducir con
`qlog_selection="pearson_counts"`; desde el 03/10 el valor predeterminado es
J_reg. La variante centrada se conserva como
implementación experimental para estudiar sus diferencias.

La comparación con los tres datasets y el barrido de 10 a 200 bins, en pasos
de 5, mostró mayor variabilidad de los parámetros de la variante nueva y
saltos pronunciados para ciertos números de bins. En viento solar, con
190 bins, seleccionó `q=2.98` (límite superior de la grilla) y
`b≈8.2e-306`. En caudal, usando `Q[t+1]-Q[t]` sin normalizar, también
aparecieron soluciones de borde con `b` extremadamente pequeño.
Estos resultados no se interpretan como estimaciones físicas confiables.
La evaluación de colas con 50, 100 y 150 bins tampoco mostró una mejora
general de la variante nueva.

Hipótesis a investigar: cambios de ocupación y posición de los bins;
selección entre regresiones cuya transformación y ponderación dependen de
`q`; incompatibilidad entre el intercepto libre y la normalización de la
PDF reconstruida; y amplificación de cambios de pendiente al convertirla
a `b`, pues esa conversión contiene el exponente `2/(3-q)`.
El centrado impuesto también puede afectar series asimétricas o desplazadas.
Esta prueba cambia simultáneamente `mu` y el criterio de selección de `q`,
por lo que no identifica por separado sus efectos. Las causas propuestas
requieren pruebas adicionales; la sensibilidad a bins no es un intervalo
de confianza.

`src/qlog_variants.py` agrega `fit_centered_qlog`: impone `mu=0` sin restar la
media a las observaciones, ajusta `ln_q(density)` frente a `1,x²` y selecciona
q por el menor `sum((residual/sigma_lnq)²)` en la grilla. Usa los mismos bins
uniformes y errores Poisson que el ajuste empírico existente. Ejemplo:

```python
import pandas as pd
from src.empirical import fit_histogram, load_increments
from src.qlog_variants import fit_centered_qlog
from src.empirical_scores import body_scores, tail_scores, tail_thresholds

datasets = load_increments(
    "data/raw/1980-2024 np.txt",
    "data/raw/Bitcoin_15_10_2013-14_12_2013_historical_data_coinmarketcap.xls",
    "data/raw/daily-data-mean.xls",
)
x = datasets["bitcoin"]["x"]
original = fit_histogram(x, bins=100, method="qlog", qlog_selection="pearson_counts")
centered, scan = fit_centered_qlog(x, bins=100, return_scan=True)
scan_table = pd.DataFrame(scan)
thresholds = tail_thresholds(x)
for result in (original, centered):
    if result.success:
        print(result)
        print(body_scores(x, result))
        print("Tail MAE (pp):", tail_scores(x, result, thresholds)[1])
```

El intercepto se ajusta libremente; b se deriva de la pendiente usando la
normalización. `normalization_intercept_gap` registra la diferencia con el
intercepto de la PDF normalizada reconstruida. Por eso `weighted_sse` mide
el error de la regresión, no necesariamente el de esa PDF. Los errores y la
transformación cambian con q; el mínimo entre valores de q es un criterio
experimental y no garantiza el mejor ajuste de las colas. Esta comparación
cambia tanto la ubicación como el criterio de selección respecto del q-log
original, por lo que no permite atribuir diferencias a un solo factor.
