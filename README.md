# q-gaussian-fitting-methods
Comparison of q-Gaussian parameter estimation methods in complex systems

## Protocolo principal del paper (octubre de 2026)

Los cuatro métodos usan una q-Gaussiana normalizada con **mu=0**. No se resta
la media ni la mediana de las muestras y no se aplica suavizado adicional.
Las nuevas corridas empíricas y sintéticas comparten los estimadores de
`src/empirical.py`; `src/estimators.py` conserva una interfaz compatible para
Monte Carlo y el protocolo anterior mediante `protocol="legacy"`.

- Viento solar: `2*(np[t+1]-np[t])/(np[t+1]+np[t])`, lag de una hora.
- Bitcoin: `log(close[t+1]/close[t])`, lag de un día.
- Caudal: `(Q[t+1]-Q[t])/mean(Q)`, lag de un día. La media se calcula sobre
  todas las observaciones limpias del archivo, antes de excluir pares con huecos.
- Se excluyen pares que cruzan tiempos faltantes.
- PDF directo: mínimos cuadrados de densidades con pesos uniformes, sobre
  bins ocupados de un histograma de ancho uniforme y rango completo.
- Q-log: método original con intercepto libre, ahora regresión contra `1,x²`.
  Usa errores uniformes en la PDF, propagados a `ln_q`: los pesos de los
  residuos cuadrados son `density**(2*q)`. No usa sigma Poisson en la regresión.
  Deriva b de la pendiente y la normalización, y selecciona q en la grilla
  1.02–2.98 (paso 0.01) minimizando **Pearson** sobre todos los bins.
  No es el estimador experimental que selecciona por SSE transformada.
- MLE y CDF: optimización con dos parámetros libres, b y q; mu queda fijado.

Los diagnósticos Poisson (`chi2_hist`) permanecen como medidas descriptivas:
son sumas totales, no chi-cuadrados reducidos ni tests calibrados para series
correlacionadas. Pesos uniformes en la PDF no implican pesos uniformes en
q-log ni igual importancia del cuerpo y las colas. Ningún cambio introduce
intervalos de confianza iid por defecto.

Controles históricos explícitos: `fit_histogram(..., centered=False,
weighting="poisson")`, `fit_unbinned(..., centered=False)`,
`fit_gaussian_cdf(..., centered=False)` y
`load_increments(..., discharge_transform="symmetric")`.
Los resultados previos guardados no fueron recalculados y no deben mezclarse
con las nuevas salidas `results/paper_*`.

## Comparación gaussiana y q-Gaussiana para una columna

Ambos modelos se ajustan mediante el mismo objetivo CDF de puntos medios,
con mu=0. El histograma de 100 bins solo se utiliza para visualizar la PDF y
calcular R² en todos sus centros, incluidos los bins vacíos.

```powershell
$data = "data/raw"
python -m scripts.compare_gaussian_qgaussian --solar "$data/1980-2024 np.txt" --bitcoin "$data/Bitcoin_15_10_2013-14_12_2013_historical_data_coinmarketcap.xls" --discharge "$data/daily-data-mean.xls" --bins 100
```

Genera tres figuras individuales en `results/paper_gaussian_comparison/`:
`solar_gaussian_comparison`, `bitcoin_gaussian_comparison` y
`discharge_gaussian_comparison`, cada una en PDF vectorial y PNG a 600 dpi.
Tamaño exacto 85 × 70 mm; ejes 9 pt, ticks/leyenda 8 pt; leyenda interior;
q-Gaussiana azul continua y Gaussiana naranja discontinua; ordenada logarítmica.
No se recorta el tamaño físico al exportar. `fits.csv`, `provenance.json` y
`captions.tex` registran resultados, tratamientos, tamaños y pies de figura.
Las únicas estadísticas anotadas en las curvas son q y R².

Para controles, `--free-mu` permite mu libre y `--discharge-transform
symmetric` o `difference` permiten otras definiciones del caudal. Cambiar
`--output-dir` para separar esas corridas del análisis principal.

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

## Piloto sintético centrado — nuevas corridas con protocolo del paper

Desde la raíz del repositorio, en el terminal de VS Code con el entorno
Python activado y `numpy`/`scipy` instalados:

```powershell
python -m scripts.run_pilot --repetitions 100
```

Se generan `results/paper_pilot_detail.csv` (2 400 ajustes),
`results/paper_pilot_summary.csv` (24 filas, una por escenario y método)
y `results/paper_pilot.json` (parámetros y versiones). Para una prueba rápida,
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
`results/paper_bins_sensitivity/`: `detail.csv`, `summary.csv`, `config.json`
y la figura `q_vs_bins.png`. Consultar el README de esa carpeta para
interpretar la figura y los limites de este experimento exploratorio.

Para la extension con 20 corridas y 10, 20, ..., 200 bins:

```powershell
python -m scripts.bins_sensitivity --repetitions 20 --bins-start 10 --bins-stop 200 --bins-step 10 --output-dir results/paper_bins_sensitivity_20
```

Esta corrida se guarda aparte en `results/paper_bins_sensitivity_20/`.

## Ajustes empíricos centrados

Los siguientes comandos usan el protocolo principal descrito arriba:
mu=0, caudal mean-scaled y errores uniformes en las regresiones PDF/q-log.
Los cuatro métodos optimizan objetivos diferentes. La aproximación de una
PDF por su valor en el centro puede ser frágil en bins muy amplios.

Desde la raíz del repositorio, con los tres archivos en `data/raw/`:

```powershell
$data = "data/raw"
python -m scripts.compare_real_methods --solar "$data/1980-2024 np.txt" --bitcoin "$data/Bitcoin_15_10_2013-14_12_2013_historical_data_coinmarketcap.xls" --discharge "$data/daily-data-mean.xls" --bins 50
python -m scripts.real_bins_sensitivity --solar "$data/1980-2024 np.txt" --bitcoin "$data/Bitcoin_15_10_2013-14_12_2013_historical_data_coinmarketcap.xls" --discharge "$data/daily-data-mean.xls" --bins-start 10 --bins-stop 200 --bins-step 5
```

El primer comando escribe `results/paper_empirical_50/fits.csv` y `fits.png`.
El segundo escribe `results/paper_empirical_bins/detail.csv`, `summary.csv`,
`references.csv` y `q_vs_bins.png`. `detail.csv` incluye `b`, `q`, `mu`,
bins ocupados, estado y objetivo para cada combinación. MLE y CDF se
calculan una vez por dataset y figuran como referencias independientes de
los bins. Los directorios de salida se pueden cambiar con `--output-dir`.

El gráfico de densidades muestra solo los percentiles 0.1 a 99.9 en el eje
horizontal, pero los ajustes usan **todos** los incrementos y el histograma
completo. `r2_hist`, `chi2_hist` y `cdf_max_abs` son diagnósticos descriptivos:
los errores Poisson y las comparaciones no corrigen dependencia temporal.
La desviación de q a través de distintos números de bins tampoco es un
intervalo de confianza. El caudal mean-scaled ya no tiene la cota artificial `(-2,2)`; esto no
garantiza que el modelo centrado reproduzca su asimetría o las colas.

Para generar dos figuras individuales por dataset usando 100 bins para el
histograma, PDF directo y q-log (MLE/CDF siguen sin bins):

```powershell
python -m scripts.plot_empirical_fits --solar "$data/1980-2024 np.txt" --bitcoin "$data/Bitcoin_15_10_2013-14_12_2013_historical_data_coinmarketcap.xls" --discharge "$data/daily-data-mean.xls" --bins 100
```

`results/paper_empirical_figures/` contendrá `solar_body.png`, `solar_tails.png`,
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

En `results/paper_empirical_scores/` se generan `global_evaluation.csv` (18 filas),
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

**Estado del análisis empírico (30/09/2026):** se mantiene el q-log original
de `src/empirical.py`, entonces con `mu` libre y selección por Pearson.
Desde octubre, el principal conserva Pearson pero fija `mu=0` y usa errores
uniformes en la PDF. La variante de SSE transformada se conserva como
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
uniformes y conserva errores Poisson propios del experimento histórico;
no sigue los pesos del protocolo principal actual. Ejemplo:

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
original = fit_histogram(x, bins=100, method="qlog", centered=False, weighting="poisson")
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

## Reproducción del protocolo sintético anterior

`--protocol legacy` en `scripts.run_pilot` y `scripts.bins_sensitivity` conserva
el estimador agrupado por deviance y el histograma con bins de cola variables.
Para reproducir exactamente una corrida antigua, recuperar además su grilla q
y configuración en el commit correspondiente. Las nuevas corridas usan el
protocolo del paper por defecto y tienen otras rutas de salida.

La comprobación de ejecución sintética con una muestra por escenario es
solamente un control computacional, no una nueva validación Monte Carlo.
Con colas muy pesadas y histogramas de rango completo pueden quedar pocos
bins ocupados y q-log no producir candidatos válidos: las fallas deben
reportarse y no excluirse silenciosamente de la validación.
