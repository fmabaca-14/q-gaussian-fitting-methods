# q-gaussian-fitting-methods
Comparison of q-Gaussian parameter estimation methods in complex systems

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
