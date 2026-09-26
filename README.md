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

## Piloto sintético

El piloto centrado con cuatro métodos se ejecuta con
`python -m scripts.run_pilot --repetitions 100 --output results/pilot.csv`.
Véase `results/README.md` para el diagnóstico y las limitaciones detectadas.
