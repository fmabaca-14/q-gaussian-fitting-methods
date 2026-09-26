# q-gaussian-fitting-methods
Comparison of q-Gaussian parameter estimation methods in complex systems

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
