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
