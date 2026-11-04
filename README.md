# Compensación SMCP — PINN y GPR

Compensación de errores del sistema de medición por coordenadas SMCP (cámara–proyector, luz estructurada) con un modelo físico + red neuronal (PINN) y + proceso gaussiano (GPR), usando un artefacto de 3 esferas calibrado en MMC.

Estancia profesional en CENAM (Querétaro). Contexto completo, reglas de validación y decisiones: [`CLAUDE.md`](CLAUDE.md).

## Estructura

```
scripts/
  procesa_30_escenas_smcp.py   segmentación RANSAC + ajuste de esferas (rechazo 2.5σ)
  pinn_smcp.py                 PINN v2: p + p·s(z,ρ) + MLP(p), k-fold por escaneo
  gp_smcp.py                   GPR: física s(z,ρ) + GP (lineal + Matérn 5/2 ARD)  ← modelo actual
  exporta_modelo_yml.py        modelo_gp_smcp.npz → .yml para SMCP.exe (C++/OpenCV)
  valida_cpp_vs_python.py      compara la corrección de SMCP.exe contra Python
resultados/
  esferas/                     resultados_30_escenas.* (diámetros y distancias medidos)
  pinn_v2/                     resultados PINN v2 + comparacion_smcp.xlsx
  gp/                          modelo_gp_smcp.{npz,mat,yml}, resultados y log k-fold
archivo/
  pinn_v1/                     PINN v1 (~222 µm) — histórico, no usar
```

## Uso

Los escaneos crudos (`.xyz`, ~14 GB) no están en el repo. En cada script solo se edita `CARPETA_XYZ` (o `CARPETA`); las salidas se escriben en `resultados/` con rutas relativas al repo.

1. `python scripts/procesa_30_escenas_smcp.py` → `resultados/esferas/`
2. `python scripts/pinn_smcp.py` (torch) → `resultados/pinn_v2/`
3. `python scripts/gp_smcp.py` (~30 min) → `resultados/gp/`
4. `python scripts/exporta_modelo_yml.py` → `resultados/gp/modelo_gp_smcp.yml` (editar `CALIBRACION` y `XYZ_PRUEBA`)

`pinn_smcp.py` y `gp_smcp.py` comparten el caché `resultados/esferas/nubes_segmentadas.npz` (se genera solo, ignorado por git).

Dependencias: numpy, scipy, pandas, openpyxl; torch solo para `pinn_smcp.py`.

## Resultados (37 escaneos, k-fold por escaneo, k=5)

| Corrección | RMS global | Mejora vs escala fija |
|---|---|---|
| Crudo | 1708 µm | — |
| Escala fija | 378 µm | — |
| PINN v2 | 97.9 µm | 74.1 % |
| **GP** | **55.4 µm** | **85.3 %** |
